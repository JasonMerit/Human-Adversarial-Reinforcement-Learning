import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .buffers import PrioritizedReplayBuffer, ReplayBuffer
from .utils import TimerRegistry
from rich import print

class KnegtNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        channels, size, _ = obs_shape
        self.n_actions = n_actions

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, size, size)
            n_flatten = self.cnn(dummy).shape[1]

        hidden = 32
        self.value_head = nn.Sequential(
            nn.Linear(n_flatten, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.adv_head = nn.Sequential(
            nn.Linear(n_flatten, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

        self.opp_head = nn.Sequential(
            nn.Linear(n_flatten, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x) -> torch.Tensor:
        # assert x.shape == (5, 3, 15, 15) or x.shape == (7, 3, 15, 15), f"Expected input shape (5, 3, 15, 15) or (1, 3, 15, 15), got {x.shape}"
        h = self.cnn(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
    
    def predict(self, x) -> torch.Tensor:
        h = self.cnn(x)
        logits = self.opp_head(h)
        return logits

    @torch.no_grad()  # Called in play
    def act(self, obs):
        assert obs.ndim == 4, f"Expected input shape (B, C, H, W), got {obs.shape}"
        q = self.forward(obs)
        return torch.argmax(q, dim=1).cpu().numpy().item()
    
    @classmethod
    def from_checkpoint(cls, path, obs_shape, n_actions, device="cpu"):
        net = cls(obs_shape=obs_shape, n_actions=n_actions).to(device)
        net.load_state_dict(torch.load(path, weights_only=True, map_location=device))        
        return net

class KnegtAgent:

    def __init__(self, 
        obs_shape, n_actions, state_example: tuple, 
        state_encode_fn, args, device, writer, player
        ):

        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.q_network = KnegtNetwork(obs_shape, n_actions).to(device)
        self.target_network = KnegtNetwork(obs_shape, n_actions).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)

        self.rb = PrioritizedReplayBuffer(state_example, state_encode_fn, player, args, device) if args.per else ReplayBuffer(state_example, state_encode_fn, player, args, device)
        
        self.player = player  # Keep track of which agent 
        self.opp_weight = 0.1  # Weight for opponent prediction loss
        self.name = ["A", "B"][player]
        self.writer = writer
        self.learning_steps = 0

    def get_num_params(self):
        return sum(p.numel() for p in self.q_network.parameters())
    
    def save(self, path, verbose=True):
        torch.save(self.q_network.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
    
    @torch.no_grad()
    def act(self, obs: np.ndarray):  # Legacy use select_action instead
        assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
        q = self.q_network(torch.as_tensor(obs).to(self.device))
        return torch.argmax(q, dim=1).cpu().numpy()
    
    @TimerRegistry.wrap_fn("rainbow_learn")
    def learn(self):
        self.learning_steps += 1

        obs, actions, rewards, next_obs, dones, weights, indices = self.rb.sample(self.batch_size)
        # print(obs[:, self.player].shape)  # [7, 3, 15, 15]
        # print(actions.shape)  # (7, 2)
        dqn_loss_per_sample = self._dqn_loss(obs[:, self.player], actions[:, self.player], rewards, next_obs[:, self.player], dones)
        # print(dqn_loss_per_sample.shape)  # [B]
        cross_entropy_loss_per_sample = self._cross_loss(obs[:, 1 - self.player], actions[:, 1 - self.player], dones)
        loss_per_sample = dqn_loss_per_sample + self.opp_weight * cross_entropy_loss_per_sample
        loss = (loss_per_sample * weights.squeeze()).mean()

        # update priorities
        new_priorities = dqn_loss_per_sample.detach().cpu().numpy()  # Prioritize based on TD error only, not opponent prediction loss
        self.rb.update(indices, new_priorities)

        self.optimizer.zero_grad()
        loss.backward()

        # Nudged between backward and optimizer for grad logging
        if self.writer and self.learning_steps % 100 == 0:
            grad, weight = self.grad_weight_norm()
            self.writer.add_scalar(f"gradients/{self.name}_grad_norm", grad, self.learning_steps)
            self.writer.add_scalar(f"gradients/{self.name}_weight_norm", weight, self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_td_mean", loss_per_sample.mean().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_td_std", loss_per_sample.std().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_cross_mean", cross_entropy_loss_per_sample.mean().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_cross_std", cross_entropy_loss_per_sample.std().item(), self.learning_steps)
            # q_values = (pred_dist * self.q_network.support).sum(dim=1)  # [B]
            # self.writer.add_scalar(f"losses/{self.name}_q_values_mean", q_values.mean().item(), self.learning_steps)
            # self.writer.add_scalar(f"losses/{self.name}_q_values_std", q_values.std().item(), self.learning_steps)

        self.optimizer.step()

    def _dqn_loss(self, obs, actions, rewards, next_obs, dones):
        with torch.no_grad():
            next_q_target = self.target_network(next_obs)
            next_q_online = self.q_network(next_obs)

            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q = next_q_target.gather(1, best_actions)

            target = rewards + self.gamma * next_q * (1 - dones.float())

        q = self.q_network(obs)
        pred = q.gather(1, actions.unsqueeze(1))

        loss_per_sample = F.smooth_l1_loss(pred, target, reduction="none")
        return loss_per_sample.squeeze(1)
    
    def _cross_loss(self, obs, opp_actions, dones):
        logits = self.q_network.predict(obs)
        loss_per_sample = F.cross_entropy(logits, opp_actions.squeeze(), reduction="none")
        mask = (~dones).float().squeeze()  # Only consider non-terminal states for opponent prediction loss
        return loss_per_sample * mask

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def grad_weight_norm(self):
        weight, grad = 0.0, 0.0
        for p in self.q_network.parameters():
            if p.grad is not None:
                grad += p.grad.data.norm(2).item() ** 2
            weight += p.data.norm(2).item() ** 2
        return grad ** 0.5, weight ** 0.5
    