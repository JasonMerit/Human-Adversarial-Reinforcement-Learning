import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .buffers import PrioritizedReplayBuffer, ReplayBuffer
from ..utils import TimerRegistry
from rich import print

class DuelingNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions, args):
        super().__init__()
        channels, size, _ = obs_shape
        self.n_actions = n_actions

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, args.conv1, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(args.conv1, args.conv2, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, size, size)
            n_flatten = self.cnn(dummy).shape[1]

        hidden = args.hidden_size
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

        # Print parameter count
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"DuelingNetwork initialized with {total_params:,} parameters.")
        # quit()

    def forward(self,x) -> torch.Tensor:
        h = self.cnn(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    @torch.inference_mode()
    def act(self, obs):
        h = self.cnn(obs)
        a = self.adv_head(h)
        return torch.argmax(a, dim=1).cpu().numpy()

    
    @classmethod
    def from_checkpoint(cls, path, obs_shape, n_actions, args, device="cpu"):
        net = cls(obs_shape=obs_shape, n_actions=n_actions, args=args).to(device)
        net_dict = torch.load(path, weights_only=True, map_location=device)
        net.load_state_dict(net_dict)
        return net

class RainbowAgent:

    def __init__(self, player, obs_shape, n_actions, state_encode_fn, args, device, writer=None):

        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        
        self.network = DuelingNetwork(obs_shape, n_actions, args).to(device)
        self.target_network = DuelingNetwork(obs_shape, n_actions, args).to(device)
        
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate, eps=1.5e-4)

        Buffer = PrioritizedReplayBuffer if args.per else ReplayBuffer
        self.rb = Buffer(state_encode_fn, player, args, device)
        
        self.player = player  # Keep track of which agent 
        self.name = "A" if player == 0 else "B"
        self.writer = writer
        self.learning_steps = 0

    def get_num_params(self):
        return sum(p.numel() for p in self.network.parameters())
    
    def save(self, path, verbose=True):
        torch.save(self.network.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
    
    @torch.inference_mode()
    def act(self, obs: np.ndarray):  
        assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
        q = self.network(torch.as_tensor(obs).to(self.device))
        return torch.argmax(q, dim=1).cpu().numpy()
    
    def add(self, *args):
        self.rb.add(*args)

    @TimerRegistry.wrap_fn("rainbow_learn")
    def learn(self):
        self.learning_steps += 1

        obs, actions, rewards, next_obs, dones, weights, indices = self.rb.sample(self.batch_size)
        # obs = obs[:, self.player]
        # next_obs = next_obs[:, self.player]

        loss_per_sample = self._dqn_loss(obs[:, self.player], actions[:, self.player], rewards, next_obs[:, self.player], dones)
        loss = (loss_per_sample * weights.squeeze()).mean()

        # update priorities
        new_priorities = loss_per_sample.detach().cpu().numpy()
        self.rb.update(indices, new_priorities)

        self.optimizer.zero_grad()
        loss.backward()

        # Nudged between backward and optimizer for grad logging
        if self.writer and self.learning_steps % 100 == 0:
            grad, weight = self.grad_weight_norm()
            self.writer.add_scalar(f"gradients/{self.name}_grad_norm", grad, self.learning_steps)
            self.writer.add_scalar(f"gradients/{self.name}_weight_norm", weight, self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_td_loss_mean", loss_per_sample.mean().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_td_loss_std", loss_per_sample.std().item(), self.learning_steps)
            # q_values = (pred_dist * self.network.support).sum(dim=1)  # [B]
            # self.writer.add_scalar(f"losses/{self.name}_q_values_mean", q_values.mean().item(), self.learning_steps)
            # self.writer.add_scalar(f"losses/{self.name}_q_values_std", q_values.std().item(), self.learning_steps)

        self.optimizer.step()

    def _dqn_loss(self, obs, actions, rewards, next_obs, dones):
        with torch.no_grad():
            next_q_target = self.target_network(next_obs)
            next_q_online = self.network(next_obs)

            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q = next_q_target.gather(1, best_actions)

            target = rewards + self.gamma * next_q * (1 - dones.float())

        q = self.network(obs)
        pred = q.gather(1, actions.unsqueeze(1))

        loss_per_sample = F.smooth_l1_loss(pred, target, reduction="none")
        return loss_per_sample.squeeze(1)

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def grad_weight_norm(self):
        weight, grad = 0.0, 0.0
        for p in self.network.parameters():
            if p.grad is not None:
                grad += p.grad.data.norm(2).item() ** 2
            weight += p.data.norm(2).item() ** 2
        return grad ** 0.5, weight ** 0.5
    