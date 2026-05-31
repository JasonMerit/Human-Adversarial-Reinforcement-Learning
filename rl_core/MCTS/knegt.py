import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from rich import print

from .buffers import PrioritizedReplayBuffer, ReplayBuffer
from ..utils import TimerRegistry
# from .agent_mcts import MCTS
# from .expectimax1 import MCTS
from rl_core.env import TronDuoEnv
from .vec_duo_tron import VecTronDuoEnv
from rl_core.player_modelling.player_model import PlayerModel, StateActionBuffer

class KnegtNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.channels, self.size, _ = obs_shape
        self.n_actions = n_actions

        self.cnn = nn.Sequential(
            nn.Conv2d(self.channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.channels, self.size, self.size)
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

    def forward(self, x) -> torch.Tensor:
        # assert x.ndim == 4, f"Expected input shape (B, C, H, W), got {x.shape}"
        # assert x.shape[1:] == (self.channels, self.size, self.size), f"Expected input shape (B, {self.channels}, {self.size}, {self.size}), got {x.shape}"
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
    def from_checkpoint(cls, path, obs_shape, n_actions, device="cpu"):
        net = cls(obs_shape=obs_shape, n_actions=n_actions).to(device)
        net.load_state_dict(torch.load(path, weights_only=True, map_location=device))        
        return net
    

class KnegtAgent:
    """Assumes VecTronDuoEnv"""

    def __init__(self, player, obs_shape, n_actions, args, device, writer=None):
        self.device = device
        self.batch_size = args.batch_size
        self.rollouts = args.rollouts

        self.network = KnegtNetwork(obs_shape, n_actions).to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate, eps=1.5e-4)

        self.envs = VecTronDuoEnv(args.rollouts, args.size)  # Dummy envs for MC rollouts

        Buffer = PrioritizedReplayBuffer if args.per else ReplayBuffer
        self.rb = Buffer(VecTronDuoEnv.encode, player, args, device)

        self.epi_rb = StateActionBuffer(args.buffer_size, args.seq_len, args.num_envs, device)  # Separate buffer for opponent modeling
        self.player_modeler = PlayerModel(self.network.cnn, obs_shape, n_actions, args, device).to(device)

        self.player = player  # Keep track of which agent
        self.name = ["A", "B"][player]
        self.writer = writer
        self.learning_steps = 0

    def get_num_params(self):
        return sum(p.numel() for p in self.network.parameters())

    def add(self, state, actions, reward, next_state, done):
        self.rb.add(state, actions, reward, next_state, done)
        self.epi_rb.add(state, actions[:, 1 - self.player], done)  # Store opponent's actions for modeling

    def save(self, path, verbose=True):
        torch.save(self.network.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
    
    def act(self, obs: np.ndarray):  
        assert obs.ndim == 4, f"Expected input shape (B, C, H, W), got {obs.shape}"
        assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
        return self.network.act(torch.as_tensor(obs).to(self.device))

    def adv_act(self, obs: np.ndarray):
        assert obs.ndim == 4, f"Expected input shape (B, C, H, W), got {obs.shape}"
        assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
        return np.random.choice(3, size=obs.shape[0])

    # def get_adv_probs(self, obs: np.ndarray):
    #     assert obs.ndim == 4, f"Expected input shape (B, C, H, W), got {obs.shape}"
    #     assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
    #     # return uniform distributed for now
    #     return np.ones((obs.shape[0], 3)) / 3
        # logits = self.network.predict(torch.as_tensor(obs).to(self.device))
        # probs = F.softmax(logits, dim=1)
        # return probs.cpu().numpy()
    
    @torch.inference_mode()
    def predict(self, obs: np.ndarray):
        assert obs.ndim == 4, f"Expected input shape (B, C, H, W), got {obs.shape}"
        assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
        logits = self.network.predict(torch.as_tensor(obs).to(self.device))
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
    @TimerRegistry.wrap_fn("knegt_learn")
    def learn(self):
        self.learning_steps += 1

        states, weights, tree_idxs = self.rb.sample(self.batch_size)

        mc_loss_per_sample = self._mc_loss(states)
        loss = (mc_loss_per_sample * weights.squeeze()).mean()

        # update priorities
        new_priorities = mc_loss_per_sample.detach().cpu().numpy()
        self.rb.update(tree_idxs, new_priorities)

        self.optimizer.zero_grad()
        loss.backward()

        # Nudged between backward and optimizer for grad logging
        if self.writer and self.learning_steps % 100 == 0:
            grad, weight = self.grad_weight_norm()
            self.writer.add_scalar(f"gradients/{self.name}_grad_norm", grad, self.learning_steps)
            self.writer.add_scalar(f"gradients/{self.name}_weight_norm", weight, self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_mc_mean", mc_loss_per_sample.mean().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_mc_std", mc_loss_per_sample.std().item(), self.learning_steps)
            # self.writer.add_scalar(f"losses/{self.name}_q_values_mean", q_values.mean().item(), self.learning_steps)
            # self.writer.add_scalar(f"losses/{self.name}_q_values_std", q_values.std().item(), self.learning_steps)

        self.optimizer.step()
    
    # def train_opponent_model(self):
    #     logits = self.network.predict(obs)
    #     loss = F.cross_entropy(logits, opp_actions.squeeze())
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    def _mc_loss(self, states):
        q_mc = torch.as_tensor(np.stack([self._rollout(s) for s in states]), device=self.device, dtype=torch.float32)

        obs = np.stack([TronDuoEnv.encode(s) for s in states], axis=0)[:, self.player]
        q_pred = self.network(torch.as_tensor(obs).to(self.device))

        loss_per_action = F.smooth_l1_loss(q_pred, q_mc, reduction="none")  # [B, 3]
        return loss_per_action.mean(dim=1)

    def _cross_loss(self, obs, opp_actions):
        logits = self.network.predict(obs)
        loss_per_sample = F.cross_entropy(logits, opp_actions.squeeze(), reduction="none")
        return loss_per_sample

    def _rollout(self, state):
        q = np.zeros(3)
        actions = np.empty((self.rollouts, 2), dtype=np.int8)

        for i in range(3):
            self.envs.set_state(state)
            obs = self.envs.get_obs()

            b = self.adv_act(obs[:, 1 - self.player])
            actions[:, self.player], actions[:, 1 - self.player] = i, b
            obs, r, d, _, _ = self.envs.step(actions)

            total = r.copy()
            active = ~d
            while active.any():
                a = self.act(obs[:, self.player])
                b = self.adv_act(obs[:, 1 - self.player])
                actions[:, self.player], actions[:, 1 - self.player] = a, b

                obs, r, d, _, _ = self.envs.step(actions)

                total[d] = r[d]
                active &= ~d
                # active = np.logical_and(active, ~d)

            q[i] = total.mean()
        
        return q
    
    def grad_weight_norm(self):
        weight, grad = 0.0, 0.0
        for p in self.network.parameters():
            if p.grad is not None:
                grad += p.grad.data.norm(2).item() ** 2
            weight += p.data.norm(2).item() ** 2
        return grad ** 0.5, weight ** 0.5
    
    @classmethod
    def from_checkpoint(cls, path, player, obs_shape, n_actions, state_example, args):
        agent = cls(player, obs_shape, n_actions, state_example, args, "cpu")
        agent.network.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        return agent



