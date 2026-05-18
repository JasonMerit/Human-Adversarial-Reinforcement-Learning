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
from .expectimax1 import MCTS
from rl_core.env import TronDuoEnv
from .vec_duo_tron import VecTronDuoEnv

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

        self.opp_head = nn.Sequential(
            nn.Linear(n_flatten, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        # make initial poponent policy uniform
        nn.init.constant_(self.opp_head[-1].weight, 0)
        nn.init.constant_(self.opp_head[-1].bias, 0)

    def forward(self, x) -> torch.Tensor:
        # assert x.ndim == 4, f"Expected input shape (B, C, H, W), got {x.shape}"
        # assert x.shape[1:] == (self.channels, self.size, self.size), f"Expected input shape (B, {self.channels}, {self.size}, {self.size}), got {x.shape}"
        h = self.cnn(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
    
    def predict(self, x) -> torch.Tensor:
        h = self.cnn(x).detach() # No gradients for opponent prediction
        logits = self.opp_head(h)
        return logits

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

    def __init__(self, player, obs_shape, n_actions, state_example: tuple, args, device, writer=None):
        self.device = device
        self.batch_size = args.batch_size

        self.network = KnegtNetwork(obs_shape, n_actions).to(device)
        self.target_network = KnegtNetwork(obs_shape, n_actions).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate, eps=1.5e-4)

        # MCTS
        env = TronDuoEnv(args.size)  # Dummy env for MCTS simulations
        self.envs = VecTronDuoEnv(args.rollouts, args.size)  # For encoding states in MCTS
        self.mcts = MCTS(player, self.act, self.adv_act, env, self.envs)

        Buffer = PrioritizedReplayBuffer if args.per else ReplayBuffer
        self.rb = Buffer(state_example, VecTronDuoEnv.encode, player, args, device)

        
        self.player = player  # Keep track of which agent 
        self.opp_weight = 0.1  # Weight for opponent prediction loss
        self.name = ["A", "B"][player]
        self.writer = writer
        self.learning_steps = 0

    def get_num_params(self):
        return sum(p.numel() for p in self.network.parameters())
    
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

        mcts_loss_per_sample = self._mcts_loss(states)
        loss = (mcts_loss_per_sample * weights.squeeze()).mean()

        # update priorities
        new_priorities = mcts_loss_per_sample.detach().cpu().numpy()  # Prioritize based on TD error only, not opponent prediction loss
        self.rb.update(tree_idxs, new_priorities)

        self.optimizer.zero_grad()
        loss.backward()

        # Nudged between backward and optimizer for grad logging
        if self.writer and self.learning_steps % 100 == 0:
            grad, weight = self.grad_weight_norm()
            self.writer.add_scalar(f"gradients/{self.name}_grad_norm", grad, self.learning_steps)
            self.writer.add_scalar(f"gradients/{self.name}_weight_norm", weight, self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_mcts_mean", mcts_loss_per_sample.mean().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_mcts_std", mcts_loss_per_sample.std().item(), self.learning_steps)
            # self.writer.add_scalar(f"losses/{self.name}_q_values_mean", q_values.mean().item(), self.learning_steps)
            # self.writer.add_scalar(f"losses/{self.name}_q_values_std", q_values.std().item(), self.learning_steps)

        self.optimizer.step()
    
    # def train_opponent_model(self):
    #     logits = self.network.predict(obs)
    #     loss = F.cross_entropy(logits, opp_actions.squeeze())
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def _mcts_loss(self, states):
        q_mcts = torch.stack([torch.tensor(self.mcts(s), device=states.device) for s in states])

        obs = np.stack([TronDuoEnv.encode(s) for s in states], axis=0)[:, self.player]
        q_pred = self.network(torch.as_tensor(obs).to(self.device))

        loss_per_action = F.smooth_l1_loss(q_pred, q_mcts, reduction="none")  # [B, 3]
        return loss_per_action.mean(dim=1)

    def _cross_loss(self, obs, opp_actions):
        logits = self.network.predict(obs)
        loss_per_sample = F.cross_entropy(logits, opp_actions.squeeze(), reduction="none")
        return loss_per_sample

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
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
        agent.update_target()
        return agent


if __name__ == "__main__":
    from ..argp import read_args
    from rl_core.env import TronDuoEnv
    from .vec_duo_tron import VecTronDuoEnv
    args = read_args()
    args.size = 15

    # env = TronDuoEnv(args.size)
    # envs = VecTronDuoEnv(64, args.size)

    env = TronDuoEnv(args.size)
    obs, info = env.reset()
    state = info['state']

    agent = KnegtAgent(0, env.obs_shape, env.n_actions, state, args, device="cpu")
    # agent = KnegtAgent.from_checkpoint("runs\KnegtReg_0\A.pth", 0, env.obs_shape, env.n_actions, state, args)
    obs = np.expand_dims(obs[1], axis=0)  # Add batch dimension and select opponent's perspective
    logits = agent.predict(obs)
    print(logits)
    
    
