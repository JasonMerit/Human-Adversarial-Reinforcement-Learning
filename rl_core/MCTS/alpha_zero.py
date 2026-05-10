import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from rich import print


class AlphaZeroNetwork(nn.Module):

    def __init__(self, obs_shape, n_actions):
        super().__init__()

        channels, size, _ = obs_shape
        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, size, size)
            n_flatten = self.trunk(dummy).shape[1]

        hidden = 128

        self.policy_head = nn.Sequential(
            nn.Linear(n_flatten, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(n_flatten, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return value, policy_logits

    def act(self, obs: torch.Tensor):
        h = self.trunk(obs)
        logits = self.policy_head(h)
        probs = torch.softmax(logits, dim=1)
        return torch.multinomial(probs, num_samples=1)

    @classmethod
    def from_checkpoint(cls, path, obs_shape, n_actions, device="cpu"):
        net = cls(obs_shape, n_actions).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        return net


class AlphaZeroReplayBuffer:

    def __init__(self, capacity, obs_shape, n_actions, device):
        self.capacity = capacity
        self.device = device

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.policies = np.zeros((capacity, n_actions), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, policy, value):
        self.obs[self.ptr] = obs
        self.policies[self.ptr] = policy
        self.values[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.obs[idx], device=self.device)
        policies = torch.tensor(self.policies[idx], device=self.device)
        values = torch.tensor(self.values[idx], device=self.device)

        return states, policies, values


class AlphaZeroAgent:

    def __init__(
        self,
        obs_shape,
        n_actions,
        buffer_size,
        batch_size,
        learning_rate,
        weight_decay,
        device,
        writer=None
    ):

        self.device = device
        self.batch_size = batch_size
        self.writer = writer

        self.network = AlphaZeroNetwork(obs_shape, n_actions).to(device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        state_shape = (obs_shape[0], obs_shape[1], obs_shape[2])
        self.rb = AlphaZeroReplayBuffer(buffer_size, state_shape, n_actions, device)

        self.learning_steps = 0

    def get_num_params(self):
        return sum(p.numel() for p in self.network.parameters())

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    @torch.no_grad()
    def policy_value(self, obs: np.ndarray):
        assert isinstance(obs, np.ndarray), "Expected obs to be a numpy array"
        assert obs.ndim == 4, f"Expected obs to have shape (batch_size, channels, height, width), got {obs.shape}"
        value, logits = self.network(torch.tensor(obs, dtype=torch.float32, device=self.device))
        policy = torch.softmax(logits, dim=1)
        return policy, value
    
    @torch.no_grad()
    def act(self, obs: np.ndarray):
        actions = self.network.act(torch.tensor(obs, dtype=torch.float32, device=self.device)).squeeze(-1)
        return actions.cpu().numpy()

    def learn(self):

        if self.rb.size < self.batch_size:
            return

        self.learning_steps += 1

        obs, target_policies, target_values = self.rb.sample(self.batch_size)

        pred_values, policy_logits = self.network(obs)

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

        value_loss = F.mse_loss(pred_values, target_values)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.writer and self.learning_steps % 100 == 0:
            self.writer.add_scalar("loss/value_loss", value_loss.item(), self.learning_steps)
            self.writer.add_scalar("loss/policy_loss", policy_loss.item(), self.learning_steps)
            self.writer.add_scalar("loss/total_loss", loss.item(), self.learning_steps)

    def grad_weight_norm(self):

        weight, grad = 0.0, 0.0

        for p in self.network.parameters():

            if p.grad is not None:
                grad += p.grad.data.norm(2).item() ** 2

            weight += p.data.norm(2).item() ** 2

        return grad ** 0.5, weight ** 0.5

if __name__ == "__main__":
    from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv

    env = VecTronDuoEnv(5, 25)
    agent1 = AlphaZeroAgent(env.obs_shape, env.n_actions, 1000, 32, 1e-3, 1e-5, "cpu")
    agent2 = AlphaZeroAgent(env.obs_shape, env.n_actions, 1000, 32, 1e-3, 1e-5, "cpu")

    obs, _ = env.reset()
    obs1, obs2 = obs[:, 0], obs[:, 1]
    
    # Unsqueeze to add batch dimension
    # obs1 = np.expand_dims(obs1, axis=0)
    # obs2 = np.expand_dims(obs2, axis=0)

    # Test forward pass
    print("Testing forward pass...", end=" ")
    policy, value = agent1.policy_value(obs1)
    print("[green]Pass[/green]")
    
    # Fill the buffer before learning step
    print("Filling buffer...", end=" ")
    for _ in range(100):
        policy1, value1 = agent1.policy_value(obs1)
        policy2, value2 = agent2.policy_value(obs2)
        
        a1 = torch.multinomial(policy1, num_samples=1).squeeze(-1)
        a2 = torch.multinomial(policy2, num_samples=1).squeeze(-1)
        actions = np.stack([a1, a2], axis=1)

        next_obs, rewards, done, _, infos = env.step(actions)
        next_obs1, next_obs2 = next_obs[:, 0], next_obs[:, 1]

        agent1.rb.add(obs1, policy1.cpu().numpy(), value1.cpu().numpy())
        agent2.rb.add(obs2, policy2.cpu().numpy(), value2.cpu().numpy())

        obs1, obs2 = next_obs1, next_obs2
    
    print("[green]Pass[/green]")
    
    # Test learning step
    print("Testing learning step...", end=" ")



    
