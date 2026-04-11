import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .buffer import PrioritizedReplayBuffer
from .utils import TimerRegistry
from rich import print

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        # factorized gaussian noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)
    

class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, n_actions, n_atoms, v_min, v_max):
        super().__init__()

        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        self.n_actions = n_actions

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 25, 25)
            conv_output_size = self.network(dummy).shape[1]
        
        size = 512
        self.value_head = nn.Sequential(
            NoisyLinear(conv_output_size, size),
            nn.ReLU(),
            NoisyLinear(size, n_atoms)
        )

        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, size),
            nn.ReLU(),
            NoisyLinear(size, n_atoms * self.n_actions)
        )

    @TimerRegistry.wrap_fn("network_forward")
    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W)
        assert isinstance(x, torch.Tensor), f"Expected input to be a torch.Tensor, got {type(x)}"
        assert x.dim() == 4 and x.shape[1] == 3, f"Expected input shape (B, 3, H, W), got {x.shape}"

        h = self.network(x)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        return F.softmax(q_atoms, dim=2)

    @TimerRegistry.wrap_fn("network_reset_noise")
    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
    
    @classmethod
    def from_checkpoint(cls, path, n_actions, device):
        net = cls(n_actions=n_actions, n_atoms=51, v_min=-10, v_max=10).to(device)
        net.load_state_dict(torch.load(path, weights_only=True, map_location=device))        
        return net

class Rainbow:

    def __init__(self, n_actions, args, device, writer, name):
        self.device = device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.n_step = args.n_step

        self.q_network = NoisyDuelingDistributionalNetwork(n_actions, args.n_atoms, args.v_min, args.v_max).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)
        self.target_network = NoisyDuelingDistributionalNetwork(n_actions, args.n_atoms, args.v_min, args.v_max).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = PrioritizedReplayBuffer(args, device)
        
        self.name = name
        self.writer = writer
        self.learning_steps = 0

    
    def save(self, path, verbose=True):
        torch.save(self.q_network.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
    
    def act(self, obs: np.ndarray):
        assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
        with torch.no_grad():
            q_dist = self.q_network(torch.as_tensor(obs).to(self.device))
            q_values = torch.sum(q_dist * self.q_network.support, dim=2)
            return torch.argmax(q_values, dim=1).cpu().numpy()

    @TimerRegistry.wrap_fn("agent_learn")
    def learn(self):
        self.learning_steps += 1
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        obs, actions, rewards, next_obs, dones, weights, indices = self.rb.sample()

        with torch.no_grad():
            next_dist = self.target_network(next_obs)  # [B, num_actions, n_atoms]
            support = self.target_network.support  # [n_atoms]

            # double q-learning
            next_dist_online = self.q_network(next_obs)  # [B, num_actions, n_atoms]
            next_q_online = torch.sum(next_dist_online * support, dim=2)  # [B, num_actions]
            best_actions = torch.argmax(next_q_online, dim=1)  # [B]
            next_pmfs = next_dist[torch.arange(self.batch_size), best_actions]  # [B, n_atoms]

            # compute the n-step Bellman update.
            gamma_n = self.gamma**self.n_step
            next_atoms = rewards + gamma_n * support * (1 - dones.float())
            assert next_atoms < self.q_network.v_max, f"Expected next_atoms < {self.q_network.v_max}, got {next_atoms.max().item()}"
            assert next_atoms > self.q_network.v_min, f"Expected next_atoms > {self.q_network.v_min}, got {next_atoms.min().item()}"
            tz = next_atoms.clamp(self.q_network.v_min, self.q_network.v_max)

            TimerRegistry.start()
            # projection
            b = (tz - self.q_network.v_min) / self.q_network.delta_z  # shape: [B, n_atoms]
            l = b.floor().clamp(0, self.q_network.n_atoms - 1)
            u = b.ceil().clamp(0, self.q_network.n_atoms - 1)

            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u.float() + (l == b).float() - b) * next_pmfs  # [B, n_atoms]
            d_m_u = (b - l) * next_pmfs  # [B, n_atoms]

            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
            TimerRegistry.stop("projection")

        TimerRegistry.start()
        dist = self.q_network(obs)  # [B, num_actions, n_atoms]
        pred_dist = dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.q_network.n_atoms)).squeeze(1)
        log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))

        loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
        loss = (loss_per_sample * weights.squeeze()).mean()
        TimerRegistry.stop("loss")

        # update priorities
        new_priorities = loss_per_sample.detach().cpu().numpy()
        self.rb.update(indices, new_priorities)

        TimerRegistry.start()
        self.optimizer.zero_grad()
        loss.backward()

        # Nudged between backward and optimizer for grad logging
        if self.writer and self.learning_steps % 100 == 0:
            grad, weight = self.grad_weight_norm()
            self.writer.add_scalar(f"gradients/{self.name}_grad_norm", grad, self.learning_steps)
            self.writer.add_scalar(f"gradients/{self.name}_weight_norm", weight, self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_td_loss_mean", loss_per_sample.mean().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_td_loss_std", loss_per_sample.std().item(), self.learning_steps)
            q_values = (pred_dist * self.q_network.support).sum(dim=1)  # [B]
            self.writer.add_scalar(f"losses/{self.name}_q_values_mean", q_values.mean().item(), self.learning_steps)
            self.writer.add_scalar(f"losses/{self.name}_q_values_std", q_values.std().item(), self.learning_steps)

        self.optimizer.step()
        TimerRegistry.stop("backward")


    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def grad_weight_norm(self):
        weight, grad = 0.0, 0.0
        for p in self.q_network.parameters():
            if p.grad is not None:
                grad += p.grad.data.norm(2).item() ** 2
            weight += p.data.norm(2).item() ** 2
        return grad ** 0.5, weight ** 0.5