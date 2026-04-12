# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .buffers import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape

        # --- CNN Feature Extractor ---
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(dummy).shape[1]

        # --- Q Head ---
        self.q_head = nn.Sequential(
            nn.Linear(n_flatten, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        features = self.cnn(x)
        return self.q_head(features)
    
    def act(self, obs):  # Called in play for singular action selection
        with torch.no_grad():
            q_values = self.forward(obs)
            action = torch.argmax(q_values, dim=1)
        return action.item()

    def opponent_act(self, obs):  # Called in play for singular action selection
        with torch.no_grad():
            q_values = self.forward(obs)
        return torch.argmax(q_values, dim=1).cpu().numpy()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, obs_shape, n_actions, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        model = cls(obs_shape, n_actions)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        return model

class DQNAgent:
    def __init__(self, obs_shape, n_actions, lr, rb, batch_size, device):
        self.rb = rb
        self.batch_size = batch_size
        self.device = device
        
        self.q_network = QNetwork(obs_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.target_network = QNetwork(obs_shape, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_num_params(self):
        return sum(p.numel() for p in self.q_network.parameters())

    def select_action(self, obs):
        with torch.no_grad():
            q_values = self.q_network.forward(obs)
        return torch.argmax(q_values, dim=1).cpu().numpy()
    
    def add_to_buffer(self, obs, next_obs, action, reward, done, info):
        self.rb.add(obs, next_obs, action, reward, done, info)
    
    def learn(self):
        data = self.rb.sample(self.batch_size)
        with torch.no_grad():
            target_max, _ = self.target_network.forward(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + target_max * (1 - data.dones.flatten())
        old_val = self.q_network.forward(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path, verbose=True):
        torch.save(self.q_network.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    from rl_core.env import TronEnv, Tron2ChannelEnv
    env = Tron2ChannelEnv()
    # env = TronEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_space = env.observation_space  # (3, H, W)
    action_space = env.action_space           # should be 3
    rb = ReplayBuffer(10000, obs_space, action_space, device)
    agent = DQNAgent(obs_shape=obs_space.shape[-3:], n_actions=action_space.nvec[0], lr=2.5e-4, rb=rb, batch_size=32, device=device)


