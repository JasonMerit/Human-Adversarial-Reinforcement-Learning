import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from environment.env import TronEnv
from environment.wrappers import TronEgo
from agents.deterministic import DeterministicAgent



# ---------------------
# CNN Q-network
# ---------------------
class QNet(nn.Module):
    def __init__(self, input_shape=(3,10,10), num_actions=3):
        super().__init__()
        c,h,w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 32 * h * w
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
    @staticmethod
    def load(path):
        model = QNet()
        model.load_state_dict(torch.load(path))
        return model

# ---------------------
# Experience Replay
# ---------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return s,a,r,s2,d

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    # ---------------------
    # Hyperparameters
    # ---------------------
    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 32
    BUFFER_SIZE = 10000
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995
    TARGET_UPDATE = 10
    NUM_EPISODES = 500

    # ---------------------
    # Environment
    # ---------------------
    env = TronEnv(DeterministicAgent(start_left=True), width=10, height=10)
    env = TronEgo(env)
    num_actions = env.action_space.n
    state_shape = env.observation_space.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNet(input_shape=state_shape, num_actions=num_actions).to(device)
    target_net = QNet(input_shape=state_shape, num_actions=num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    eps = EPS_START

    # ---------------------
    # Training Loop
    # ---------------------
    for episode in range(NUM_EPISODES):
        state,_ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            if random.random() < eps:
                action = np.random.randint(num_actions)
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            buffer.push(state.cpu().numpy(), action, reward, next_state_tensor.cpu().numpy(), done)

            state = next_state_tensor
            total_reward += reward

            if len(buffer) >= BATCH_SIZE:
                s,a,r,s2,d = buffer.sample(BATCH_SIZE)
                s = torch.tensor(s, dtype=torch.float32, device=device).squeeze(1)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device).squeeze(1)
                a = torch.tensor(a, device=device, dtype=torch.long)
                r = torch.tensor(r, device=device, dtype=torch.float32)
                d = torch.tensor(d, device=device, dtype=torch.float32)

                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(s2).max(1)[0]
                    q_target = r + GAMMA * q_next * (1 - d)
                loss = nn.functional.mse_loss(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        eps = max(EPS_END, eps * EPS_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode} | Total Reward: {total_reward} | Epsilon: {eps:.3f}", end="\r")

    # Save the model
    torch.save(q_net.state_dict(), "q_net.pth")
    torch.save(target_net.state_dict(), "target_net.pth")
    print("\nTraining finished and models saved.")
