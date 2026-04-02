import numpy as np
from tqdm import tqdm
import gymnasium as gym

class TrafficLightEnv(gym.Env):
    PATTERN = [0, 0, 0, 1, 2, 2, 1]  # GREEN, YELLOW, RED, YELLOW
    total_steps = 50

    def __init__(self):
        self.reset()

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(3)

    def reset(self):
        self.t = 0
        return self.PATTERN[self.t]

    def step(self, action):
        """
        action = predicted next light
        reward = 1 if correct
        """

        self.t += 1
        next_light = self.PATTERN[self.t % len(self.PATTERN)]

        done = self.t >= self.total_steps
        reward = int(action == next_light)

        return next_light, reward, done


import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(3, 8)

        self.rnn = nn.GRU(
            input_size=8,
            hidden_size=16,
            batch_first=True
        )

        self.fc = nn.Linear(16, 3)

    def forward(self, x, h):

        x = self.embed(x)

        out, h = self.rnn(x, h)

        logits = self.fc(out)

        return logits, h
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

def select_action(policy, obs, hidden):
    obs = torch.tensor([[obs]])
    logits, hidden = policy(obs, hidden)

    probs = torch.softmax(logits[:, -1], dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    logprob = dist.log_prob(action)

    return action.item(), logprob, hidden


if __name__ == "__main__":
    import torch.optim as optim

    env = TrafficLightEnv()
    policy = RNNPolicy()

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    gamma = 0.99

    log_freq = 10
    pbar = tqdm(range(10_000), desc="Training", miniters=log_freq)
    reward_history = []
    for episode in pbar:

        obs = env.reset()

        hidden = torch.zeros(1,1,16)

        log_probs = []
        rewards = []

        done = False

        while not done:

            action, logprob, hidden = select_action(policy, obs, hidden)

            obs, reward, done = env.step(action)

            log_probs.append(logprob)
            rewards.append(reward)

        # compute returns
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)

        loss = 0

        for logprob, G in zip(log_probs, returns):
            loss -= logprob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % log_freq == 0:
            acc = sum(rewards) / env.total_steps
            pbar.set_postfix({"Accuracy": acc})
            reward_history.append(acc)

    policy.save("rl_core/rnn_policy.pth")
    reward_history = np.array(reward_history)
    np.save("rl_core/rnn_reward_history.npy", reward_history)