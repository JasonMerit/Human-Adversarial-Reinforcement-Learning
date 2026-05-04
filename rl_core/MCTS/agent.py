import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rich import print
from .mcts import MCTS, Node


class AlphaZeroNet(nn.Module):

    def __init__(self, obs_shape, n_actions):
        super().__init__()

        c, h, w = obs_shape

        self.trunk = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        flat = 64 * h * w

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.trunk(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value.squeeze(-1)


class AlphaZeroAgent:

    def __init__(self, env, mcts, net, device="cpu"):
        self.env = env
        self.mcts = mcts
        self.net = net.to(device)

        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def evaluate(self, state):
        """Network evaluation used by MCTS."""

        obs = self.env.encode(state)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.net(obs)

        policy = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        value = value.item()

        return policy, value
    
    def act(self, state, sims=200, temperature=1.0):
        root = Node(state, self.env.n_actions)

        action, visit_counts = self.mcts.plan(root, sims)

        pi = visit_counts ** (1 / temperature)
        pi = pi / pi.sum()

        action = np.random.choice(len(pi), p=pi)

        return action, pi
    
    def self_play(self, sims=200):
        state = self.env.reset()[1]["state"]

        history = []
        while True:
            action, pi = self.act(state, sims)
            obs = self.env.encode(state)
            history.append((obs, pi))

            _, reward, done, _, info = self.env.step(action)
            state = info["state"]

            if done:
                break

        z = reward
        data = []
        for obs, pi in history:
            data.append((obs, pi, z))

        return data
    
    def learn(self, dataset, batch_size=64, epochs=5):
        self.net.train()

        for _ in range(epochs):

            np.random.shuffle(dataset)

            for i in range(0, len(dataset), batch_size):

                batch = dataset[i:i+batch_size]

                obs = torch.tensor(
                    np.stack([x[0] for x in batch]),
                    dtype=torch.float32
                ).to(self.device)

                pi = torch.tensor(
                    np.stack([x[1] for x in batch]),
                    dtype=torch.float32
                ).to(self.device)

                z = torch.tensor(
                    [x[2] for x in batch],
                    dtype=torch.float32
                ).to(self.device)

                logits, values = self.net(obs)

                policy_loss = -(pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = torch.mean((z - values) ** 2)

                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

if __name__ == "__main__":
    from rl_core.env import PoLEnv
    dataset = []
    SIZE = 10
    agent = AlphaZeroAgent(env=PoLEnv(SIZE), mcts=MCTS(None, None), net=AlphaZeroNet((3, SIZE, SIZE), 5))

    for iteration in range(1000):

        print("Self-play...")

        for _ in range(25):
            game_data = agent.self_play(sims=200)
            dataset.extend(game_data)

        print("Training...")
        agent.learn(dataset)

        dataset = dataset[-10000:]  # replay buffer

wood chopper 30 / min
stone miner 20 / min

    planks 2 : 1
    stakes 45 = 3 : 2

    bricks 30 = 3 : 2
    slabs 40 = 2 : 1

    Light 120/120 = 4/6 = 2/3


WORLD 2 COIN 42  time 52 