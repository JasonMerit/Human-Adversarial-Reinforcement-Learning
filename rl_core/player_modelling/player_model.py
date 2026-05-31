import sys, time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rl_core.env import GameState, TronDuoEnv


class StateActionDataset(Dataset):
    def __init__(self, X, Y, encode_fn, player):
        self.X, self.Y = X, Y
        self.encode = encode_fn
        self.player = player

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        state_window, action_window = self.X[idx]
        target_action = self.Y[idx]

        obs = np.stack([TronDuoEnv.encode(s)[1 - self.player] for s in state_window], axis=0)

        return (
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(action_window, dtype=torch.long),
            torch.as_tensor(target_action, dtype=torch.long),
        )


class StateActionBuffer:
    def __init__(self, capacity, seq_len, num_envs):
        self.capacity, self.seq_len, self.num_envs = capacity, seq_len, num_envs
        self.current_states = [[] for _ in range(num_envs)]
        self.current_actions = [[] for _ in range(num_envs)]
        self.X, self.Y = deque(maxlen=capacity), deque(maxlen=capacity)

    def add(self, states, actions, dones):
        states = [GameState(*e) for e in zip(*states)]

        for i in range(self.num_envs):
            self.current_states[i].append(states[i])
            self.current_actions[i].append(actions[i])

            if dones[i]:
                s, a = self.current_states[i], self.current_actions[i]

                if len(a) >= self.seq_len:
                    for t in range(self.seq_len - 1, len(a)):
                        self.X.append((np.asarray(s[t-self.seq_len+1:t+1], dtype=object),
                                       np.asarray(a[t-self.seq_len+1:t], dtype=np.int64)))
                        self.Y.append(np.int64(a[t]))

                self.current_states[i].clear()
                self.current_actions[i].clear()
    
    def get_dataset(self, encode_fn, player):
        return StateActionDataset(self.X, self.Y, encode_fn, player)


class PlayerModelNetwork(nn.Module):
    def __init__(self, cnn, obs_shape, action_dim, emb_dim, hidden):
        super().__init__()
        self.cnn = cnn
        self.action_emb = nn.Embedding(action_dim, emb_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            cnn_dim = self.cnn(dummy).shape[-1]

        self.gru = nn.GRU(input_size=cnn_dim + emb_dim, hidden_size=hidden, batch_first=True)
        self.policy = nn.Linear(hidden, action_dim)

        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)

    def forward(self, states, prev_actions):
        B, L, C, H, W = states.shape

        x = states.reshape(B * L, C, H, W)

        with torch.no_grad():
            s_emb = self.cnn(x)

        s_emb = s_emb.reshape(B, L, -1)
        a_emb = self.action_emb(prev_actions)

        x = torch.cat([s_emb[:, :-1], a_emb], dim=-1)

        out, _ = self.gru(x)
        return self.policy(out[:, -1])


class PlayerModel:
    def __init__(self, cnn, obs_shape, n_actions, encode_fn, player, args, device):
        self.device, self.batch_size, self.player, self.encode = device, args.batch_size, player, encode_fn

        self.buffer = StateActionBuffer(args.buffer_size, args.seq_len, args.num_envs)

        self.net = PlayerModelNetwork(cnn, obs_shape, n_actions, args.emb_dim, args.hidden).to(device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

    def add(self, states, actions, dones):
        self.buffer.add(states, actions, dones)

    def train_predictor(self, epochs=1):
        dataset = self.buffer.get_dataset(self.encode, self.player)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.net.train()

        for _ in range(epochs):
            total_loss = total_correct = total = 0

            for states, prev_actions, target_actions in loader:
                states, prev_actions, target_actions = states.to(self.device), prev_actions.to(self.device), target_actions.to(self.device)

                logits = self.net(states, prev_actions)
                loss = F.cross_entropy(logits, target_actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * states.size(0)
                total_correct += (logits.argmax(dim=1) == target_actions).sum().item()
                total += states.size(0)

            print(f"loss={total_loss/total:.4f} acc={total_correct/total:.4f}")
        
        self.net.eval()