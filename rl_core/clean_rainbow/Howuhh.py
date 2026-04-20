# https://github.com/Howuhh/prioritized_experience_replay/tree/main
import numpy as np
import torch
import random

class ReplayBuffer:
    def __init__(self, obs_shape, n_actions, buffer_size, device, n_envs):
        self.device = device
        self.size = buffer_size
        self.n_envs = n_envs

        self.state = np.empty((buffer_size, *obs_shape), dtype=np.uint8)
        self.next_state = np.empty((buffer_size, *obs_shape), dtype=np.uint8)

        self.action = np.empty((buffer_size,), dtype=np.int64)
        self.reward = np.empty((buffer_size,), dtype=np.float32)
        self.done = np.empty((buffer_size,), dtype=np.bool_)

        self.count = 0
        self.real_size = 0

    def add(self, state, action, reward, next_state, done):
        batch_size = state.shape[0]

        idxs = (np.arange(batch_size) + self.count) % self.size

        self.state[idxs] = state
        self.action[idxs] = action
        self.reward[idxs] = reward
        self.next_state[idxs] = next_state
        self.done[idxs] = done

        self.count = (self.count + batch_size) % self.size
        self.real_size = min(self.size, self.real_size + batch_size)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        idxs = np.random.randint(0, self.real_size, size=batch_size)

        states = torch.from_numpy(self.state[idxs]).to(self.device).float()
        actions = torch.from_numpy(self.action[idxs]).to(self.device)
        rewards = torch.from_numpy(self.reward[idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_state[idxs]).to(self.device).float()
        dones = torch.from_numpy(self.done[idxs]).to(self.device)

        return states, actions, rewards, next_states, dones

class PrioritizedReplayBuffer:
    def __init__(self, obs_shape, n_actions, buffer_size, device, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)
        self.device = device

        # PER params
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps

        # replay storage (NUMPY)
        self.state = np.empty((buffer_size, *obs_shape), dtype=np.float32)
        self.action = np.empty((buffer_size, n_actions), dtype=np.float32)
        self.reward = np.empty(buffer_size, dtype=np.float32)
        self.next_state = np.empty((buffer_size, *obs_shape), dtype=np.float32)
        self.done = np.empty(buffer_size, dtype=np.int32)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        states, actions, rewards, next_states, dones = transition
        batch_size = states.shape[0]  # should equal num_envs

        idxs = (np.arange(batch_size) + self.count) % self.size

        # store transitions
        self.state[idxs] = states
        self.action[idxs] = actions
        self.reward[idxs] = rewards
        self.next_state[idxs] = next_states
        self.done[idxs] = dones

        # add priorities
        for idx in idxs:
            self.tree.add(self.max_priority, idx)

        # update counters
        self.count = (self.count + batch_size) % self.size
        self.real_size = min(self.size, self.real_size + batch_size)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = []
        tree_idxs = []
        priorities = np.empty(batch_size, dtype=np.float32)

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        sample_idxs = np.array(sample_idxs)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** (-self.beta)
        weights /= weights.max()

        # convert to torch here
        batch = (
            torch.tensor(self.state[sample_idxs], device=self.device),
            torch.tensor(self.action[sample_idxs], device=self.device),
            torch.tensor(self.reward[sample_idxs], device=self.device),
            torch.tensor(self.next_state[sample_idxs], device=self.device),
            torch.tensor(self.done[sample_idxs], device=self.device),
        )

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"