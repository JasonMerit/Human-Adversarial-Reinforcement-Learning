# https://github.com/Howuhh/prioritized_experience_replay/tree/main
import numpy as np
import torch
import random

class ReplayBuffer:
    def __init__(self, state_example, state_encode_fn: callable, args, device):
        self.device = device
        self.size = args.buffer_size
        self.num_envs = args.num_envs
        self.encode = state_encode_fn
        
        size = args.buffer_size
        self.state_storage = []
        self.next_state_storage = []
        for array in state_example:  # Dynamic tuple length
            assert isinstance(array, np.ndarray), f"Expected state example to be a tuple of numpy arrays, got {type(array)}"
            shape, dtype = array.shape[1:], array.dtype
            self.state_storage.append(np.empty((size, *shape), dtype=dtype))
            self.next_state_storage.append(np.empty((size, *shape), dtype=dtype))

        self.action = np.empty((args.buffer_size,), dtype=np.int8)
        self.reward = np.empty((args.buffer_size,), dtype=np.float32)
        self.done = np.empty((args.buffer_size,), dtype=np.bool_)

        self.count = 0
        self.real_size = 0

    def add(self, state, action, reward, next_state, done):
        batch_size = action.shape[0]

        idxs = (np.arange(batch_size) + self.count) % self.size

        for i, (array, array_) in enumerate(zip(state, next_state)):
            self.state_storage[i][idxs] = array
            self.next_state_storage[i][idxs] = array_

        self.action[idxs] = action
        self.reward[idxs] = reward
        self.done[idxs] = done

        self.count = (self.count + batch_size) % self.size
        self.real_size = min(self.size, self.real_size + batch_size)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        idxs = np.random.randint(0, self.real_size, size=batch_size)

        # Extract state
        state = [storage[idxs] for storage in self.state_storage]
        next_state = [storage[idxs] for storage in self.next_state_storage]

        obs = torch.from_numpy(self.encode(state)).to(self.device).float()
        actions = torch.from_numpy(self.action[idxs]).to(self.device).long().unsqueeze(1)
        rewards = torch.from_numpy(self.reward[idxs]).to(self.device).unsqueeze(1)
        dones = torch.from_numpy(self.done[idxs]).to(self.device).unsqueeze(1)
        next_obs = torch.from_numpy(self.encode(next_state)).to(self.device).float()
        weights = torch.ones((batch_size, 1), device=self.device)
        indices = None

        return obs, actions, rewards, next_obs, dones, weights, indices
    
    def update(self, data_idxs, priorities):
        pass  # No priorities to update in a standard replay buffer

class PrioritizedReplayBuffer:
    def __init__(self, state_example: tuple, state_encode_fn: callable, args, device):
        self.tree = SumTree(size=args.buffer_size)
        self.device = device
        self.encode = state_encode_fn
        self.size = args.buffer_size

        # PER params
        self.eps = args.per_eps
        self.alpha = args.per_alpha
        self.beta = args.per_beta
        self.max_priority = 1.0

        # replay storage
        self.state_storage = []
        self.next_state_storage = []
        for array in state_example:  # Dynamic tuple length
            assert isinstance(array, np.ndarray), f"Expected state example to be a tuple of numpy arrays, got {type(array)}"
            shape, dtype = array.shape[1:], array.dtype
            self.state_storage.append(np.empty((self.size, *shape), dtype=dtype))
            self.next_state_storage.append(np.empty((self.size, *shape), dtype=dtype))

        self.action = np.empty((self.size,), dtype=np.int8)
        self.reward = np.empty((self.size,), dtype=np.float32)
        self.done = np.empty((self.size,), dtype=np.bool_)

        self.count = 0
        self.real_size = 0

    def add(self, state, action, reward, next_state, done):
        batch_size = action.shape[0]  # should equal num_envs
        idxs = (np.arange(batch_size) + self.count) % self.size

        # store transitions
        for i, (array, array_) in enumerate(zip(state, next_state)):
            self.state_storage[i][idxs] = array
            self.next_state_storage[i][idxs] = array_

        self.action[idxs] = action
        self.reward[idxs] = reward
        self.done[idxs] = done

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

        # Extract state
        state = [storage[sample_idxs] for storage in self.state_storage]
        next_state = [storage[sample_idxs] for storage in self.next_state_storage]

        obs = torch.from_numpy(self.encode(state)).to(self.device).float()
        actions = torch.from_numpy(self.action[sample_idxs]).to(self.device).long().unsqueeze(1)
        rewards = torch.from_numpy(self.reward[sample_idxs]).to(self.device).unsqueeze(1)
        next_obs = torch.from_numpy(self.encode(next_state)).to(self.device).float()
        dones = torch.from_numpy(self.done[sample_idxs]).to(self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        return obs, actions, rewards, next_obs, dones, weights, tree_idxs

    def update(self, data_idxs, priorities):
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