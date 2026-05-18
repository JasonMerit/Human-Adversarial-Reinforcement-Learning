# https://github.com/rlcode/per/tree/master
# import random
import torch, random
import numpy as np
from rl_core.env import GameState


def mirror(obs, actions, next_obs):
    return (
        np.flip(obs.copy(), axis=-1).copy(),
        2 - actions,
        np.flip(next_obs.copy(), axis=-1).copy(),
    )


class ReplayBuffer:
    def __init__(self, state_example, state_encode_fn: callable, player: int, args, device):
        assert player in [0, 1], "Player must be 0 or 1"
        self.device = device
        self.size = args.buffer_size
        self.num_envs = args.num_envs
        self.mirror_prob = args.mirror_prob
        self.encode = state_encode_fn
        self.player = player
        
        # replay storage
        self.state_storage = []
        self.next_state_storage = []
        for element in state_example:  # Dynamic tuple length
            if isinstance(element, np.ndarray):
                shape, dtype = element.shape[1:], element.dtype
                self.state_storage.append(np.empty((self.size, *shape), dtype=dtype))
                self.next_state_storage.append(np.empty((self.size, *shape), dtype=dtype))

            elif isinstance(element, (int, float)):
                self.state_storage.append(np.empty((self.size,), dtype=type(element)))
                self.next_state_storage.append(np.empty((self.size,), dtype=type(element)))
            else:
                raise TypeError(f"Expected state example to be a tuple of numpy array or scalar, got {type(element)}")

        self.action = np.empty((self.size,2), dtype=np.int8)  # Also storing opponent's action
        self.reward = np.empty((self.size,), dtype=np.float32)
        self.done = np.empty((self.size,), dtype=np.bool_)

        self.count = 0
        self.real_size = 0

    def add(self, state, actions, reward, next_state, done):
        batch_size = actions.shape[0]
        idxs = (np.arange(batch_size) + self.count) % self.size

        for i, (array, array_) in enumerate(zip(state, next_state)):
            self.state_storage[i][idxs] = array
            self.next_state_storage[i][idxs] = array_

        self.action[idxs] = actions
        self.reward[idxs] = reward
        self.done[idxs] = done

        self.count = (self.count + batch_size) % self.size
        self.real_size = min(self.size, self.real_size + batch_size)

    def sample(self, batch_size):
        assert self.real_size >= batch_size
        idxs = np.random.randint(0, self.real_size, size=batch_size)

        # Extract states
        states = [storage[idxs] for storage in self.state_storage]
        next_states = [storage[idxs] for storage in self.next_state_storage]

        # Extract obs, next_obs and actions for potential mirroring
        obs = self.encode(states)
        next_obs = self.encode(next_states)
        actions = self.action[idxs]

        # Mirror the obs, next_obs and actions
        if np.random.rand() < self.mirror_prob:  # Randomly decide to mirror or not
            obs, actions, next_obs = mirror(obs, actions, next_obs)

        obs = torch.from_numpy(obs).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device).long()
        rewards = torch.from_numpy(self.reward[idxs]).to(self.device).unsqueeze(1)
        dones = torch.from_numpy(self.done[idxs]).to(self.device).unsqueeze(1)
        next_obs = torch.from_numpy(next_obs).to(self.device).float()
        weights = torch.ones((batch_size, 1), device=self.device)
        indices = None
            
        return obs, actions, rewards, next_obs, dones, weights, indices
    
    def update(self, data_idxs, priorities):
        pass  # No priorities to update in a standard replay buffer

class PrioritizedReplayBuffer:

    def __init__(self, state_example, state_encode_fn: callable, player: int, args, device):
        assert player in [0, 1]

        self.device = device
        self.capacity = args.buffer_size
        self.mirror_prob = args.mirror_prob
        self.encode = state_encode_fn
        self.player = player

        self.eps = args.per_eps
        self.alpha = args.per_alpha
        self.beta = args.per_beta

        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

        self.state_storage = np.empty(self.capacity, dtype=object)  # Store entire GameState tuples
        # self.next_state_storage = np.empty(self.capacity, dtype=object)

        # for element in state_example:
        #     if isinstance(element, np.ndarray):
        #         shape, dtype = element.shape[1:], element.dtype
        #         self.state_storage.append(np.empty((self.capacity, *shape), dtype=dtype))
        #         self.next_state_storage.append(np.empty((self.capacity, *shape), dtype=dtype))

        #     elif isinstance(element, (int, float)):
        #         self.state_storage.append(np.empty((self.capacity,), dtype=type(element)))
        #         self.next_state_storage.append(np.empty((self.capacity,), dtype=type(element)))

        #     else:
        #         raise TypeError

        self.action = np.empty((self.capacity, 2), dtype=np.int8)
        self.reward = np.empty((self.capacity,), dtype=np.float32)
        self.done = np.empty((self.capacity,), dtype=np.bool_)

        self.count = 0
        self.real_size = 0

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, states, actions, reward, next_state, done):
        batch_size = actions.shape[0]
        idxs = (np.arange(batch_size) + self.count) % self.capacity

        self.state_storage[idxs] = [GameState(*elements) for elements in zip(*states)]  # Reconstruct GameState tuples
        # print(f"Stored state types: {[type(element) for element in self.state_storage[0]]}")
        # quit()
        self.action[idxs] = actions
        self.reward[idxs] = reward
        self.done[idxs] = done

        for idx in idxs:
            self.tree.add(self.max_priority, idx)

        self.count = (self.count + batch_size) % self.capacity
        self.real_size = min(self.capacity, self.real_size + batch_size)

    def sample(self, batch_size):
        assert self.real_size >= batch_size
        assert self.tree.total() > 0

        idxs = []
        tree_idxs = []
        priorities = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            tree_idx, p, data_idx = self.tree.get(s)

            idxs.append(data_idx)
            tree_idxs.append(tree_idx)
            priorities.append(p)

        idxs = np.array(idxs)
        tree_idxs = np.array(tree_idxs)
        priorities = np.array(priorities)

        states = self.state_storage[idxs]  # Extract GameState tuples based on sampled indices
        assert np.all([isinstance(s, GameState) for s in states]), "Expected all sampled states to be GameState instances"

        sampling_probabilities = priorities / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).to(self.device).unsqueeze(1).float()

        return states, weights, tree_idxs

    def update(self, tree_idxs, errors):
        priorities = self._get_priority(errors)
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, p in zip(tree_idxs, priorities):
            self.tree.update(idx, p)


class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)

        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data_idx):
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx