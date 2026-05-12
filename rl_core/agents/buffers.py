# https://github.com/Howuhh/prioritized_experience_replay/tree/main
import numpy as np
import torch
import random
from rich import print

def mirror(obs, actions, next_obs):
    """Expects (B, P, C, H, W) for obs and next_obs, (B, 2) for actions. Player axis should be second and channel axis should be third."""
    # assert obs.ndim == 5, "Forgot the batch or included obs for both agents? Expected shape (B, 3, size, size)"
    # assert next_obs.ndim == 5, "Forgot the batch or included obs for both agents? Expected shape (B, 3, size, size)"
    # assert obs.shape[1:3] == (2, 3), f"Expected player second axis and channel third axis, got shape {obs.shape}"
    # assert next_obs.shape[1:3] == (2, 3), f"Expected player second axis and channel third axis, got shape {next_obs.shape}"
    # assert actions.shape == (obs.shape[0], 2), f"Expected shape (B, 2) for actions, got {actions.shape}"
    return np.flip(obs.copy(), axis=-1).copy(), 2-actions, np.flip(next_obs.copy(), axis=-1).copy()  # copy to remove negative stride

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

    def _sample_all(self):
        idxs = np.arange(0, self.real_size)

        # Extract state
        state = [storage[idxs] for storage in self.state_storage]
        next_state = [storage[idxs] for storage in self.next_state_storage]

        obs = torch.from_numpy(self.encode(state)).to(self.device).float()
        actions = torch.from_numpy(self.action[idxs]).to(self.device).long().unsqueeze(1)
        rewards = torch.from_numpy(self.reward[idxs]).to(self.device).unsqueeze(1)
        dones = torch.from_numpy(self.done[idxs]).to(self.device).unsqueeze(1)
        next_obs = torch.from_numpy(self.encode(next_state)).to(self.device).float()
        weights = torch.ones((self.real_size, 1), device=self.device)
        indices = None

        return obs[:, self.player], actions, rewards, next_obs[:, self.player], dones, weights, indices
    
    def update(self, data_idxs, priorities):
        pass  # No priorities to update in a standard replay buffer

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_example: tuple, state_encode_fn: callable, args, device):
        super().__init__(state_example, state_encode_fn, args, device)
        self.tree = SumTree(size=args.buffer_size)
        # self.device = device
        # self.encode = state_encode_fn
        # self.size = args.buffer_size

        # PER params
        self.eps = args.per_eps
        self.alpha = args.per_alpha
        self.beta = args.per_beta
        self.max_priority = 1.0

        # replay storage
        # self.state_storage = []
        # self.next_state_storage = []
        # for array in state_example:  # Dynamic tuple length
        #     if isinstance(array, np.ndarray):
        #         shape, dtype = array.shape[1:], array.dtype
        #         self.state_storage.append(np.empty((self.size, *shape), dtype=dtype))
        #         self.next_state_storage.append(np.empty((self.size, *shape), dtype=dtype))

        #     elif isinstance(array, (int, float)):
        #         self.state_storage.append(np.empty((self.size,), dtype=type(array)))
        #         self.next_state_storage.append(np.empty((self.size,), dtype=type(array)))

        #     else:
        #         raise TypeError(f"Expected state example to be a tuple of numpy arrays, got {type(array)}")

        # self.action = np.empty((self.size,), dtype=np.int8)
        # self.reward = np.empty((self.size,), dtype=np.float32)
        # self.done = np.empty((self.size,), dtype=np.bool_)

        # self.count = 0
        # self.real_size = 0

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        batch_size = action.shape[0]
        idxs = (np.arange(batch_size) + self.count) % self.size

        # add priorities
        for idx in idxs:
            self.tree.add(self.max_priority, idx)

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

        return obs[:, self.player], actions, rewards, next_obs[:, self.player], dones, weights, tree_idxs

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


if __name__ == "__main__":
    from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv
    args = type('args', (), {'buffer_size': 100, 'num_envs': 1})
    RENDER=False
    envs = VecTronDuoEnv(args.num_envs, 5, render=RENDER)
    (obs1, obs2), infos = envs.reset()

    # np.random.seed(13)

    # # Don't start from beginning
    # envs.step(envs.sample_actions())
    # obs, _, _, _, infos = envs.step(envs.sample_actions())
    # state = infos["state"]

    # # Test replay buffer
    # buffer = ReplayBuffer(state, VecTronDuoEnv.encode, 0, args, 'cpu')

    for _ in range(2):
        joint_actions = envs.sample_actions()
        (obs1, obs2), rewards, dones, _, infos = envs.step(joint_actions)
        # next_obs, rewards, dones, _, infos = envs.step(joint_actions)
        next_state = infos["state"]
        # buffer.add(state, joint_actions[:, 0], rewards, next_state, dones)
    
    # obs, actions, rewards, next_obs, dones, weights, indices = buffer._sample_all()
    # print(obs[:, 0] + 2*obs[:, 1] + 3*obs[:, 2])  # Should be the same for both agents since they are mirrors of each other
    # print(next_obs[:, 0] + 2*next_obs[:, 1] + 3*next_obs[:, 2])
    # print(actions)
    
    joint_actions = envs.sample_actions()
    (next_obs1, next_obs2), rewards, dones, _, infos = envs.step(joint_actions)
    _obs, _actions, _next_obs = mirror(obs1, joint_actions[:, 0], next_obs1)

    print(obs1.shape, next_obs1.shape, joint_actions[:, 0].shape)
    print(_obs.shape, _next_obs.shape, _actions.shape)
    print()
    print(obs1[:, 0] + 2*obs1[:, 1] + 3*obs1[:, 2])
    print(_obs[:, 0] + 2*_obs[:, 1] + 3*_obs[:, 2])
    print()
    print(next_obs1[:, 0] + 2*next_obs1[:, 1] + 3*next_obs1[:, 2])
    print(_next_obs[:, 0] + 2*_next_obs[:, 1] + 3*_next_obs[:, 2])
    print()
    print(joint_actions[:, 0])
    print(_actions)
    
    
