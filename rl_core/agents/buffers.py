# https://github.com/Howuhh/prioritized_experience_replay/tree/main
import numpy as np
import torch
import random

def mirror_actions(joint_actions):
    # {0, 1, 2} -> {2, 1, 0} - left <-> right, up stays the same
    return 2 - joint_actions

MAPS = np.array([[0, 3, 2, 1], [2, 1, 0, 3]])  # Left <-> Right, Up <-> Down
def mirror_headings(h1, h2, player):
    H = np.stack([h1, h2], axis=0)
    other = 1 - player

    perp = ((H[0] - H[1]) % 2).astype(bool)

    # choose map per batch element
    map_idx = H[player] % 2                  # (B,)
    M = MAPS[map_idx]                        # (B,4)

    H_other = H[other]
    H[other] = np.where(perp, M[np.arange(len(H_other)), H_other], H_other)

    return H[0], H[1]

def mirror_states(states, player):
    # next states heading is given by h1, h2
    walls, p1, p2, h1, h2 = states
    walls, p1, p2 = walls.copy(), p1.copy(), p2.copy()

    size = walls.shape[1]
    headings = [h1, h2][player]

    for i, heading in enumerate(headings):
        axis = heading % 2

        walls[i] = np.flip(walls[i], axis=axis)

        coord = axis - 1
        p1[i, coord] = size - 1 - p1[i, coord]
        p2[i, coord] = size - 1 - p2[i, coord]

    h1, h2 = mirror_headings(h1, h2, player)

    return walls, p1, p2, h1, h2
    
def mirror_transition(states, joint_actions, next_states, player):
    joint_actions_m = mirror_actions(joint_actions)
    states_m = mirror_states(states, player)
    next_states_m = mirror_states(next_states, player)
    return states_m, joint_actions_m, next_states_m

class ReplayBuffer:
    def __init__(self, state_example, state_encode_fn: callable, player: int, args, device):
        assert player in [0, 1], "Player must be 0 or 1"
        self.device = device
        self.size = args.buffer_size
        self.num_envs = args.num_envs
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

        self.action = np.empty((self.size,), dtype=np.int8)
        self.reward = np.empty((self.size,), dtype=np.float32)
        self.done = np.empty((self.size,), dtype=np.bool_)

        self.count = 0
        self.real_size = 0

    def _store(self, state, action, reward, next_state, done):
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
    
    def add(self, state, joint_actions, reward, next_state, done):
        self._store(state, joint_actions[:, self.player], reward, next_state, done)

        # state_m, joint_actions_m, next_state_m = mirror_transition(state, joint_actions, next_state, self.player)
        # self._store(state_m, joint_actions_m[:, self.player], reward, next_state_m, done)

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
    envs.reset()

    np.random.seed(13)

    # Don't start from beginning
    envs.step(envs.sample_actions())
    obs, _, _, _, infos = envs.step(envs.sample_actions())
    state = infos["state"]

    # Test replay buffer
    buffer = ReplayBuffer(state, VecTronDuoEnv.encode, 0, args, 'cpu')

    for _ in range(1):
        joint_actions = envs.sample_actions()
        next_obs, rewards, dones, _, infos = envs.step(joint_actions)
        next_state = infos["state"]
        buffer.add(state, joint_actions, rewards, next_state, dones)
    
    # Original
    envs.set_states(state)
    envs.view(flush=False)
    print("PERFORMING", joint_actions, state[3], next_state[3])
    envs.set_states(next_state)
    envs.view(flush=False)
    print()
    # Mirrored
    state_m, joint_actions_m, next_state_m = mirror_transition(state, joint_actions, next_state, player=0)
    envs.set_states(state_m)
    envs.view(flush=False)
    print("PERFORMING", joint_actions_m, state_m[3], next_state_m[3])
    envs.set_states(next_state_m)
    envs.view(flush=False)
    print()
    print()
    # Sample all
    obs, actions, rewards, next_obs, dones, weights, indices = buffer._sample_all()
    print(obs[:, 0] + 2*obs[:, 1] + 3*obs[:, 2])  # Should be the same for both agents since they are mirrors of each other
    print(next_obs[:, 0] + 2*next_obs[:, 1] + 3*next_obs[:, 2])
    
    
