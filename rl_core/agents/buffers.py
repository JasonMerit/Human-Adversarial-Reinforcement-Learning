# https://github.com/Howuhh/prioritized_experience_replay/tree/main
import numpy as np
import torch
import random
from rich import print

def mirror(obs, actions, next_obs):
    """Expects (B, P, C, H, W) for obs and next_obs, (B, 2) for actions. Player axis should be second and channel axis should be third."""
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
    
    def update(self, data_idxs, priorities):
        pass  # No priorities to update in a standard replay buffer

# class PrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, state_example, state_encode_fn: callable, player: int, args, device):
#         super().__init__(state_example, state_encode_fn, player, args, device)
#         self.tree = SumTree(size=args.buffer_size)

#         # PER params
#         self.eps = args.per_eps
#         self.alpha = args.per_alpha
#         self.beta = args.per_beta
#         self.max_priority = 1.0

#     def add(self, state, actions, reward, next_state, done):
#         batch_size = actions.shape[0]
#         idxs = (np.arange(batch_size) + self.count) % self.size

#         for i, (array, array_) in enumerate(zip(state, next_state)):
#             self.state_storage[i][idxs] = array
#             self.next_state_storage[i][idxs] = array_

#         # add priorities
#         for idx in idxs:
#             self.tree.add(self.max_priority, idx)

#         self.action[idxs] = actions
#         self.reward[idxs] = reward
#         self.done[idxs] = done

#         self.count = (self.count + batch_size) % self.size
#         self.real_size = min(self.size, self.real_size + batch_size)


#     def sample(self, batch_size):
#         assert self.real_size >= batch_size

#         idxs, tree_idxs = [], []
#         priorities = np.empty(batch_size, dtype=np.float32)

#         segment = self.tree.total / batch_size

#         for i in range(batch_size):
#             a, b = segment * i, segment * (i + 1)
#             cumsum = random.uniform(a, b)

#             tree_idx, priority, idx = self.tree.get(cumsum)

#             priorities[i] = priority
#             tree_idxs.append(tree_idx)
#             idxs.append(idx)

#         idxs = np.array(idxs)

#         probs = priorities / self.tree.total

#         weights = (self.real_size * probs) ** (-self.beta)
#         weights /= weights.max()

#         # Extract state
#         states = [storage[idxs] for storage in self.state_storage]
#         next_states = [storage[idxs] for storage in self.next_state_storage]

#         # Extract obs, next_obs and actions for potential mirroring
#         obs = self.encode(states)
#         next_obs = self.encode(next_states)
#         actions = self.action[idxs]

#         # Mirror the obs, next_obs and actions
#         if np.random.rand() < self.mirror_prob:  # Randomly decide to mirror or not
#             obs, actions, next_obs = mirror(obs, actions, next_obs)

#         obs = torch.from_numpy(obs).to(self.device).float()
#         actions = torch.from_numpy(actions).to(self.device).long()
#         rewards = torch.from_numpy(self.reward[idxs]).to(self.device).unsqueeze(1)
#         next_obs = torch.from_numpy(next_obs).to(self.device).float()
#         dones = torch.from_numpy(self.done[idxs]).to(self.device).unsqueeze(1)
#         weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

#         return obs, actions, rewards, next_obs, dones, weights, tree_idxs

#     def update(self, data_idxs, priorities):
#         if isinstance(priorities, torch.Tensor):
#             priorities = priorities.detach().cpu().numpy()

#         for data_idx, priority in zip(data_idxs, priorities):
#             priority = (priority + self.eps) ** self.alpha

#             self.tree.update(data_idx, priority)
#             self.max_priority = max(self.max_priority, priority)


# class SumTree:

#     def __init__(self, size):
#         self.nodes = [0] * (2 * size - 1)
#         self.data = [None] * size

#         self.size = size
#         self.count = 0
#         self.real_size = 0

#     @property
#     def total(self):
#         return self.nodes[0]

#     def update(self, data_idx, value):
#         idx = data_idx + self.size - 1  # child index in tree array
#         change = value - self.nodes[idx]

#         self.nodes[idx] = value

#         parent = (idx - 1) // 2
#         while parent >= 0:
#             self.nodes[parent] += change
#             parent = (parent - 1) // 2

#     def add(self, value, data):
#         self.data[self.count] = data
#         self.update(self.count, value)

#         self.count = (self.count + 1) % self.size
#         self.real_size = min(self.size, self.real_size + 1)

#     def get(self, cumsum):
#         assert cumsum <= self.total

#         idx = 0
#         while 2 * idx + 1 < len(self.nodes):
#             left, right = 2*idx + 1, 2*idx + 2

#             if cumsum <= self.nodes[left]:
#                 idx = left
#             else:
#                 idx = right
#                 cumsum = cumsum - self.nodes[left]

#         data_idx = idx - self.size + 1

#         return data_idx, self.nodes[idx], self.data[data_idx]

#     def __repr__(self):
#         return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


class PrioritizedReplayBuffer:
    def __init__(self, state_example, state_encode_fn: callable, player: int, args, device):
        assert player in [0, 1]

        self.device = device
        self.size = args.buffer_size
        self.mirror_prob = args.mirror_prob

        self.encode = state_encode_fn

        # PER params
        self.eps = args.per_eps
        self.alpha = args.per_alpha
        self.beta = args.per_beta

        self.max_priority = 1.0

        # storage (match your existing architecture)
        self.state_storage = []
        self.next_state_storage = []

        for element in state_example:
            if isinstance(element, np.ndarray):
                shape, dtype = element.shape[1:], element.dtype
                self.state_storage.append(np.empty((self.size, *shape), dtype=dtype))
                self.next_state_storage.append(np.empty((self.size, *shape), dtype=dtype))
            else:
                self.state_storage.append(np.empty((self.size,), dtype=type(element)))
                self.next_state_storage.append(np.empty((self.size,), dtype=type(element)))

        self.action = np.empty((self.size, 2), dtype=np.int64)
        self.reward = np.empty((self.size,), dtype=np.float32)
        self.done = np.empty((self.size,), dtype=np.bool_)

        # PER priorities
        self.priorities = np.zeros((self.size,), dtype=np.float32)

        self.ptr = 0
        self.real_size = 0

    def add(self, state, actions, reward, next_state, done):
        batch_size = actions.shape[0]
        idxs = (np.arange(batch_size) + self.ptr) % self.size

        for i, (s, ns) in enumerate(zip(state, next_state)):
            self.state_storage[i][idxs] = s
            self.next_state_storage[i][idxs] = ns

        self.action[idxs] = actions
        self.reward[idxs] = reward
        self.done[idxs] = done

        # initialize priorities safely (never zero)
        self.priorities[idxs] = self.max_priority

        self.ptr = (self.ptr + batch_size) % self.size
        self.real_size = min(self.size, self.real_size + batch_size)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        # --- stable probability distribution ---
        priorities = self.priorities[:self.real_size]

        probs = priorities / np.maximum(priorities.sum(), 1e-8)

        idxs = np.random.choice(self.real_size, batch_size, replace=True, p=probs)

        # importance sampling weights
        weights = (self.real_size * probs[idxs])
        weights = np.maximum(weights, 1e-8)
        weights = weights ** (-self.beta)
        weights /= np.max(weights) + 1e-8

        # encode batch
        states = [s[idxs] for s in self.state_storage]
        next_states = [ns[idxs] for ns in self.next_state_storage]

        obs = self.encode(states)
        next_obs = self.encode(next_states)
        actions = self.action[idxs]

        # Mirror the obs, next_obs and actions
        if np.random.rand() < self.mirror_prob:  # Randomly decide to mirror or not
            obs, actions, next_obs = mirror(obs, actions, next_obs)

        rewards = self.reward[idxs]
        dones = self.done[idxs]

        # tensors
        obs = torch.from_numpy(obs).to(self.device).float()
        next_obs = torch.from_numpy(next_obs).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device).long()
        rewards = torch.from_numpy(rewards).to(self.device).unsqueeze(1)
        dones = torch.from_numpy(dones).to(self.device).unsqueeze(1)

        weights = torch.from_numpy(weights).to(self.device).unsqueeze(1)

        return obs, actions, rewards, next_obs, dones, weights, idxs

    def update(self, idxs, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        td_errors = np.abs(td_errors) + self.eps
        td_errors = td_errors ** self.alpha

        self.priorities[idxs] = td_errors
        self.max_priority = max(self.max_priority, td_errors.max())

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
    
    
