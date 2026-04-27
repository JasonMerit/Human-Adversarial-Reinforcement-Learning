import os, time
import numpy as np
from rl_core.agents.utils import TimerRegistry

class VecPoLEnv:
    """
    Vectorized PoL environment using standard grid tensors.
    Stable, debuggable, MCTS-friendly.
    """
    def __init__(self, num_envs, size, render=False):
        self.num_envs = num_envs
        self.size = size
        self.render = render

        if render:
            os.system('cls')

        self.obs_shape = (3, size, size)
        self.n_actions = 4  # up, right, down, left
        self.state_shape = (self.num_envs, 3, self.size, self.size)

        # 0: up, 1: right, 2: down, 3: left
        self.dirs = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)
        self.goal = np.array([self.size - 1, self.size - 1], dtype=np.int64)
        self.pos = np.zeros((num_envs, 2), dtype=np.int64)
        self.walls = np.zeros((num_envs, size, size), dtype=np.int8)
        self.env_ids = np.arange(num_envs)
    
    def sample_actions(self):
        return np.random.randint(self.n_actions, size=self.num_envs)

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        assert mask.shape == (self.num_envs,)
        assert mask.dtype == bool

        self.pos[mask] = 0
        self.walls[mask] = 0

        return VecPoLEnv.encode(self.state), {"state": self.state} 

    @TimerRegistry.wrap_fn("vec_pol.step")
    def step(self, actions: np.ndarray):
        assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
        assert actions.shape == (self.num_envs,), f"Wrong actions shape, expected ({self.num_envs},)"
        actions = np.asarray(actions, dtype=np.int8)

        # ---- POTENTIAL (before move) ----
        old_dist = np.abs(self.pos - self.goal).sum(axis=1)

        # Mark current position before moving
        self.walls[self.env_ids, self.pos[:,0], self.pos[:,1]] = 1

        # ---- dynamics ----
        delta = self.dirs[actions]
        new_pos = self.pos + delta
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, self.size - 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, self.size - 1)
        self.pos = new_pos

        # collisions
        crash = self.walls[self.env_ids, new_pos[:,0], new_pos[:,1]].astype(bool)
        is_goal = np.all(new_pos == self.goal, axis=1)
        
        new_dist = np.abs(new_pos - self.goal).sum(axis=1)
        shaping = old_dist - new_dist

        reward = shaping * .1
        reward[is_goal] = 1.0
        reward[crash] = -1.0

        done = is_goal | crash
        
        self.reset(mask=done)  # Auto reset done envs

        if self.render:
            self.view()  # render

        infos = {"state": self.state}
        return VecPoLEnv.encode(self.state), reward, done, None, infos
    
    def view(self):
        print("\033[H", end="")

        for i in range(self.num_envs):
            board = np.full((self.size, self.size), ".", dtype=str)
            board[self.walls[i] == 1] = "#"
            board[self.goal[0], self.goal[1]] = "G"
            board[self.pos[i][0], self.pos[i][1]] = "A"

            for row in board:
                print(" ".join(row))
            print("="*self.size*2)
        time.sleep(0.1)
    
    def set_state(self, state):
        assert isinstance(state, tuple), f"Expected tuple, got {type(state)}"
        walls, pos = state
        assert walls.shape == (self.size, self.size), f"Expected shape {(self.num_envs, self.size, self.size)}, got {walls.shape}"
        assert pos.shape == (2,), f"Expected shape {(2,)}, got {pos.shape}"

        # Duplicate the walls num_envs times
        self.walls = np.array([walls.copy() for _ in range(self.num_envs)], dtype=np.int8)
        self.pos = np.array([pos.copy() for _ in range(self.num_envs)], dtype=np.int64)
    
    def set_states(self, states):
        assert isinstance(states, tuple), f"Expected tuple, got {type(states)}"
        walls, pos = states
        assert walls.shape == (self.num_envs, self.size, self.size), f"Expected shape {(self.num_envs, self.size, self.size)}, got {walls.shape}"
        assert pos.shape == (self.num_envs, 2), f"Expected shape {(self.num_envs, 2)}, got {pos.shape}"

        self.walls = walls
        self.pos = pos

    def close(self):
        pass

    @staticmethod
    def encode(state):
        walls, pos = state
        num_envs, size, _ = walls.shape
        obs = np.zeros((num_envs, 3, size, size), dtype=np.float32)

        # walls channel
        obs[:, 0] = walls

        # agent positions
        env_ids = np.arange(num_envs)
        obs[env_ids, 1, pos[:, 0], pos[:, 1]] = 1.0

        # goal (same for all envs)
        goal = np.array([size - 1, size - 1], dtype=np.int64)
        obs[:, 2, goal[0], goal[1]] = 1.0

        return obs
    
    @property
    def state(self):
        return self.walls.copy(), self.pos.copy()

if __name__ == "__main__":
    from tqdm import trange
    
    num_envs=2
    envs = VecPoLEnv(num_envs, 5, True)
    envs.reset()
    np.random.seed(21)
    
    steps = 0
    copy_state = None
    while True:
    # for i in range(5):
        steps += 1
        # print(f"=== {i} ====")
        actions = np.random.randint(1, 3, num_envs)
        obs, reward, done, _, infos = envs.step(actions)
        if steps == 5:
            # print(obs)
            state = infos["state"]
            # print(state)
            obs = VecPoLEnv.encode(state)

            break
            # copy_state = obs
        if steps > 5 and steps % 5 == 0:
            envs.set_state(copy_state)
        # actions = np.array([0])
        # actions = np.array([1]*num_envs, dtype=np.int8)  # always move right
        
        if done.all():
            break
