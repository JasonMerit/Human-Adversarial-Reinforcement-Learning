import numpy as np


class VecPoLEnv:
    """
    Vectorized PoL environment using standard grid tensors.
    Stable, debuggable, MCTS-friendly.
    """

    def __init__(self, num_envs, size):
        self.num_envs = num_envs
        self.size = size

        # 0: up, 1: right, 2: down, 3: left
        self.dirs = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)

        self.pos = np.zeros((self.num_envs, 2), dtype=np.int16)
        self.walls = np.zeros((self.num_envs, self.size, self.size), dtype=np.int8)
        self.done = np.zeros(self.num_envs, dtype=bool)

        # goal is bottom-right
        self.goal = np.array([self.size - 1, self.size - 1], dtype=np.int16)

        return self._obs()

    def step(self, actions):
        assert actions.shape == (self.num_envs,), f"Wrong actions shape, expected ({self.num_envs},)"
        actions = np.asarray(actions, dtype=np.int8)

        # ---- POTENTIAL (before move) ----
        old_dist = np.abs(self.pos - self.goal).sum(axis=1)

        # mark visited (only active envs)
        for i in range(self.num_envs):
            if self.done[i]:
                continue
            y, x = self.pos[i]
            self.walls[i, y, x] = 1

        # ---- dynamics ----
        delta = self.dirs[actions]
        new_pos = self.pos + delta

        # bounds check
        out_of_bounds = (
            (new_pos[:, 0] < 0) |
            (new_pos[:, 0] >= self.size) |
            (new_pos[:, 1] < 0) |
            (new_pos[:, 1] >= self.size)
        )

        # clip for safe indexing
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, self.size - 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, self.size - 1)
        self.pos = new_pos

        # collisions
        crash = np.array([
            self.walls[i, new_pos[i, 0], new_pos[i, 1]]
            for i in range(self.num_envs)
        ], dtype=bool)

        goal = np.all(new_pos == self.goal, axis=1)
        
        new_dist = np.abs(new_pos - self.goal).sum(axis=1)
        shaping = (-new_dist) - (-old_dist)

        reward = shaping.astype(np.float32) * .1
        reward[goal] += 1.0
        reward[crash] -= 1.0

        done = goal | crash | out_of_bounds
        self.done |= done

        return self._obs(), reward, done, {}

    def _obs(self):
        obs = np.zeros((self.num_envs, 3, self.size, self.size), dtype=np.float32)

        for i in range(self.num_envs):
            obs[i, 0] = self.walls[i]

            y, x = self.pos[i]
            obs[i, 1, y, x] = 1.0

            gy, gx = self.goal
            obs[i, 2, gy, gx] = 1.0

        return obs


if __name__ == "__main__":
    import time
    from tqdm import trange
    
    num_envs=2
    envs = VecPoLEnv(num_envs, 25)
    envs.reset()
    
    t0 = time.time()
    for _ in range(10):
        actions = np.random.randint(0, 3, num_envs)
        obs, reward, done, result = envs.step(actions)
        print(reward)
        
        # if done.any():
        #     envs.reset(done)

    print(f"Finished 1000 * 512 = {1000*512:,} steps after {time.time() - t0:.2f} seconds")