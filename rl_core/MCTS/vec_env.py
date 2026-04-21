import numpy as np


class Result:
    DRAW = 0
    BIKE2_CRASH = 1
    BIKE1_CRASH = 2
    PLAYING = -1


class TronVecEnv:

    action_mapping = np.array([
        [0, -1],   # up
        [1, 0],    # right
        [0, 1],    # down
        [-1, 0],   # left
    ], dtype=np.int8)

    reward_dict = {
        Result.DRAW: 0,
        Result.BIKE2_CRASH: 1,
        Result.BIKE1_CRASH: -1,
        Result.PLAYING: 0,
    }

    def __init__(self, num_envs=128, size=25):

        self.num_envs = num_envs
        self.size = size
        self.area = size * size

        self.walls = np.zeros((num_envs, size, size), dtype=np.int8)

        self.pos1 = np.zeros((num_envs, 2), dtype=np.int16)
        self.pos2 = np.zeros((num_envs, 2), dtype=np.int16)

        self.heading1 = np.zeros(num_envs, dtype=np.int8)
        self.heading2 = np.zeros(num_envs, dtype=np.int8)

        self.done = np.zeros(num_envs, dtype=bool)

        self.reset()

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------

    def reset(self, mask=None):

        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)

        idx = np.where(mask)[0]

        self.walls[idx] = 0

        self.pos1[idx] = np.array([self.size // 6, self.size // 2])
        self.pos2[idx] = np.array([5 * self.size // 6, self.size // 2])

        self.heading1[idx] = 1
        self.heading2[idx] = 3

        self.done[idx] = False

        y1 = self.pos1[idx, 1]
        x1 = self.pos1[idx, 0]

        y2 = self.pos2[idx, 1]
        x2 = self.pos2[idx, 0]

        self.walls[idx, y1, x1] = 1
        self.walls[idx, y2, x2] = 2

        return self.get_obs()

    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------

    def step(self, actions):

        a1 = actions[:, 0]
        a2 = actions[:, 1]

        self.heading1 = (self.heading1 + (a1 - 1)) % 4
        self.heading2 = (self.heading2 + (a2 - 1)) % 4

        dir1 = self.action_mapping[self.heading1]
        dir2 = self.action_mapping[self.heading2]

        new_pos1 = self.pos1 + dir1
        new_pos2 = self.pos2 + dir2

        x1 = new_pos1[:, 0]
        y1 = new_pos1[:, 1]

        x2 = new_pos2[:, 0]
        y2 = new_pos2[:, 1]

        # ---------------------------
        # Out of bounds
        # ---------------------------

        oob1 = (x1 < 0) | (x1 >= self.size) | (y1 < 0) | (y1 >= self.size)
        oob2 = (x2 < 0) | (x2 >= self.size) | (y2 < 0) | (y2 >= self.size)

        # ---------------------------
        # Flat indexing collision
        # ---------------------------

        flat = self.walls.reshape(self.num_envs, -1)

        idx1 = y1 * self.size + x1
        idx2 = y2 * self.size + x2

        hit1 = np.zeros(self.num_envs, dtype=bool)
        hit2 = np.zeros(self.num_envs, dtype=bool)

        valid1 = ~oob1
        valid2 = ~oob2

        hit1[valid1] = flat[np.arange(self.num_envs)[valid1], idx1[valid1]] != 0
        hit2[valid2] = flat[np.arange(self.num_envs)[valid2], idx2[valid2]] != 0

        bike1_crash = hit1 | oob1
        bike2_crash = hit2 | oob2

        same_cell = np.all(new_pos1 == new_pos2, axis=1)

        draw = (bike1_crash & bike2_crash) | same_cell
        bike1_only = bike1_crash & ~bike2_crash
        bike2_only = bike2_crash & ~bike1_crash

        result = np.full(self.num_envs, Result.PLAYING, dtype=np.int8)
        result[draw] = Result.DRAW
        result[bike1_only] = Result.BIKE1_CRASH
        result[bike2_only] = Result.BIKE2_CRASH

        alive = result == Result.PLAYING

        self.pos1[alive] = new_pos1[alive]
        self.pos2[alive] = new_pos2[alive]

        y1 = self.pos1[:, 1]
        x1 = self.pos1[:, 0]

        y2 = self.pos2[:, 1]
        x2 = self.pos2[:, 0]

        flat[np.arange(self.num_envs), y1 * self.size + x1] = 1
        flat[np.arange(self.num_envs), y2 * self.size + x2] = 2

        self.done = result != Result.PLAYING

        reward = np.vectorize(self.reward_dict.get)(result)

        return self.get_obs(), reward, self.done, result

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------

    def get_obs(self):

        walls = self.walls

        bike1 = np.zeros_like(walls)
        bike2 = np.zeros_like(walls)

        bike1[np.arange(self.num_envs), self.pos1[:, 1], self.pos1[:, 0]] = 1
        bike2[np.arange(self.num_envs), self.pos2[:, 1], self.pos2[:, 0]] = 1

        obs1 = np.stack([walls, bike1, bike2], axis=1)
        obs2 = np.stack([walls, bike2, bike1], axis=1)

        obs = np.stack([obs1, obs2], axis=1)

        return obs.astype(np.float32)


# -------------------------------------------------------
# simple speed test
# -------------------------------------------------------

if __name__ == "__main__":
    import time
    from tqdm import trange
    
    num_envs=512
    envs = TronVecEnv(num_envs, size=25)
    envs.reset()
    
    t0 = time.time()
    for _ in trange(1000):
        actions = np.random.randint(0, 3, (512, 2))
        obs, reward, done, result = envs.step(actions)
        
        if done.any():
            envs.reset(done)

    print(f"Finished 1000 * 512 = {1000*512:,} steps after {time.time() - t0:.2f} seconds")
    
    # from rl_core.env import TronDuoEnv
    # env = TronDuoEnv(25)
    # env.reset()

    # t0 = time.time()
    # for _ in trange(1000*num_envs):
    #     actions = np.random.randint(0, 3, (2,))
    #     obs, reward, done, _, info = env.step(actions)
    
    # print(f"Finished 512,000 steps after {time.time() - t0:.2f} seconds")
