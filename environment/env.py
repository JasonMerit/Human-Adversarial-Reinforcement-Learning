import numpy as np
import gymnasium as gym

from environment.tron import Tron

class TronEnv(gym.Env):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=int)  # up, right, down, left
    reward_mapping = [0.0, -1.0, 1, 0.5]  # playing, lose, win, draw

    def __init__(self, opponent, width, height):
        self.tron = Tron(width, height)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(height, width, 3), dtype=float)
        self.opponent = opponent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
        self.tron.tick(self.action_mapping[1], self.action_mapping[3])
        self.opponent.reset(seed=seed)
            
        return self._get_state(), {'result': 0}
    
    def step(self, action):
        assert self.action_space.contains(action), "Jason! Invalid Action"
        dir1 = self.action_mapping[action]
        dir2 = self.opponent.get_direction(self.tron.walls, self.tron.bike2.pos)  # Bike 2
    
        result = self.tron.tick(dir1, dir2)
        done = result != 0
        state = self._get_state(done)
        reward = self.reward_mapping[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self, done=False):
        walls = self.tron.walls.copy()
        occ = (walls > 0).astype(float)

        bike1 = np.zeros_like(occ)
        bike2 = np.zeros_like(occ)  
        if not done:
            x1, y1 = self.tron.bike1.pos
            bike1[y1, x1] = 1.0

            x2, y2 = self.tron.bike2.pos
            bike2[y2, x2] = 1.0  # Out of bounds - Skip if done.

        # Stack into CNN input
        state = np.stack([occ, bike1, bike2], axis=-1)
        assert state.shape == self.observation_space.shape, "Jason! State shape mismatch"
        return state
    
    
