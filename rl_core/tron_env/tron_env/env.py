import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import gymnasium as gym

from .tron import Tron, Result
from . import utils
from .heuristic import get_best_action
# from rl_core.agents import Agent

class TronEnv(gym.Env):
    """Wraps TronEnvBase with all the wrappers"""

    def __init__(self, size=(11, 11)):
        super().__init__()
        from .wrappers import TronImage, TronEgo
        env = TronEgo(TronImage(TronEnvBase(size)))
        self.env = env
        self.tron = env.unwrapped.tron
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

class TronEnvBase(gym.Env):

    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)  # up, right, down, left
    reward_dict = { Result.DRAW: 0, Result.BIKE2_CRASH: -1, Result.BIKE1_CRASH: 1, Result.PLAYING: 0 }

    def __init__(self, size=(11, 11)):
        self.tron = Tron(size)
        self.width, height = size

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=2, shape=(height, self.width), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width-1, height-1]), shape=(2,), dtype=np.int8),
                gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.width-1, height-1]), shape=(2,), dtype=np.int8)
            ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tron.reset()
            
        return self._get_state(), {'result': 0}
    
    def step(self, action : int):
        assert self.action_space.contains(action), utils.red(f"Jason! Invalid Action {action}")
        
        dir1 = self.action_mapping[get_best_action(self._get_state())]  # Human's action
        dir2 = self.action_mapping[action]
    
        result = self.tron.tick(dir1, dir2)
        done = result != Result.PLAYING
        state = self._get_state()
        reward = self.reward_dict[result]
        info = {'result': result}
        return state, reward, done, False, info
    
    def _get_state(self):
        walls, you, opp = self.tron.walls, self.tron.bike1.pos, self.tron.bike2.pos
        return walls, you, opp

    

