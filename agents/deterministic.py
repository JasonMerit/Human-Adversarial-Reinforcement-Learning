import numpy as np
from agents.base import Agent
from utils.helper import bcolors
from gymnasium import spaces
from environment.env import TronDualEnv

class DeterministicAgent(Agent):
    def __init__(self):
        self.last_action = self.first_action = 3
        self.is_opponent = True
        self.np_random = np.random.RandomState()

    def reset(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.last_action = self.first_action

    def _get_action(self, walls, pos):
        action = self.last_action
        possible_actions = {0, 1, 2, 3} - {action, (action + 2) % 4}

        while not self._is_valid_action(action, walls, pos) and len(possible_actions) > 0:
            action = self.np_random.choice(list(possible_actions))
            possible_actions.remove(action)

        self.last_action = action
        return action
    
    def __call__(self, state):
        """Assumed call from player"""
        walls, player, opp = state
        pos = player if not self.is_opponent else opp
        return self._get_action(walls, pos)

    def _is_valid_action(self, action, walls, pos):
        new_pos = pos + TronDualEnv.action_mapping[action]
        x, y = new_pos
        return not (not 0 <= y < len(walls) or not 0 <= x < len(walls[0]) or walls[y, x] != 0)

    def _check_env(self, env):
        # Hack done to distinguish player from opponent
        self.last_action = self.first_action = 3  
        self.is_opponent = False

        if not isinstance(env.observation_space, spaces.Tuple):
            raise ValueError(f"{bcolors.FAIL}DeterministicAgent requires an environment with a tuple observation space. Try removing wrappers.{bcolors.ENDC}")    