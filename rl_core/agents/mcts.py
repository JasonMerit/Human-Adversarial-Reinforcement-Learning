import numpy as np
from rl_core.utils.heuristics import voronoi
from gymnasium import spaces
from rl_core.utils.helper import bcolors
from .base import Agent

class HeuristicAgent(Agent):
    """
    Tron agent that selects moves by evaluating the chamber_heuristic
    for all possible actions and picking the one with highest score.
    """
    def reset(self, seed=None):
        pass

    def __call__(self, state : tuple):
        """
        Returns the action that maximizes chamber_heuristic.
        """
        trails, you, other = state

        best_score = -np.inf
        best_action = None
        for action, dir in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            new_pos = you + dir

            if not(0 <= new_pos[0] < trails.shape[1]) or \
               not(0 <= new_pos[1] < trails.shape[0]) or \
               trails[new_pos[1], new_pos[0]] != 0:
                continue

            score = voronoi(trails, new_pos, other)

            if score > best_score:
                best_score = score
                best_action = action
            
        if best_action is None:
            return 0  # If no valid moves, just pick up (or any default)
        return best_action
    
    def _check_env(self, env):
        if not isinstance(env.observation_space, spaces.Tuple):
            raise ValueError(f"{bcolors.FAIL}HeuristicAgent requires an environment with a tuple observation space. Try removing wrappers.{bcolors.ENDC}")    
    
