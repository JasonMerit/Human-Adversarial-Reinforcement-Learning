import numpy as np
from rl_core.utils.heuristics import chamber_heuristic
from gymnasium import spaces
from rl_core.utils.helper import bcolors
from .base import Agent

class HeuristicAgent(Agent):
    """
    Tron agent that selects moves by evaluating the chamber_heuristic
    for all possible actions and picking the one with highest score.
    """
    def __init__(self):
        self.is_opponent = True

    def reset(self, seed=None):
        pass

    def __call__(self, state : tuple):
        """
        Returns the action that maximizes chamber_heuristic.
        """
        walls, player, opp = state
        if self.is_opponent:
            player, opp = opp, player

        best_score = -np.inf
        best_action = None

        for action, dir in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            # Simulate the move
            new_pos = player + dir

            # Skip invalid moves (into walls or out of bounds)
            if not(0 <= new_pos[0] < walls.shape[1]) or \
               not(0 <= new_pos[1] < walls.shape[0]) or \
               walls[new_pos[1], new_pos[0]] != 0:
                continue

            # Evaluate heuristic
            score = chamber_heuristic(walls, new_pos, opp)

            if score > best_score:
                best_score = score
                best_action = action

        # Return the best move
        if best_action is None:
            # If no valid moves, just pick up (or any default)
            return 0
        return best_action
    
    def _check_env(self, env):
        self.is_opponent = False
        if not isinstance(env.observation_space, spaces.Tuple):
            raise ValueError(f"{bcolors.FAIL}HeuristicAgent requires an environment with a tuple observation space. Try removing wrappers.{bcolors.ENDC}")    