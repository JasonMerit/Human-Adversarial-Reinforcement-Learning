import numpy as np
from environment.env import TronEnv
from utils.heuristics import chamber_heuristic

class HeuristicAgent():
    """
    Tron agent that selects moves by evaluating the chamber_heuristic
    for all possible actions and picking the one with highest score.
    """

    def __call__(self, walls : np.ndarray, player : np.ndarray, opponent : np.ndarray):
        """
        Returns the action that maximizes chamber_heuristic.
        """
        best_score = -np.inf
        best_action = None

        for action, dir in enumerate(TronEnv.action_mapping):
            # Simulate the move
            new_pos = player + dir

            # Skip invalid moves (into walls or out of bounds)
            if new_pos[0] < 0 or new_pos[0] >= walls.shape[1]:
                continue
            if new_pos[1] < 0 or new_pos[1] >= walls.shape[0]:
                continue
            if walls[new_pos[1], new_pos[0]] != 0:
                continue

            # Evaluate heuristic
            score = chamber_heuristic(walls, new_pos, opponent)

            if score > best_score:
                best_score = score
                best_action = action

        # Return the best move
        if best_action is None:
            # If no valid moves, just pick up (or any default)
            return 0
        return best_action
