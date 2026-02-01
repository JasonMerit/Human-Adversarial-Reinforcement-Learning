import numpy as np

from environment.tron import Bike

class DeterministicAgent:
    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=int)  # up, right, down, left

    def __init__(self, first_action):
        self.last_action = self.first_action = first_action
        self.np_random = np.random.RandomState()

    def reset(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.last_action = self.first_action

    def get_direction(self, walls, pos):
        """
        Semi-deterministic agent that repeats its last action unless that action is invalid.
        In that case, it randomly selects a valid action.

        :return: Direction tuple (dx, dy) for opponent bike
        """
        return self.action_mapping[self._get_action(walls, pos)]

    def _get_action(self, walls, pos):
        action = self.last_action
        possible_actions = {0, 1, 2, 3} - {action, (action + 2) % 4}

        while not self._is_valid_action(action, walls, pos) and len(possible_actions) > 0:
            action = self.np_random.choice(list(possible_actions))
            possible_actions.remove(action)

        self.last_action = action
        return action

    def compute_single_action(self, state):
        walls = state[:, :, 0]
        head = state[:, :, 1]  # Bike 1
        try:
            y, x = np.argwhere(head == 1)[0]  # Flipped coordinates
            pos = np.array([x, y])
        except IndexError:
            return self.first_action  # Terminal state - pos is out of bounds
        return self._get_action(walls, pos)

    def _is_valid_action(self, action, walls, pos):
        new_pos = pos + self.action_mapping[action]
        return not Bike.is_hit(new_pos, walls)