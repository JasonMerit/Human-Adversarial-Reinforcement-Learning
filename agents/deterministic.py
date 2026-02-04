import numpy as np

class DeterministicAgent:
    action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=int)  # up, right, down, left

    def __init__(self, start_left=False):
        self.last_action = self.first_action = 3 if start_left else 1
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
        walls = state[0]
        head = state[1]  # Bike 1

        try:
            y, x = np.argwhere(head == 1)[0]  # Flipped coordinates
            pos = np.array([x, y], dtype=int)
        except IndexError:
            return self.first_action  # Terminal state - pos is out of bounds
        return self._get_action(walls, pos)

    def _is_valid_action(self, action, walls, pos):
        new_pos = pos + self.action_mapping[action]
        x, y = new_pos
        return not (not 0 <= y < len(walls) or not 0 <= x < len(walls[0]) or walls[y, x] != 0)

class Random:

    def __init__(self, env):
        self.action_space = env.action_space

    def compute_single_action(self, state):
        return self.action_space.sample()

    def reset(self, seed=None):
        self.np_random = np.random.RandomState(seed)