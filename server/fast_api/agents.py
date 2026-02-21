import numpy as np

class DeterministicAgent():
    def __init__(self):
        self.last_action = self.first_action = 1
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
        walls, you, _ = state
        return self._get_action(walls, you)

    def _is_valid_action(self, action, walls, pos):
        action_mapping = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int8)
        new_pos = pos + action_mapping[action]
        x, y = new_pos
        return not (not 0 <= y < len(walls) or not 0 <= x < len(walls[0]) or walls[y, x] != 0)

class SemiDeterministicAgent(DeterministicAgent):
    def __init__(self, random_prob):
        super().__init__()
        self.random_prob = random_prob

    def _get_action(self, walls, pos):
        if self.np_random.random() < self.random_prob:  # 10% chance to be random
            action = self.np_random.choice(range(4))
            possible_actions = set(range(4)) - {action}
        else:
            action = self.last_action
            possible_actions = set(range(4)) - {action, (action + 2) % 4}

        while not self._is_valid_action(action, walls, pos) and len(possible_actions) > 0:
            action = self.np_random.choice(list(possible_actions))
            possible_actions.remove(action)

        self.last_action = action
        return action