import numpy as np
from rl_core.env import GameState
from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv


class Node:
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # agent action leading here

        self.children = [None] * 3

        self.N = 0
        self.W = 0.0

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def is_fully_expanded(self):
        return not any(c is None for c in self.children)


class MCTS:
    """Assumes state provided is never itself terminal, since state and next-state is stored in buffer"""
    def __init__(self, player, policy, adv_policy, envs: VecTronDuoEnv, sims=200, c=1.4):
        self.player = player
        self.policy = policy
        self.adv_policy = adv_policy

        self.envs = envs

        self.sims = sims
        self.c = c

    def __call__(self, state: GameState) -> np.ndarray:
        assert isinstance(state, GameState), f"Expected input to be a GameState, got {type(state)}"
        root = Node(state)

        for _ in range(self.sims):
            self.simulate(root)

        return np.array([child.Q for child in root.children])

    def simulate(self, node: Node):
        while node.is_fully_expanded():
            node = self.select(node)

        node = self.expand(node)
        value = self.rollout_vec(node.state, node.action)
        self.backup(node, value)

    def select(self, node):
        best_score = -1e9
        best_child = None

        for child in node.children:
            if child is None:
                continue

            if child.N == 0:
                return child

            uct = child.Q + self.c * np.sqrt(np.log(node.N + 1) / child.N)

            if uct > best_score:
                best_score = uct
                best_child = child

        return best_child

    def expand(self, node):
        for a in range(3):
            if node.children[a] is None:
                child = Node(state=node.state, parent=node, action=a)
                node.children[a] = child
                return child

        raise Exception("fully expanded")

    def rollout_vec(self, state, action):
        self.envs.set_state(state)
        obs = self.envs.get_obs()
        actions = np.empty((self.envs.num_envs, 2), dtype=a1.dtype)

        a1 = np.full(self.envs.num_envs, action)
        a2 = self.adv_policy(obs[:, 1 - self.player])
        actions[:, self.player], actions[:, 1 - self.player] = a1, a2

        obs, total_rewards, done, _, _ = self.envs.step(actions)

        active = ~done
        while active.any():

            a1 = self.policy(obs[:, self.player])
            a2 = self.adv_policy(obs[:, 1 - self.player])
            actions[:, self.player], actions[:, 1 - self.player] = a1, a2

            obs, r, d, _, _ = self.envs.step(actions)

            total_rewards[active] += r[active]

            active = np.logical_and(active, ~d)

        return total_rewards.mean()

    def backup(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent