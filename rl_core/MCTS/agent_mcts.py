import numpy as np
# from rich import print

from rl_core.env import TronDuoEnv, GameState
from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv
from rl_core.utils import TimerRegistry
from rl_core.env.heuristic import voronoi

class Node:
    
    def __init__(self, state: GameState, parent=None, action=None, terminal=False, reward=0, n_actions=3):
        assert isinstance(state, GameState), f"Expected state to be a GameState, got {type(state)}"
        self.state = state
        self.parent = parent
        self.action = action # Debugging
        self.terminal = terminal
        self.reward = reward

        self.children = [None for _ in range(3)]

        self.N = 0
        self.W = 0.0
    
    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N
    
    # @property
    # def is_expanded(self):
    #     return len(self.untried) == 0
    
    def add_child(self, joint_action, node):
        a1, a2 = joint_action
        self._children[a1][a2] = node
    
    def q_values(self):
        # Construct 3x3 array of Q values for all joint actions, using 0.0 for unvisited actions
        q_values = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                child = self._children[i][j]
                if child is None or child.N == 0:
                    raise ValueError("Child node from root is None or has not been visited")
                q_values[i, j] = child.Q
                # q_values[i, j] = 0.0 if child is None else child.Q
        return q_values

class MCTS:
    """Returns only interested in terminal states, otherwise value must be cumulative discounted when backup"""

    def __init__(self, policy: callable, adv_policy: callable, env: TronDuoEnv, envs: VecTronDuoEnv, rollouts: int, horizon=200):
        self.policy = policy
        self.adv_policy = adv_policy
        self.env = env  # For structured search
        self.envs = envs  # For structured search
        self.rollouts = rollouts
        self.horizon = horizon

    @TimerRegistry.wrap_fn("MCTS.simulate_q_values")
    def __call__(self, state, sims=400):
        root = Node(state)
        for _ in range(sims):
            self.simulate(root)

        return root.q_values()

    def simulate(self, root: Node):
        node = root

        # selection
        while node.is_expanded and not node.terminal:  # fully expanded and non-terminal
            node = self.select(node)

        # expansion
        if not node.is_expanded and not node.terminal:  # non-terminal and not fully expanded
            node = self.expand(node)

        # evaluation
        value = node.reward if node.terminal else self.rollout_vec(node)

        # backup            
        self.backup(node, value)

    def select(self, node, c=1.4) -> Node:
        """Select child with highest UCT value."""
        best = None
        best_score = -1e9

        for child in node.children:
            if child.N == 0:
                return child
            uct = child.W/child.N + c*np.sqrt(np.log(node.N)/child.N)
            if uct > best_score:
                best_score = uct
                best = child

        # print(f"Selecting {best.action} ({best_score:.2f})")
        return best

    def expand(self, node):
        for a in range(3):

            state, reward, done = self.env.simulate(node.state, a, b)

            child = Node(
                state=state,
                parent=node,
                action=(a, b),
                reward=reward,
                terminal=done
            )

            node.children.append(child)

    
    # @TimerRegistry.wrap_fn("MCTS.rollout_vec")
    # def rollout_vec(self, node: Node):
    #     self.envs.set_state(node.state)
    #     obs = self.envs.encode(self.envs.state)

    #     runs = 0
    #     total = 0.0
    #     # returns = np.zeros(self.envs.num_envs, dtype=np.float32)
    #     # for _ in range(self.horizon):
    #     while True:
    #         a1 = self.policy(obs[:, 0])
    #         a2 = self.adv_policy(obs[:, 1])
    #         # a2 = np.random.randint(0, self.envs.n_actions, self.envs.num_envs)  # Random opponent
    #         actions = np.stack([a1, a2], axis=1)
    #         obs, r, done, _, _ = self.envs.step(actions)

    #         # returns += r
    #         if done.any():
    #             self.envs.set_state(node.state, mask=done)
    #             total += r.sum()
    #             runs += done.sum()
    #             if runs >= self.rollouts:
    #                 break
        
    #     return total / runs

    @TimerRegistry.wrap_fn("MCTS.rollout_vec")  # Other envs are stil running, and they randomly be done same time as a first timem env
    def rollout_vec(self, node: Node):

        self.envs.set_state(node.state)
        obs = self.envs.encode(self.envs.state)

        runs = 0
        total = 0.0

        while runs < self.rollouts:
            a1 = self.policy(obs[:, 0])
            a2 = self.adv_policy(obs[:, 1])
            actions = np.stack([a1, a2], axis=1)

            obs, r, done, _, _ = self.envs.step(actions)

            if done.any():

                done_mask = done.astype(bool)

                total += r[done_mask].sum()
                runs += done_mask.sum()

                # restart finished envs
                self.envs.set_state(node.state, mask=done_mask)

                obs = self.envs.encode(self.envs.state)

        return total / runs
    
    @TimerRegistry.wrap_fn("MCTS.voronoi_value")
    def voronoi_value(self, node, max_steps):
        state = node.state
        val = voronoi(state[0], state[2], state[1])  # flip positions because agent is bike2
        # print(f"{val:.2f}")
        return val

    def backup(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value            
            node = node.parent
    