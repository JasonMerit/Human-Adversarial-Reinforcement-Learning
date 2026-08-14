# Based on mcts.py + regret from Google Ai studio
 
import numpy as np
from rich import print

from rl_core.env import TronDuoEnv, GameState
from rl_core.MCTS.vec_duo_env import VecTronDuoEnv
from rl_core.utils import TimerRegistry
from rl_core.env.heuristic import voronoi

class Node:
    def __init__(self, state, parent=None, action=None, terminal=False, reward=0):
        self.state = state
        self.parent = parent
        self.action = action # (a1, a2)
        self.terminal = terminal
        self.reward = reward

        # 3x3 matrix for joint action children
        self.children = np.full((3, 3), None, dtype=object)
        # assert shape is 3x3
        assert self.children.shape == (3, 3), f"Children shape is {self.children.shape}, expected (3, 3)"
        
        # Regret storage for Player 1 (Agent)
        self.regret_sum = np.zeros(3) 
        self.strategy_sum = np.zeros(3)
        self.N = 0
        
        # Track Q-values for the 3x3 joint action matrix
        self.Q_matrix = np.zeros((3, 3)) 

    def get_strategy(self):
        """Regret Matching formula: S(a) = max(0, R(a)) / sum(max(0, R(a)))"""
        regrets = np.maximum(self.regret_sum, 0)
        norm = np.sum(regrets)
        return regrets / norm if norm > 0 else np.ones(3) / 3.0

    def update_strategy(self):
        # Accumulate the current strategy to compute the average later
        self.strategy_sum += self.get_strategy()

    def is_expanded(self):
        return not np.any(self.children == None)

    def __repr__(self):
        return f"Node(action={self.action}, terminal={self.terminal}, reward={self.reward}, N={self.N})"



class MCTS:
    """Returns only interested in terminal states, otherwise value must be cumulative discounted when backup"""

    def __init__(self, prior_policy: callable, opp_policy: callable, env: TronDuoEnv, envs: VecTronDuoEnv, rollouts: int, horizon=200):
        self.prior_policy = prior_policy
        self.opp_policy = opp_policy
        # asser both policy and opp_policy are methods
        assert callable(prior_policy), "Prior policy must be a callable function"
        assert callable(opp_policy), "Opponent policy must be a callable function"
        # Assert both produce a 3x1 array of probabilities
        dummy_state = env.state  # envs here?
        assert isinstance(prior_policy(dummy_state), np.ndarray) and prior_policy(dummy_state).shape == (3,), "Prior policy must return a 3-element numpy array"
        assert isinstance(opp_policy(dummy_state), np.ndarray) and opp_policy(dummy_state).shape == (3,), "Opponent policy must return a 3-element numpy array"
        
        self.env = env  # For structured search
        self.envs = envs  # For structured search
        self.rollouts = rollouts

    def act(self, node: Node):
        probs = node.get_strategy() 
        return np.random.choice(3, p=probs)
    
    @TimerRegistry.wrap_fn("MCTS.simulate_q_values")
    def plan(self, root, sims=400):
        for _ in range(sims):
            self.simulate(root)

        root.update_strategy()  # Update the average strategy after simulations
        strategy = root.strategy_sum / np.sum(root.strategy_sum)
        return np.random.choice(3, p=strategy)

    def simulate(self, node: Node):
        path = []  # For backup

        # selection
        while node.is_expanded() and not node.terminal:
            p1_strat = node.get_strategy()
            a = np.random.choice(3, p=p1_strat)
            # print(f"Selected action {a} for Player 1 based on strategy {p1_strat}")
            b = np.random.choice(3, p=self.opp_policy(node.state))  # Opponent's action based on its policy
            
            path.append((node, (a, b)))
            node = node.children[a, b]

        # expansion
        if not node.is_expanded() and not node.terminal:  # non-terminal and not fully expanded
        # if not node.is_expanded():# TODO: Need this part? and not node.terminal:  # non-terminal and not fully expanded
            self.expand(node)

        # evaluation
        value = node.reward if node.terminal else self.rollout_vec(node)

        # backup            
        self.backup(path, node, value)

    def expand(self, node):
        for a1 in range(3):
            for a2 in range(3):
                # Simulate joint action (a1, a2)
                self.env.set_state(node.state)
                _, reward, done, _, _ = self.env.step((a1, a2))
                
                # Create 3x3 children matrix
                child = Node(
                    self.env.state,
                    parent=node,
                    action=(a1, a2),
                    terminal=done,
                    reward=reward,
                )
                node.children[a1, a2] = child
        

    @TimerRegistry.wrap_fn("MCTS.rollout_vec")  # Other envs are stil running, and they randomly be done same time as a first timem env
    def rollout_vec(self, node: Node):
        actions = np.empty((self.rollouts, 2), dtype=np.int8)

        self.envs.set_state(node.state)

        total = 0.0
        active = np.ones(self.envs.num_envs, dtype=bool)
        assert self.envs.num_envs == self.rollouts, f"Rollouts must match number of parallel environments. Got {self.envs.num_envs} num envs and {self.rollouts} rollouts."

        while active.any():
            # a = self.act(node)
            a = np.random.choice(3, size=self.rollouts, p=node.get_strategy())  # Player 1's actions based on its strategy
            b = np.random.choice(3, size=self.rollouts, p=self.opp_policy(node.state))  # Opponent's action based on its policy  TODO: Iterate over each state
            actions[:, 0], actions[:, 1] = a, b

            _, r, d, _, _ = self.envs.step(actions)

            # Add to total those that just became terminal this iteration
            total += r[active & d].sum()
            active &= ~d

        return total / self.rollouts    

    def backup(self, path, leaf_node, value):
            # 1. Update the leaf node
            leaf_node.N += 1
            
            # 2. Propagate back through the path taken
            for node, (a1, a2) in reversed(path):
                node.N += 1
                
                # A. Update the specific cell in the Q-matrix (Running Average)
                node.Q_matrix[a1, a2] += (value - node.Q_matrix[a1, a2]) / node.N
                
                # B. Regret Update for Player 1
                # We assume the opponent plays uniformly for now (p2_strat)
                p2_strat = self.opp_policy(node.state)  # Opponent's strategy based on its policy
                
                # Calculate utility for each of our possible actions a (0, 1, 2)
                # based on the Q_matrix and the opponent's strategy
                ev_each_action = np.zeros(3)
                for a in range(3):
                    ev_each_action[a] = np.sum(node.Q_matrix[a, :] * p2_strat)
                
                # The value of our current strategy (what we actually got on average)
                current_strategy = node.get_strategy()
                expected_val = np.sum(ev_each_action * current_strategy)
                
                # Update regret: (Value of action a - Current strategy value)
                for a in range(3):
                    node.regret_sum[a] += (ev_each_action[a] - expected_val)
                
                # C. Update the average strategy (to converge to Nash Equilibrium)
                node.update_strategy()
    