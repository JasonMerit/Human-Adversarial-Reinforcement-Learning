# Based on mcts.py + regret from Google Ai studio
 
import numpy as np
from rich import print

from rl_core.env import TronDuoEnv, GameState
from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv
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



class MCTS:
    """Returns only interested in terminal states, otherwise value must be cumulative discounted when backup"""

    def __init__(self, policy: callable, adv_policy: callable, env: TronDuoEnv, envs: VecTronDuoEnv, rollouts: int, horizon=200):
        self.policy = policy
        self.adv_policy = adv_policy
        self.env = env  # For structured search
        self.envs = envs  # For structured search
        self.rollouts = rollouts

    @TimerRegistry.wrap_fn("MCTS.simulate_q_values")
    def __call__(self, state, sims=400):
        root = Node(state)
        for _ in range(sims):
            self.simulate(root)

        # return root.get_strategy()  # Use argmax on this array
        return root.strategy_sum / np.sum(root.strategy_sum)  # Average strategy over simulations

    def simulate(self, root: Node):
        node = root
        path = []  # For backup

        # selection
        while node.is_expanded() and not node.terminal:
            p1_strat = node.get_strategy()
            a1 = np.random.choice(3, p=p1_strat)
            a2 = np.random.choice(3) # Can use adv_policy here if known
            
            path.append((node, (a1, a2)))
            node = node.children[a1, a2]

        # expansion
        if not node.is_expanded:# TODO: Need this part? and not node.terminal:  # non-terminal and not fully expanded
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
            p2_strat = np.ones(3) / 3.0 
            
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
    