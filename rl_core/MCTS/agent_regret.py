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
        self.regret_sum_p1 = np.zeros(3) 
        self.strategy_sum_p1 = np.zeros(3)
        
        # Regret storage for Player 2 (Opponent)
        self.regret_sum_p2 = np.zeros(3) 
        self.strategy_sum_p2 = np.zeros(3)

        # Track Q-values for the 3x3 joint action matrix
        self.Q_matrix = np.zeros((3, 3)) 
        self.N = 0

    def get_strategy_p1(self, gamma):
        regrets = np.maximum(self.regret_sum_p1, 0)
        norm = np.sum(regrets)
        P = regrets / norm if norm > 0 else np.ones(3) / 3.0
        return (1 - gamma) * P + (gamma / 3.0)


    def get_strategy_p2(self, gamma):
        # Opponent wants to MINIMIZE Q (Zero-sum: My win is their loss)
        # So we use negative Q for regret matching
        regrets = np.maximum(-self.regret_sum_p2, 0) # Simplified logic
        norm = np.sum(regrets)
        P = regrets / norm if norm > 0 else np.ones(3) / 3.0
        return (1 - gamma) * P + (gamma / 3.0)

    def update_strategy(self):
        # Accumulate the current strategy to compute the average later
        self.strategy_sum_p1 += self.get_strategy_p1(0.0)  # Gamma = 0 because this is the final exploiting action suggestion
        self.strategy_sum_p2 += self.get_strategy_p2(0.0)

    def is_expanded(self):
        return not np.any(self.children == None)

    def __repr__(self):
        return f"Node(action={self.action}, terminal={self.terminal}, reward={self.reward}, N={self.N})"



class MCTS:
    """Returns only interested in terminal states, otherwise value must be cumulative discounted when backup"""

    def __init__(self, env: TronDuoEnv, envs: VecTronDuoEnv, rollouts: int, gamma=.3):
        # Gamma is the exploration constant to enable searching on dead nodes
        # Assert both produce a 3x1 array of probabilities
        # dummy_state = env.state  # envs here?
        # assert isinstance(prior_policy(dummy_state), np.ndarray) and prior_policy(dummy_state).shape == (3,), "Prior policy must return a 3-element numpy array"
        # assert isinstance(opp_policy(dummy_state), np.ndarray) and opp_policy(dummy_state).shape == (3,), "Opponent policy must return a 3-element numpy array"
        
        self.env = env  # For structured search
        self.envs = envs  # For structured search
        self.rollouts = rollouts
        self.gamma = gamma

    def reset(self):
        self.env.reset()
        self.envs.reset()
    
    @TimerRegistry.wrap_fn("MCTS.simulate_q_values")
    def plan(self, root, sims=400):
        for _ in range(sims):
            self.simulate(root)

        root.update_strategy()  # Update the average strategy after simulations
        strategy = root.strategy_sum_p1 / np.sum(root.strategy_sum_p1)
        return np.random.choice(3, p=strategy)

    def simulate(self, node: Node):
        path = []  # For backup

        # selection
        while node.is_expanded() and not node.terminal:
            a1 = np.random.choice(3, p=node.get_strategy_p1(self.gamma))
            a2 = np.random.choice(3, p=node.get_strategy_p2(self.gamma))
            
            path.append((node, (a1, a2)))
            node = node.children[a1, a2]

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
        

    @TimerRegistry.wrap_fn("MCTS.rollout_vec") 
    def rollout_vec(self, node: Node):
        actions = np.empty((self.rollouts, 2), dtype=np.int8)

        self.envs.set_state(node.state)

        total = 0.0
        active = np.ones(self.envs.num_envs, dtype=bool)
        assert self.envs.num_envs == self.rollouts, f"Rollouts must match number of parallel environments. Got {self.envs.num_envs} num envs and {self.rollouts} rollouts."

        while active.any():
            # a = self.act(node)
            a1 = np.random.choice(3, size=self.rollouts, p=node.get_strategy_p1(self.gamma))  # Player 1's actions based on its strategy
            a2 = np.random.choice(3, size=self.rollouts, p=node.get_strategy_p2(self.gamma))  # Player 2's actions based on its strategy
            actions[:, 0], actions[:, 1] = a1, a2

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
                
                # B. Regret Updates
                # --- P1 Regret (Maximize Q) ---
                p2_strat = node.get_strategy_p2(self.gamma)
                ev_p1 = np.dot(node.Q_matrix, p2_strat) # EV for each P1 action
                current_ev_p1 = np.dot(node.get_strategy_p1(self.gamma), ev_p1)
                node.regret_sum_p1 += (ev_p1 - current_ev_p1)
                
                # --- P2 Regret (Minimize Q) ---
                # P2 wants to minimize Q, so regret is (Current - Action_EV)
                p1_strat = node.get_strategy_p1(self.gamma)
                ev_p2 = np.dot(p1_strat, node.Q_matrix) # EV for each P2 action
                current_ev_p2 = np.dot(p2_strat, ev_p2)
                node.regret_sum_p2 += (current_ev_p2 - ev_p2)
                
                # C. Update the average strategy (to converge to Nash Equilibrium)
                node.update_strategy()
    