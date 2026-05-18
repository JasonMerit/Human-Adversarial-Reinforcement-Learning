import numpy as np
from rl_core.env import TronDuoEnv, GameState
from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv
from rich import print


class StateNode:
    def __init__(self, state, adv_probs: np.ndarray, parent=None, action=None):
        if parent:
            assert isinstance(parent, ChanceNode), f"Expected parent to be a ChanceNode, got {type(parent)}"
        assert isinstance(state, GameState), f"Expected state to be a GameState, got {type(state)}"
        self.state = state
        self.adv_probs = adv_probs
        self.parent = parent
        self.action = action  # action from parent chance node taken by opponent to reach this state, for debugging

        self.children = [None] * 3  # agent actions → ChanceNodes

        self.N = 0
        self.W = 0.0

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def is_fully_expanded(self):
        return all(c is not None for c in self.children)


class ChanceNode:
    """
    Chance node uncertain state outcome from agent_action.
    Does opponent action enumeration
    """

    def __init__(self, parent, agent_action):
        assert isinstance(parent, StateNode), f"Expected parent to be a StateNode, got {type(parent)}"
        self.parent = parent
        self.agent_action = agent_action
        self.children = [None] * 3  # opponent actions → StateNodes
        self.N = 0
        self.W = 0.0

        self.expecti = 0.0  # Expected value of this chance node, updated after expansion

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def is_fully_expanded(self):
        return all(c is not None for c in self.children)

class Leaf:
    def __init__(self, value, parent):
        self.value = value
        self.parent = parent


class MCTS:
    def __init__(self, player, policy, adv_probs, env: TronDuoEnv, envs: VecTronDuoEnv, sims=200, c=1.4):
        self.player = player
        self.policy = policy
        self.adv_probs = adv_probs  # p(a|obs) vector np.array of shape [B, 3]
        self.env = env
        self.envs = envs

        self.sims = sims
        self.c = c
    
    def _traver_tree(self, node):  # Debugging purposes
        def _traverse(node, depth=0):
            indent = "    " * depth
            if isinstance(node, StateNode):
                print(f"{indent}StateNode (N={node.N}, W={node.W:.2f}, Q={node.Q:.2f})")
                for child in node.children:
                    if child is not None:
                        _traverse(child, depth + 1)
            elif isinstance(node, ChanceNode):
                print(f"{indent}ChanceNode (a={node.agent_action}, N={node.N}, W={node.W:.2f}, Q={node.Q:.2f})")
                for child in node.children:
                    if child is not None:
                        _traverse(child, depth + 1)
            elif isinstance(node, Leaf):
                print(f"{indent}Leaf (value={node.value:.2f})")
        _traverse(node)
        print("==" * 20)
        input("Press Enter to continue...\n>")
    
    def get_opp_probs(self, state) -> np.ndarray:
        assert isinstance(state, GameState), f"Expected GameState, got {type(state)}"
        # obs = TronDuoEnv.encode(state)[1 - self.player]
        # probs = self.adv_policy(obs)
        probs = np.array([1/3, 1/3, 1/3])  # Uniform random for now
        assert isinstance(probs, np.ndarray), f"Expected np.ndarray, got {type(probs)}"
        assert probs.shape == (3,), f"Expected shape (3,), got {probs.shape}"
        return probs
    
    def adv_act(self, obs: np.ndarray):
        assert obs.ndim == 4, f"Expected input shape (B, C, H, W), got {obs.shape}"
        assert isinstance(obs, np.ndarray), f"Expected input to be a np.ndarray, got {type(obs)}"
        return np.random.choice(3, size=obs.shape[0])

    def __call__(self, state: GameState):
        root = StateNode(state, self.get_opp_probs(state))

        for _ in range(self.sims):
            self.simulate(root)
            self._traver_tree(root)

        return np.array([c.Q for c in root.children])

    def simulate(self, state_node: StateNode):
        while state_node.is_fully_expanded():
            state_node = self.select_state(state_node)

        chance_node = self.expand_state(state_node)  
        value = self.evaluate(chance_node)

        self.backup(chance_node, value)

    def select_state(self, node: StateNode):
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

    def expand_state(self, node: StateNode):
        for a in range(3):
            if node.children[a] is None:
                node.children[a] = ChanceNode(node, a)
                return node.children[a]

        raise Exception("StateNode fully expanded")

    def expand_chance(self, chance_node: ChanceNode):
        values = np.zeros(3)

        for b in range(3):
            self.env.set_state(chance_node.parent.state)

            joint_action = [chance_node.agent_action, b]  # TODO: reorder by player index
            _, r, done, _, info = self.env.step(joint_action)
            if done:
                values[b] = r
                chance_node.children[b] = Leaf(r, parent=chance_node)
            else:
                next_state = info['state']
                values[b] = self.evaluate(next_state)  # 
                chance_node.children[b] = StateNode(next_state, self.get_opp_probs(next_state), parent=chance_node, action=b)

        return np.dot(chance_node.parent.adv_probs, values)

    def evaluate(self, state):
        self.envs.set_state(state)
        obs = self.envs.get_obs()

        total = np.zeros(self.envs.num_envs)

        done = np.zeros(self.envs.num_envs, dtype=bool)

        while not done.all():
            a = self.policy(obs[:, self.player])
            b = self.adv_act(obs[:, 1 - self.player])

            actions = np.stack([a, b], axis=1) # TODO: Reorder by player index

            obs, r, d, _, _ = self.envs.step(actions)

            total[~done] += r[~done]
            done |= d

        return total.mean()

    def backup(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent
