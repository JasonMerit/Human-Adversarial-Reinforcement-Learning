import numpy as np

from rl_core.env import PoLEnv  
from rl_core.MCTS.vec_pol import VecPoLEnv
from rl_core.agents.utils import TimerRegistry

class Node:
    def __init__(self, state, parent=None, action=None, terminal=False, reward=0):
        self.state = state
        self.parent = parent
        self.action = action # Debugging
        self.terminal = terminal
        self.reward = reward

        self.children = {}  # action -> Node
        self.untried = [0,1,2,3]
        # self.untried = list(range(env.n_actions))

        self.N = 0
        self.W = 0.0
        self.Q = 0.0

class MCTS:
    """Returns only interested in terminal states, otherwise value must be cumulative discounted when backprop"""

    def __init__(self, env: PoLEnv, envs: VecPoLEnv):
        self.env = env
        self.envs = envs

    @TimerRegistry.wrap_fn("MCTS.plan")
    def plan(self, root, sims=400):
        for _ in range(sims):
            self.simulate(root)

        return self.best_action(root)

    def simulate(self, root):
        node = root

        # selection
        while not node.untried and node.children:  # fully expanded and non-terminal
            node = self.select(node)

        # expansion
        if node.untried and not node.terminal:  # non-terminal and not fully expanded
            node = self.expand(node)

        # evalution
        # value = node.reward if node.terminal else self.rollout(node)
        value = node.reward if node.terminal else self.rollout_vec(node)

        # backprop            
        self.backprop(node, value)

    def select(self, node, c=1.4) -> Node:
        """Select child with highest UCT value."""
        best = None
        best_score = -1e9

        for child in node.children.values():
            uct = child.Q + c * np.sqrt(np.log(node.N + 1) / (child.N + 1))
            if uct > best_score:
                best_score = uct
                best = child

        return best


    def expand(self, node: Node):
        """Expand by taking an untried action."""
        action = node.untried.pop(np.random.randint(len(node.untried)))

        self.env.set_state(node.state)
        _, reward, done, _, _ = self.env.step(action)

        child = Node(
            self.env.state if not done else None,
            parent=node,
            action=action,
            terminal=done,
            reward=reward,
        )

        node.children[action] = child
        return child

    @TimerRegistry.wrap_fn("MCTS.rollout")
    def rollout(self, node, max_steps=200):
        self.env.set_state(node.state)

        total = 0.0
        for _ in range(max_steps):
            a = np.random.randint(self.env.n_actions)
            _, r, done, _, _ = self.env.step(a)

            total += r
            if done:
                break

        return total
    
    @TimerRegistry.wrap_fn("MCTS.rollout_vec")
    def rollout_vec(self, node, max_steps=200):
        self.envs.set_state(node.state)

        total = 0.0
        for _ in range(max_steps):
            a = np.random.randint(self.envs.n_actions)
            _, r, done, _, _ = self.envs.step(a)

            total += r
            if done:
                break

        return total

    def backprop(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    def best_action(self, root):
        """Return the action of the most visited child."""
        return max(root.children.items(), key=lambda x: x[1].N)[0]
        # visits = [(child.N, action) for action, child in root.children.items()]
        # visits.sort(reverse=True)

        # return visits[0][1]

if __name__ == "__main__":
    from tqdm import trange
    SIZE=6
    actual_env = PoLEnv(SIZE)
    sim_env = PoLEnv(SIZE)
    sim_envs = VecPoLEnv(64, SIZE)

    wins = 0
    tries = 10
    for _ in trange(tries):
        actual_env.reset()
        mcts = MCTS(sim_env, sim_envs)
        root = Node(actual_env.state)

        while True:
            action = mcts.plan(root, sims=500)

            obs, reward, done, _, _ = actual_env.step(action)
            root = root.children[action]  # Move down the tree
            root.parent = None

            if done:
                if reward > 0:
                    wins += 1
                break
    
    print(f"Win rate: {wins}/{tries} = {wins/tries:.2f}")
    TimerRegistry.report()
        