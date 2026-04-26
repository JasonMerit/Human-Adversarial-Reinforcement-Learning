import numpy as np
from rich import print

from rl_core.env import PoLEnv  
from rl_core.MCTS.vec_pol import VecPoLEnv
from rl_core.agents.utils import TimerRegistry

class Node:
    def __init__(self, state, n_actions, parent=None, action=None, terminal=False, reward=0):
        self.state = state
        self.parent = parent
        self.action = action # Debugging
        self.terminal = terminal
        self.reward = reward

        self.children = [None] * n_actions
        self.untried = list(range(n_actions))

        self.N = 0
        self.W = 0.0
        self.Q = 0.0

class MCTS:
    """Returns only interested in terminal states, otherwise value must be cumulative discounted when backprop"""

    def __init__(self, env: PoLEnv, envs: VecPoLEnv):
        self.env = env  # For structured search
        self.envs = envs  # For parallel rollouts

    @TimerRegistry.wrap_fn("MCTS.plan")
    def plan(self, root, sims=400):
        for _ in range(sims):
            self.simulate(root)

        return self.best_action(root)

    def simulate(self, root):
        node = root

        # selection
        while not node.untried and not node.terminal:  # fully expanded and non-terminal
            node = self.select(node)

        # expansion
        if node.untried and not node.terminal:  # non-terminal and not fully expanded
            node = self.expand(node)

        # evalution # TODO TOGGLE HERE FOR PROOF OF BETTER ACTION HISTORY
        # value = node.reward if node.terminal else self.rollout(node)
        value = node.reward if node.terminal else self.rollout_vec(node)

        # backprop            
        self.backprop(node, value)

    def select(self, node, c=1.4) -> Node:
        """Select child with highest UCT value."""
        best = None
        best_score = -1e9

        for child in node.children:
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
            self.env.state,
            n_actions=self.env.n_actions,
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
        B = self.envs.num_envs

        alive = np.ones(B, dtype=np.float32)
        total = np.zeros(B, dtype=np.float32)
        for _ in range(max_steps):
            actions = self.envs.sample_actions()
            _, r, done, _, _ = self.envs.step(actions)

            total += r * alive
            alive *= (1.0 - done)
            if not alive.any():
                break

        return total.mean()

    def backprop(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    def best_action(self, root):
        """Return the action of the most visited child."""
        return max(root.children, key=lambda x: x.N).action

if __name__ == "__main__":
    from tqdm import trange
    SIZE=6
    NUM_ENVS = 64
    actual_env = PoLEnv(SIZE)
    sim_env = PoLEnv(SIZE)
    sim_envs = VecPoLEnv(NUM_ENVS, SIZE)

    wins = 0
    runs = 100
    # for _ in range(runs):
    history = [[] for _ in range(runs)]
    for i in trange(runs):
        actual_env.reset()
        mcts = MCTS(sim_env, sim_envs)
        root = Node(actual_env.state, actual_env.n_actions)

        steps = 0
        while True:
            action = mcts.plan(root, sims=503)

            obs, reward, done, _, _ = actual_env.step(action)

            root = root.children[action]  # Move down the tree
            root.parent = None
            history[i].append(action)

            steps += 1
            if done:
                if reward > 0:
                    wins += 1
                break
    
    # print lengths of each history entry
    length = sum(len(h) for h in history) / len(history)
    print(f"Win rate: {wins}/{runs} = {wins/runs:.2f} with an avg length {length:.2f}")
    TimerRegistry.report()
        