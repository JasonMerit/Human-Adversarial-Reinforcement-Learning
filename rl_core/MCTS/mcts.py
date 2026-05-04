import numpy as np
# from rich import print

from rl_core.env import PoLEnv  
from rl_core.MCTS.vec_pol import VecPoLEnv
from rl_core.agents.utils import TimerRegistry
from rl_core.env.heuristic import voronoi

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
    
    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

class MCTS:
    """Returns only interested in terminal states, otherwise value must be cumulative discounted when backprop"""

    def __init__(self, env: PoLEnv, envs: VecPoLEnv, rollouts: int, max_steps=200):
        self.env = env  # For structured search
        self.envs = envs  # For structured search
        self.rollouts = rollouts
        self.max_steps = max_steps

    @TimerRegistry.wrap_fn("MCTS.plan")
    def plan(self, root, sims=400):
        # assert not root.terminal, "wtf m8"
        assert not root.terminal, "Planning from a terminal state is not meaningful"
        for _ in range(sims):
            # print("======", _, "=======")
            self.simulate(root)
            # print(_, self.best_action(root), end="\r")

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
        # value = node.reward if node.terminal else self.rollout(node, self.max_steps)
        # value = node.reward if node.terminal else self.rollout_vec(node, self.max_steps)
        value = node.reward if node.terminal else self.voronoi_value(node, self.max_steps)
        # if node.terminal:
            # print(f"Terminal ({value})")

        # backprop            
        self.backprop(node, value)

    def select(self, node, c=1.4) -> Node:
        """Select child with highest UCT value."""
        best = None
        best_score = -1e9


        for child in node.children:
            if child.N == 0:
                return child
            uct = child.W/child.N + c*np.sqrt(np.log(node.N)/child.N)
            # uct = child.Q + c * np.sqrt(np.log(node.N + 1) / (child.N + 1))
            if uct > best_score:
                best_score = uct
                best = child

        # print(f"Selecting {best.action} ({best_score:.2f})")
        return best

    def expand(self, node: Node):
        """Expand by taking an untried action."""
        action = node.untried.pop(np.random.randint(len(node.untried)))

        self.env.set_state(node.state)
        _, reward, done, _, _ = self.env.step(action)
        # print(f"Expanding {action} ({reward}) {done=}")

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

    # @TimerRegistry.wrap_fn("MCTS.rollout")
    # def rollout(self, node, max_steps):
    #     total = 0.0
    #     for _ in range(self.rollouts):
    #         self.env.set_state(node.state)
    #         # for _ in range(max_steps):
    #         while True:
    #             _, r, done, _, _ = self.env.step(self.env.sample_action())

    #             total += r
    #             if done:
    #                 break
        
    #     return total / self.rollouts
    
    @TimerRegistry.wrap_fn("MCTS.rollout_vec")
    def rollout_vec(self, node, max_steps):
        self.envs.set_state(node.state)

        runs = 0
        total = 0.0
        # returns = np.zeros(self.envs.num_envs, dtype=np.float32)
        # for _ in range(max_steps):
        while True:
            actions = self.envs.sample_actions()
            _, r, done, _, _ = self.envs.step(actions)

            # returns += r
            if done.any():
                self.envs.set_state(node.state, mask=done)
                # total += returns[done].sum()
                # assert (returns[~done] == 0).all(), "Non-done envs should have zero returns"
                # returns[done] = 0.0
                total += r.sum()
                runs += done.sum()
                if runs >= self.rollouts:
                    break
        
        return total / runs
    
    @TimerRegistry.wrap_fn("MCTS.voronoi_value")
    def voronoi_value(self, node, max_steps):
        state = node.state
        val = voronoi(state[0], state[2], state[1])  # flip positions because agent is bike2
        print(f"{val:.2f}")
        return val

    def backprop(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value            
            node = node.parent

    def best_action(self, root):
        """Return the action of the most visited child."""
        assert any(c is not None for c in root.children), "No children expanded"
        valid_children = [c for c in root.children if c is not None]
        best = max(valid_children, key=lambda c: c.N).action
        # for i, child in enumerate(root.children):
            # print(i, child.N, child.Q / child.N)
        # print(f"Best action from {[c.N for c in valid_children]} is {best}")
        return best

if __name__ == "__main__":
    from tqdm import trange
    SIZE=25
    NUM_ENVS = 64
    actual_env = PoLEnv(SIZE)
    sim_env = PoLEnv(SIZE)
    sim_envs = VecPoLEnv(NUM_ENVS, SIZE)

    wins = 0
    runs = 1
    # for _ in range(runs):
    history = [[] for _ in range(runs)]
    for i in range(runs):
        actual_env.reset()
        mcts = MCTS(sim_env, sim_envs)
        root = Node(actual_env.state, actual_env.n_actions)

        steps = 0
        while True:
            action = mcts.plan(root, sims=200)
            obs, reward, done, _, _ = actual_env.step(action)

            child = root.children[action]  # Reuse the subtree if it exists
            if child is None:
                root = Node(actual_env.state, actual_env.n_actions)
            else:
                child.parent = None
                root = child

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
        