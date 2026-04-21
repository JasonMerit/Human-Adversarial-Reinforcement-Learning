import numpy as np
import torch

from .rainbow import RainbowAgent
from rl_core.env import PoLEnv

class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action

        self.children = {}
        self.visits = 0
        self.value = 0.0

class MCTSAgent(RainbowAgent):

    def __init__(self, obs_shape, n_actions, args, device, writer, name):
        super().__init__(obs_shape, n_actions, args, device, writer, name)

        self.rollouts = args.mcts_rollouts
        self.c_puct = args.mcts_c
    
    def mcts(self, env):
        root = MCTSNode(env.clone())
        for _ in range(self.rollouts):
            self._simulate(root)

        best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]

        return best_action

    def select_action(self, obs, env):
        action = self.mcts(env)

        return np.array([action])
    
    def _ucb(self, parent, child):
        if child.visits == 0:
            return float("inf")

        exploit = child.value / child.visits
        explore = self.c_puct * np.sqrt(np.log(parent.visits) / child.visits)

        return exploit + explore

    def _select(self, node):
        while node.children:

            node = max(
                node.children.values(),
                key=lambda c: self._ucb(node, c)
            )

        return node
    
    def _expand(self, node):
        if node.children:
            return node

        for a in range(self.n_actions):

            env = node.env.clone()
            obs, reward, done, _, _ = env.step(a)

            child = MCTSNode(env, parent=node, action=a)

            child.reward = reward
            child.done = done

            node.children[a] = child

        return np.random.choice(list(node.children.values()))
    
    def _rollout(self, env):
        total_reward = 0
        while True:
            action = np.random.randint(self.n_actions)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward
    
    # def _rollout(self, env):
    #     obs = env._get_state()
    #     with torch.no_grad():
    #         q = self.q_network(torch.tensor(obs, device=self.device).unsqueeze(0))
    #     return q.max().item()
    
    def _backup(self, node, value):
        while node is not None:

            node.visits += 1
            node.value += value

            node = node.parent
    
    def _simulate(self, root):
        node = self._select(root)

        if node.visits > 0 and not node.done:
            node = self._expand(node)

        if node.done:
            value = node.reward
        else:
            value = node.reward + self._rollout(node.env.clone())

        self._backup(node, value)