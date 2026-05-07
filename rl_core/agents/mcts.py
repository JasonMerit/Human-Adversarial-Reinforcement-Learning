import numpy as np
import torch

from .rainbow import RainbowAgent
from rl_core.MCTS.mcts import MCTS, Node


class MCTSAgent(RainbowAgent):
    def __init__(self, env, envs, rollouts=100):
        super().__init__(env.obs_shape, env.n_actions)
        self.mcts = MCTS(env, envs, rollouts)


if __name__ == "__main__":
    from ..env import TronDuoEnv
    from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv
    SIZE=5
    NUM_ENVS=5
    agent = MCTSAgent(TronDuoEnv(SIZE), VecTronDuoEnv(NUM_ENVS,SIZE), rollouts=100)
    env = TronDuoEnv(SIZE)
    obs, info = env.reset()
    state = info["state"]

    # Populate buffer with random samples just to have something to train on
    for _ in range(200):
        action = env.action_space.sample()
        next_obs, reward, done, _, info = env.step(action)
        state = info["state"]
        agent.rb.add(state, action, reward, next_state, done)
        next_state = state

        if done:
            env.reset()
            print("kek")

