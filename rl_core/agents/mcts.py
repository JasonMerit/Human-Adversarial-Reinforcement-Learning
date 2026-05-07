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

