import os, sys
import numpy as np
from rich import print

from rl_core.env import TronView, TronEnv
from rl_core.agents.utils import TimerRegistry

from .mcts import MCTS, Node
from .vec_env import VecTronEnv

import gymnasium as gym
from rl_core.env import Tron, Result

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == "__main__":
    # os.system('cls')
    # TimerRegistry.disable()  # Disable timers for this test
    from tqdm import trange
    SIZE=9
    NUM_ENVS = 64

    actual_env = TronEnv(SIZE)
    actual_env = TronView(TronEnv(SIZE), fps=10)
    sim_env = TronEnv(SIZE)
    actual_env.reset()
    sim_env.reset()
    sim_envs = VecTronEnv(NUM_ENVS, SIZE)

    np.random.seed(3)

    wins = 0
    runs = 10
    # for _ in range(runs):
    history = [[] for _ in range(runs)]
    for i in trange(runs):
        actual_env.reset()
        sim_env.reset()
        mcts = MCTS(sim_env, sim_envs, 100)
        root = Node(actual_env.state, actual_env.n_actions)

        steps = 0
        while True:
            # with HiddenPrints():
            action = mcts.plan(root, sims=200)
            # action = A[i][steps]

            obs, reward, done, _, _ = actual_env.step(action)

            # if done:
            #     print("[bold red] ENV IS DONE")


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

    import winsound
    winsound.Beep(800, 200)  # frequency (Hz), duration (ms)
    winsound.Beep(700, 100)  # frequency (Hz), duration (ms)
    winsound.Beep(800, 400)  # frequency (Hz), duration (ms)

