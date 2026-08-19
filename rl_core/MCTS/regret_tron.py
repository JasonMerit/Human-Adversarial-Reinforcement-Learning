import os, sys
import numpy as np
from rich import print

from rl_core.env import TronView, TronDuoEnv
from rl_core.utils import TimerRegistry

from .agent_regret import MCTS, Node
from .vec_duo_env import VecTronDuoEnv


if __name__ == "__main__":
    from tqdm import trange
    SIZE=7
    NUM_ENVS = 64

    # actual_env = TronDuoEnv(SIZE)
    actual_env = TronView(TronDuoEnv(SIZE), fps=10)
    sim_env = TronDuoEnv(SIZE)
    actual_env.reset()
    sim_env.reset()
    sim_envs = VecTronDuoEnv(NUM_ENVS, SIZE)

    # prior_policy = lambda state: np.ones(3) / 3.0  # Uniform
    opp_policy = lambda state: np.ones(3) / 3.0  # Uniform
    # opp_policy = lambda state: np.array([0.0, 1.0, 0.0]) # Always 1
    np.random.seed(3)

    wins = 0
    runs = 10
    # for _ in range(runs):
    history = [[] for _ in range(runs)]
    for i in trange(runs):
        actual_env.reset()
        sim_env.reset()
        mcts = MCTS(sim_env, sim_envs, rollouts=NUM_ENVS)
        root = Node(actual_env.state)

        steps = 0
        while True:
            action = mcts.plan(root, sims=200)
            # print(root.children)
            opp_action = np.random.choice(3, p=opp_policy(actual_env.state))
            joint_action = np.array([action, opp_action])

            obs, reward, done, _, _ = actual_env.step(joint_action)

            # joint_action is a numpy array [action, opp_action]; index children with separate indices
            child = root.children[action, opp_action]  # Reuse the subtree if it exists
            print(child)
            if not isinstance(child, Node):
                print(f"Child is not of [cyan]Node[/cyan]. Got {type(child)} instead")
                quit()
            if child is None:
                root = Node(actual_env.state)
            else:
                child.parent = None
                root = child

            history[i].append(joint_action)

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

