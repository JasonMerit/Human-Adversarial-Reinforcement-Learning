import time
import os
from pathlib import Path
import torch
import numpy as np

from .battle import get_agent_files, make_dqn, make_rainbow, make_clean_rainbow
from rl_core.env import PoLEnv

clear = lambda: os.system('cls')

def view(obs, action, reward):
    clear()
    print(f"Action: {action}, Reward: {reward}")
    size = len(obs[0])
    board = np.full((size, size), ".", dtype=str)
    board[obs[0] == 1] = "X"
    board[obs[2] == 1] = "G"
    board[obs[1] == 1] = "A"

    for row in board:
        print(" ".join(row))

    time.sleep(0.5)

def play(agent, env: PoLEnv, render=False):
    obs, info = env.reset()
    while True:
        action = agent.act(torch.as_tensor(obs).unsqueeze(0))
        obs, reward, done, _, info = env.step(action)
        if render:
            view(obs, action, reward)
        if done:
            return reward == 1.0

def solve(env: PoLEnv):
    # Seek max reward at each step
    obs, info = env.reset()
    while True:
        rewards = [env.peek_reward(a) for a in range(env.action_space.n)]
        action = rewards.index(max(rewards))
        obs, reward, done, _, info = env.step(action)
        
        view(obs, action, reward)
        if done:
            break

def eval(args):
    size = 5
    env = PoLEnv(size)

    n_actions = env.action_space.n  # Either is fine (symmetric environment)
    obs_shape = env.observation_space.shape[-3:]  # Ignore the stacked observations

    path = Path("runs") / args.folder / "A.pth"
    agent = make_dqn(path, obs_shape, n_actions)
    agent.eval()
    if play(agent, env):
        print("WIN!")
    else:
        print("FAIL!")

def evals(args):
    # Find all folders with the given prefix
    folders = [f for f in os.listdir("runs") if f.startswith("PoL")]
    # filter out folder lower than starting_indx
    folders = [f for f in folders if args.starting_index <= int(f.split("_")[-1]) <= args.ending_index]
    # Sort folders based on indx, e.g. PoL_5000 should be after PoL_4000
    folders.sort(key=lambda x: int(x.split("_")[-1]))
    
    size = 25
    env = PoLEnv(size)

    n_actions = env.action_space.n  # Either is fine (symmetric environment)
    obs_shape = env.observation_space.shape[-3:]  # Ignore the stacked observations

    for f in folders:
        print(f"{f}...", end=" ")
        path = Path("runs") / f / "A.pth"
        agent = make_dqn(path, obs_shape, n_actions)
        agent.eval()
        print("WIN" if play(agent, env, args.render) else "FAIL")

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    # parser.add_argument("folder", type=str, default="", help="Path folder of trained model checkpoints.")
    parser.add_argument("starting_index", nargs="?", type=int, default=0, help="Starting index for evaluation.")
    parser.add_argument("ending_index", nargs="?", type=float, default=float('inf'), help="Starting index for evaluation.")
    parser.add_argument("--render", action="store_true", help="Whether to render the environment during evaluation.")

    args = parser.parse_args()
    evals(args)
    # solve(PoLEnv(size=25))