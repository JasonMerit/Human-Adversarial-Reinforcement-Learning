import time
import os
from pathlib import Path
import torch
import numpy as np
import yaml
from rich import print

from .battle import get_agent_files, make_dqn, make_rainbow
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

    time.sleep(0.2)

def make_agent(path, obs_shape, n_actions, args):
    make_fn = make_dqn if "dqn" in args.folder.lower() else make_rainbow
    return make_fn(path, obs_shape, n_actions)

def play(agent, env: PoLEnv, render=False):
    obs, info = env.reset()
    steps = 0
    while True:
        action = agent.act(torch.as_tensor(obs).unsqueeze(0))
        obs, reward, done, _, info = env.step(action)
        steps += 1
        if render:
            view(obs, action, reward)
        if done:
            return reward == 1.0

def solve(env: PoLEnv):
    # Seek max reward at each step
    obs, info = env.reset()
    steps = 0
    while True:
        rewards = [env.peek_reward(a) for a in range(env.action_space.n)]
        action = rewards.index(max(rewards))
        obs, reward, done, _, info = env.step(action)
        steps += 1
        
        view(obs, action, reward)
        if done:
            print(steps)
            break

def eval(agent, env: PoLEnv, device):
    obs, info = env.reset()
    steps = 0
    while True:
        action = agent.act(torch.as_tensor(obs, device=device).unsqueeze(0))
        obs, reward, done, _, info = env.step(action)
        steps += 1
        
        if done:
            return steps if reward == 1.0 else env.size * env.size  # Max steps is size^2, so return that as a penalty for losing

def evals(args):
    # Find all folders with the given prefix
    folders = [f for f in os.listdir("runs") if f.startswith(args.folder + "_")]
    # filter out folder lower than starting_indx
    folders = [f for f in folders if args.starting_index <= int(f.split("_")[-1]) <= args.ending_index]
    # Sort folders based on indx, e.g. PoL_5000 should be after PoL_4000
    folders.sort(key=lambda x: int(x.split("_")[-1]))
    
    env = PoLEnv(args.size)
    n_actions = env.action_space.n  # Either is fine (symmetric environment)
    obs_shape = env.observation_space.shape[-3:]  # Ignore the stacked observations
    
    wins = 0
    total_steps = []
    for f in folders:
        path = Path("runs") / f / "results.yml"
        # Read steps_taken from file and add to toal_steps
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping...")
            continue
        with open(path) as stream:
            total_steps.append(yaml.safe_load(stream)["global_steps"])

        path = Path("runs") / f / "A.pth"
        agent = make_agent(path, obs_shape, n_actions, args)
        agent.eval()
        wins += play(agent, env, args.render)
    total_steps = np.array(total_steps)
    print(f"[bold yellow]{args.folder}[/bold yellow] total steps avg: {total_steps.mean():,.1f} std: {total_steps.std():,.1f}")
    # print(f"Win ratio: {wins}/{len(folders)}")

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("folder", type=str, default="", help="Path folder of trained model checkpoints.")
    parser.add_argument("size", type=int, help="Size of the PoLEnv to evaluate on.")
    parser.add_argument("starting_index", nargs="?", type=int, default=0, help="Starting index for evaluation.")
    parser.add_argument("ending_index", nargs="?", type=float, default=float('inf'), help="Starting index for evaluation.")
    parser.add_argument("--render", action="store_true", help="Whether to render the environment during evaluation.")

    args = parser.parse_args()
    evals(args)
    # solve(PoLEnv(size=5))