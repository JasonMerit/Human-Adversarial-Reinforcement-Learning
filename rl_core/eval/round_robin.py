import os
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from rich import print

from rl_core.env import TronDuoEnv, TronView
from rl_core.agents.dqn import QNetwork
from .battle_team import battle_team, make_team


from rl_core.env.wrappers import TorchObservationWrapper

def battle(agent_0 : QNetwork, agent_1 : QNetwork, env):
    obs, _ = env.reset()
    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = agent_0.act(obs0), agent_1.act(obs1)  # Use the loaded model to select an action
        obs, _, done, _, info = env.step([a0, a1])

        if done:
            return info.get("result")
    
def round_robin(team_names):
    env = TronDuoEnv()
    # env = TronView(env)
    env = TorchObservationWrapper(env, device="cpu")

    n = len(team_names)
    win_matrix = np.full((n, n), np.nan)  

    total_team_battles = n * (n - 1) // 2  # 8 choose 2 = 28
    total_battles = 0

    for i in range(n):
        team_i = make_team(team_names[i], env)
        for j in range(i + 1, n):
            team_j = make_team(team_names[j], env)
            res = battle_team(team_i, team_j, env)

            win_matrix[i, j] = res["score"]
            win_matrix[j, i] = 1.0 - res["score"]

            total_battles += 1
            print(f"Completed {total_battles}/{total_team_battles} battles ({team_names[i]} vs {team_names[j]}: {res['score']:.2f})")

    quit()
    return win_matrix

if __name__ == "__main__":
    # Get args
    # import argparse
    # parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    # parser.add_argument("path", type=str, help="Path folder of trained model checkpoints.")
    # args = parser.parse_args()
    name = "NN"
    save_file = f"rl_core/eval/rr_results/{name}.npy"
    # if os.path.exists(save_file):
    #     raise ValueError(f"Save file '{save_file}' already exists. Please choose a different name or remove the existing file.")
    
    team_names = [f"{name}{i}" for i in range(8)]

    # Assert all folders and contain a trained Path.A.pth file
    for name in team_names:
        if not os.path.isdir(os.path.join("runs", name + "_0")):
            raise ValueError(f"Folder for team '{name}' not found in 'runs/' directory.")
        # Get all that start with "name" to check if each has a A.pth
        samples = [f for f in os.listdir('runs/') if os.path.isdir(Path('runs/') / f) and f.startswith(name)]

        for sample in samples:            
            if not os.path.exists(Path('runs') / sample / "A.pth"):
                raise ValueError(f"Checkpoint 'A.pth' not found for team '{name}' in folder '{sample}'. Not done training :(")
    # print(total)
    # quit()
    win_matrix = round_robin(team_names)
    np.save(save_file, win_matrix)
    