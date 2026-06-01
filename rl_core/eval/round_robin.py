import os
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    env = TronDuoEnv(15)
    env = TorchObservationWrapper(env, device="cpu")

    n = len(team_names)
    # Matrix: rows = agent i, columns = agent j
    win_matrix = np.full((n, n), np.nan)

    for i in range(n):
        team_i = make_team(team_names[i], env)
        for j in range(i + 1, n):
            team_j = make_team(team_names[j], env)
            res = battle_team(team_i, team_j, env)

            win_matrix[i, j] = res["score"]
            win_matrix[j, i] = 1.0 - res["score"]

    return win_matrix

if __name__ == "__main__":
    # Get args
    # import argparse
    # parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    # parser.add_argument("path", type=str, help="Path folder of trained model checkpoints.")
    # args = parser.parse_args()
    team_names = [
        "Bench15",
        "Mirroring"
    ]
    win_matrix = round_robin(team_names)
    np.save("rl_core/round_robin_results.npy", win_matrix)
    