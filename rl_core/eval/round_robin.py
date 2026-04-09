import os
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl_core.env import TronDuoEnv, TronView
from rl_core.agents.dqn import QNetwork


class TorchObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def observation(self, obs):
        return torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

def load_agents(agent_folder, env):
    # Select the appropriate agent class based on the file name
    # find all files ending in .pth in the given path
    files = [f for f in os.listdir(agent_folder) if f.endswith(".pth") and f[-5] == "0"]
    # Sort agents based on the number in the file name, e.g. human_1000.pth -> 1000
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))


    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    return [QNetwork.from_checkpoint(os.path.join(agent_folder, f), obs_shape, n_actions) for f in files], files

def battle(agent_0 : QNetwork, agent_1 : QNetwork, env):
    obs, _ = env.reset()
    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = agent_0.act(obs0), agent_1.act(obs1)  # Use the loaded model to select an action
        obs, _, done, _, info = env.step([a0, a1])

        if done:
            return info.get("result")
    
def round_robin(path):
    env = TronDuoEnv()
    env = TorchObservationWrapper(env, device="cpu")

    agents, names = load_agents(path, env)
    for agent in agents:
        agent.eval()

    n = len(agents)
    # Matrix: rows = agent i, columns = agent j
    win_matrix = np.zeros((n, n), dtype=float)

    for i in tqdm(range(n), desc="Round Robin", leave=False):
        agent_i = agents[i]
        for j in range(n):
            if i == j:
                win_matrix[i, j] = np.nan  # no self-play
                continue
            result = battle(agent_i, agents[j], env)  # 0=draw,1=win,2=loss
            if result == 0:  # draw
                win_matrix[i, j] = 0.5
            elif result == 1:  # agent_i wins
                win_matrix[i, j] = 1
            else:  # agent_i loses
                win_matrix[i, j] = 0

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(win_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, label="Effective win rate")
    plt.xticks(range(n), names, rotation=90)
    plt.yticks(range(n), names)
    plt.title("Round Robin Winnings Heatmap (row wins vs column = 1.0)")
    plt.tight_layout()
    plt.show()

    return win_matrix

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("path", type=str, help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    win_matrix = round_robin(args.path)
    np.save("rl_core/round_robin_results.npy", win_matrix)
    