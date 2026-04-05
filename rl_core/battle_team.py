import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from tqdm.contrib import itertools

from rl_core.env import TronDuoEnv, TronView
from rl_core.env.env import TronCoreEnv, encode_observation, encode_observation_2channel
from rl_core.env.wrappers import TorchObservationWrapper
from rl_core.agents.dqn import QNetwork


def load_experiment(path, env):
    # Select the appropriate agent class based on the file name
    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    folder = "runs/"

    # Find all folders in runs that start with the given path
    runs = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f.startswith(os.path.basename(path))]

    team = []
    for run in runs:
        files = [f for f in os.listdir(os.path.join(folder, run)) if f.endswith(".pth")]
        file = max(files, key=lambda f: os.path.getmtime(os.path.join(folder, run, f)))
        network = QNetwork.from_checkpoint(os.path.join(folder, run, file), obs_shape, n_actions)
        network.eval()
        team.append(network)
    
    return team

def load_benchmark(path, env):
    # Find all folders in runs that start with the given path
    folder = "runs/"
    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]  

    runs = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f.startswith(os.path.basename(path))]

    team = []
    for run in runs:
        human_file = os.path.join(folder, run, "human.pth")
        adv_file = os.path.join(folder, run, "adversary.pth")

        network = QNetwork.from_checkpoint(human_file, obs_shape, n_actions)
        network.eval()
        team.append(network)

        network = QNetwork.from_checkpoint(adv_file, obs_shape, n_actions)
        network.eval()
        team.append(network)

    return team

def play_manual_encode(agent_0, agent_1, env, encode_0=encode_observation, encode_1=encode_observation):
    (heading0, heading1, walls, bike1_pos, bike2_pos), _ = env.reset()
    while True:
        obs0, obs1 = encode_0(walls, bike1_pos, bike2_pos), encode_1(walls, bike2_pos, bike1_pos)
        obs0, obs1 = np.rot90(obs0, k=heading0, axes=(1, 2)).copy(), np.rot90(obs1, k=heading1, axes=(1, 2)).copy()
        obs0, obs1 = torch.as_tensor(obs0, dtype=torch.float32).unsqueeze(0), torch.as_tensor(obs1, dtype=torch.float32).unsqueeze(0)
        
        a0, a1 = agent_0.act(obs0), agent_1.act(obs1)

        (heading0, heading1, walls, bike1_pos, bike2_pos), _, done, _, info = env.step([a0, a1])
        if done:
            return info.get("result")

def play(agent_0, agent_1, env):
    obs, _ = env.reset()
    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = agent_0.act(obs0), agent_1.act(obs1)  # Use the loaded model to select an action
        obs, _, done, _, info = env.step([a0, a1])

        if done:
            return info.get("result")
    
def battle_team(path0, path1):
    # env = TronCoreEnv()
    env = TronDuoEnv()
    env = TorchObservationWrapper(env, device="cpu")
    # env = TronView(env, fps=100)
    # env.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 3, 25, 25), dtype=np.float32)
    team_1 = load_experiment(path0, env)

    # env.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 2, 25, 25), dtype=np.float32)
    team_2 = load_benchmark(path1, env)
    print(f"Team sizes: {len(team_1)}, {len(team_2)}")

    # Have each team_1 member play against each team_2 member and print results
    results = [0, 0, 0]
    for agent_0, agent_1 in itertools.product(team_1, team_2, desc="Battling Teams", leave=False):
        result = play(agent_0, agent_1, env)
        results[result] += 1
    
    print(f"Results: {results}")
    print(f"Team 1 win rate: {results[1] / (results[1] + results[2]) * 100:.1f}% with {results[0]} draws")

if __name__ == "__main__":
    battle_team("Pooling", "BenchMark")
    # battle_team("runs/BenchMark", "runs/TwoChannel")
    
