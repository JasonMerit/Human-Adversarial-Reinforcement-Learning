import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm

from rl_core.env import TronView
from rl_core.env.env import TronCoreEnv, encode_observation, encode_observation_2channel
from rl_core.agents.dqn import QNetwork


def load_team(path, env):
    # Select the appropriate agent class based on the file name
    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    folder = "runs/"
    
    # Find all folders in runs that start with the given path
    runs = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f.startswith(os.path.basename(path))]
    
    team = []
    for run in runs:
        human_file = os.path.join(folder, run, "human.pth")
        adversary_file = os.path.join(folder, run, "adversary.pth")
        team.append(QNetwork.from_checkpoint(human_file, obs_shape, n_actions))
        team.append(QNetwork.from_checkpoint(adversary_file, obs_shape, n_actions))
    
    # Set all agents in eval mode
    for agent in team:
        agent.eval()
    return team

def play(agent_0, agent_1, env, encode_0: callable, encode_1: callable):
    (heading0, heading1, walls, bike1_pos, bike2_pos), _ = env.reset()
    while True:
        obs0, obs1 = encode_0(walls, bike1_pos, bike2_pos), encode_1(walls, bike2_pos, bike1_pos)
        obs0, obs1 = np.rot90(obs0, k=heading0, axes=(1, 2)).copy(), np.rot90(obs1, k=heading1, axes=(1, 2)).copy()
        obs0, obs1 = torch.as_tensor(obs0, dtype=torch.float32).unsqueeze(0), torch.as_tensor(obs1, dtype=torch.float32).unsqueeze(0)
        
        a0, a1 = agent_0.act(obs0), agent_1.act(obs1)

        (heading0, heading1, walls, bike1_pos, bike2_pos), _, done, _, info = env.step([a0, a1])
        if done:
            return info.get("result")
    
def battle_team(path0, path1):
    env = TronCoreEnv()
    # env = TronView(env, fps=100)
    env.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 3, 25, 25), dtype=np.float32)
    team_1 = load_team(path0, env)

    env.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 2, 25, 25), dtype=np.float32)
    team_2 = load_team(path1, env)

    # Have each team_1 member play against each team_2 member and print results
    results = [0, 0, 0]
    for agent_0 in tqdm(team_1, desc="Team 1", leave=False):
        for agent_1 in team_2:
            # result = play(agent_0, agent_1, env, encode_observation, encode_observation)
            result = play(agent_0, agent_1, env, encode_observation, encode_observation_2channel)
            results[result] += 1
    
    print(f"Results: {results}")
    print(f"Team 1 win rate: {results[1] / (results[1] + results[2]) * 100:.1f}% with {results[0]} draws")

if __name__ == "__main__":
    battle_team("BenchMark", "TwoChannel")
    # battle_team("runs/BenchMark", "runs/TwoChannel")

