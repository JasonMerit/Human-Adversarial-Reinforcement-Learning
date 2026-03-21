from rl_core.tron_env.tron_env.env import TronDuoEnv
from rl_core.tron_env.tron_env.wrappers import TronView

import torch
import gymnasium as gym
import numpy as np
import os

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

def load_agent(agent_path, env):
    # Select the appropriate agent class based on the file name
    if "self" in agent_path.lower():  # Takes the folder where models resisde
        # e.g. runs\self_train_595179
        human_file = agent_path + "/human.pth"
        adv_file = agent_path + "/adversary.pth"

        from rl_core.agents.dqn import QNetwork
        obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
        n_actions = env.action_space.nvec[0]           # should be 3
        return QNetwork.from_checkpoint(human_file, obs_shape, n_actions), QNetwork.from_checkpoint(adv_file, obs_shape, n_actions)
    
    obs_shape = env.observation_space.shape  # (3, H, W)
    n_actions = env.action_space.n           # should be 3
    if "train" in agent_path.lower():
        from rl_core.agents.dqn import QNetwork
        return QNetwork.from_checkpoint(agent_path, obs_shape, n_actions)
    elif "dqn" in agent_path.lower():
        from rl_core.cleanrl.cleanrl.dqn import QNetwork as QNetwork
        return QNetwork.from_checkpoint(agent_path,obs_shape, n_actions)
    elif "ppo" in agent_path.lower():
        from rl_core.cleanrl.cleanrl.ppo import Agent
        return Agent.from_checkpoint(agent_path, obs_shape, n_actions)
    elif "rainbow" in agent_path.lower():
        from rl_core.cleanrl.cleanrl.rainbow import NoisyDuelingDistributionalNetwork
        return NoisyDuelingDistributionalNetwork.from_checkpoint(agent_path, obs_shape, n_actions)
    else:
        raise ValueError(f"Unknown agent type in path: {agent_path}")

def play(path):
    env = TronView(TronDuoEnv())
    env = TorchObservationWrapper(env, device="cpu")
    obs, _ = env.reset()

    human, adversary = load_agent(path, env)
    human.eval()  # Set the agent to evaluation mode
    adversary.eval()  # Set the agent to evaluation mode

    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = human(obs0), adversary(obs1)  # Use the loaded model to select an action
        obs, reward, done, _, info = env.step([a0, a1])

        if done:
            obs, _ = env.reset()
            print(info.get("result"))

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("path", type=str, help="Path to the trained model checkpoint.")
    args = parser.parse_args()
    play(args.path)
    # play("runs/dqn_1/dqn.pth")
    # play("runs/ppo_cnn_881413/ppo_cnn.pth")