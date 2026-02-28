from tron_env import TronEnv, TronView
from rl_core.cleanrl.cleanrl.ppo_cnn import Agent

import torch
import gymnasium as gym
import numpy as np

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
    obs_shape = env.observation_space.shape  # (3, H, W)
    n_actions = env.action_space.n           # should be 3
    return Agent.from_checkpoint(agent_path, obs_shape, n_actions)

def play():
    env = TronView(TronEnv())
    env = TorchObservationWrapper(env, device="cpu")
    state, _ = env.reset()

    agent = load_agent("runs\Tron-v0__ppo_cnn__1__1772283291\ppo_cnn.pth", env)
    agent.eval()  # Set the agent to evaluation mode
    
    while True:
        action = agent.get_action_and_value(state)[0].item()  # Use the loaded model to select an action
        state, reward, done, _, info = env.step(action)

        if done:
            state, _ = env.reset()

if __name__ == "__main__":
    play()