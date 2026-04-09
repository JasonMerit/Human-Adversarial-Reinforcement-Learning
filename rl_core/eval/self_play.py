import os

from rl_core.env import TronDuoEnv, TronView
from rl_core.agents.dqn import QNetwork

import torch
import gymnasium as gym

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

def load_agent(agent_folder, env):
    # Select the appropriate agent class based on the file name
    # find all files ending in .pth in the given path
    files = [f for f in os.listdir(agent_folder) if f.endswith(".pth")]
    # Sort agents based on creation date
    files.sort(key=lambda x: os.path.getctime(os.path.join(agent_folder, x)), reverse=True)
    # Select the two most recent files
    files = files[:2]
    print(files)
    quit()


    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    return [QNetwork.from_checkpoint(os.path.join(agent_folder, f), obs_shape, n_actions) for f in files], files
    agent1_file = agent_path + "/agent1.pth"
    agent2_file = agent_path + "/agent2.pth"

    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    return QNetwork.from_checkpoint(agent1_file, obs_shape, n_actions), QNetwork.from_checkpoint(agent2_file, obs_shape, n_actions)
    
def play(path):
    env = TronView(TronDuoEnv())
    env = TorchObservationWrapper(env, device="cpu")
    obs, _ = env.reset()

    agent1, agent2 = load_agent(path, env)
    agent1.eval()  # Set the agent to evaluation mode
    agent2.eval()  # Set the agent to evaluation mode

    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = agent1.act(obs0), agent2.act(obs1)  # Use the loaded model to select an action
        obs, reward, done, _, info = env.step([a0, a1])

        if done:
            obs, _ = env.reset()
            print(info.get("result"))

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("path", type=str, help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    play(args.path)
    