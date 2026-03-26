import time

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

def load_agent(agent_path, env):
    # Select the appropriate agent class based on the file name
    # e.g. runs\self_train_595179
    human_file = agent_path + "/human.pth"
    adv_file = agent_path + "/adversary.pth"

    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    return QNetwork.from_checkpoint(human_file, obs_shape, n_actions), QNetwork.from_checkpoint(adv_file, obs_shape, n_actions)
    
def play(path):
    env = TronView(TronDuoEnv())
    env = TorchObservationWrapper(env, device="cpu")
    obs, _ = env.reset()

    human, adversary = load_agent(path, env)
    human.eval()  # Set the agent to evaluation mode
    adversary.eval()  # Set the agent to evaluation mode

    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = human.act(obs0), adversary.act(obs1)  # Use the loaded model to select an action
        obs, reward, done, _, info = env.step([a0, a1])

        if done:
            time.sleep(100)
            obs, _ = env.reset()
            print(info.get("result"))

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("path", type=str, help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    play(args.path)
    