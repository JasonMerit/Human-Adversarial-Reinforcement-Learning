import torch
import gymnasium as gym

from rl_core.env import TronDuoEnv, TronView
from rl_core.agents.dqn import QNetwork
from rl_core.env.wrappers import TorchObservationWrapper


def load_agent(agent_path0, agent_path1, env):
    # Select the appropriate agent class based on the file name
    # e.g. runs\self_train_595179
    human_file = agent_path0 + "/adversary.pth"
    adv_file = agent_path1 + "/adversary.pth"

    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    return QNetwork.from_checkpoint(human_file, obs_shape, n_actions), QNetwork.from_checkpoint(adv_file, obs_shape, n_actions)
    
def battle(path0, path1):
    env = TronView(TronDuoEnv())
    env = TorchObservationWrapper(env, device="cpu")
    obs, _ = env.reset()

    human, adversary = load_agent(path0, path1, env)
    human.eval()  # Set the agent to evaluation mode
    adversary.eval()  # Set the agent to evaluation mode

    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = human.act(obs0), adversary.act(obs1)  # Use the loaded model to select an action
        obs, reward, done, _, info = env.step([a0, a1])

        if done:
            obs, _ = env.reset()
            print(info.get("result"))

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("path0", type=str, help="Path folder of trained model checkpoints.")
    parser.add_argument("path1", type=str, help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    battle(args.path0, args.path1)
    