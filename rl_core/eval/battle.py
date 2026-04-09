from pathlib import Path
import torch
import gymnasium as gym
import os

from rich import print

from rl_core.env import TronDuoEnv, TronView
from rl_core.agents import QNetwork, ActorCriticNetwork
from rl_core.env.wrappers import TorchObservationWrapper
from rl_core.rainbow.common.rainbow import Rainbow
# from rl_core.rainbow.common.networks import get_model



def make_rainbow(path):
    return Rainbow.from_checkpoint(path, obs_shape=(3, 25, 25), n_actions=3, device="cpu")

def get_agent_files(agent_folder, num_agents):
    # Select the appropriate agent class based on the file name
    # find all files ending in .pth in the given path
    agent_folder = Path("runs/") / agent_folder
    files = [f for f in os.listdir(agent_folder) if f.endswith(".pth")]
    # Sort agents based on indx, e.g. A_5000.pth should be after A_4000.pth
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    return [os.path.join(agent_folder, f) for f in files[:num_agents]]
    
def load_agent(agent_path1, agent_path2, env):
    # Select the appropriate agent class based on the file name
    # e.g. runs\self_train_595179
    human_file = agent_path1 + "/adversary.pth"
    adv_file = agent_path2 + "/adversary.pth"

    obs_shape = env.observation_space.shape[-3:]  # (3, H, W)
    n_actions = env.action_space.nvec[0]           # should be 3
    return QNetwork.from_checkpoint(human_file, obs_shape, n_actions), QNetwork.from_checkpoint(adv_file, obs_shape, n_actions)

def rainbow_act(policy, obs):
    with torch.no_grad():
        action_values = policy(obs, advantages_only=True)
        actions = torch.argmax(action_values, dim=1)
    #  tensor([0]), Agent 2 action: tensor([2])
    # return the integer value of the action, e.g. 0, 1, or 2
    return actions.item()

def play(agent1, agent2, env):
    obs, _ = env.reset()
    while True:
        obs0, obs1 = obs[:, 0], obs[:, 1]
        a0, a1 = rainbow_act(agent2, obs0), rainbow_act(agent1, obs1)  # Use the loaded model to select an action
        # a0, a1 = rainbow_act(agent1, obs0), rainbow_act(agent2, obs1)  # Use the loaded model to select an action
        obs, reward, done, _, info = env.step([a0, a1])

        if done:
            obs, _ = env.reset()
            print(info.get("result"))
    
def battle(folder1, folder2):
    # env = TronDuoEnv()
    env = TronView(TronDuoEnv())
    env = TorchObservationWrapper(env, device="cpu")
    obs, _ = env.reset()

    if folder2 == "":
        path1, path2 = get_agent_files(folder1, num_agents=2)
        agent1, agent2 = make_rainbow(path1), make_rainbow(path2)
    else:
        raise Exception("Currently only supports self-play. Please provide a single folder with two checkpoints for the agents.")
        path1 = get_agent_files(folder1, num_agents=1)[0]
        path2 = get_agent_files(folder2, num_agents=1)[0]
        agent1, agent2 = load_agent(path1, path2, env)

    agent1.eval()
    agent2.eval()
    play(agent1, agent2, env)
        

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("folder1", type=str, default="", help="Path folder of trained model checkpoints.")
    parser.add_argument("folder2", type=str, nargs="?", default="", help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    battle(args.folder1, args.folder2)
    