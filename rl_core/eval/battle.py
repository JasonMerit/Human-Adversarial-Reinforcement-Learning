from pathlib import Path
import torch
import gymnasium as gym
import os

from rich import print

from rl_core.env import TronDuoEnv, TronView, TronEnv
from rl_core.env.wrappers import TorchObservationWrapper

# from rl_core.rainbow.common.networks import get_model


def make_dqn(path, obs_shape, n_actions):
    from rl_core.agents.dqn import QNetwork
    return QNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_rainbow(path):
    from rl_core.rainbow.common.rainbow import Rainbow
    return Rainbow.from_checkpoint(path, obs_shape=(3, 25, 25), n_actions=3, device="cpu")

def make_clean_rainbow(path, obs_shape, n_actions):
    from rl_core.clean_rainbow.network import DuelingNetwork
    return DuelingNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def get_agent_files(agent_folder, num_agents):
    # Select the appropriate agent class based on the file name
    # find all files ending in .pth in the given path
    agent_folder = Path("runs/") / agent_folder
    files = [f for f in os.listdir(agent_folder) if f.endswith(".pth")]
    # Sort agents based on indx, e.g. A_5000.pth should be after A_4000.pth
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    return [os.path.join(agent_folder, f) for f in files[:num_agents]]
    
def rainbow_act(policy, obs):
    with torch.no_grad():
        action_values = policy(obs, advantages_only=True)
        actions = torch.argmax(action_values, dim=1)
    return actions.item()

def cleanrain_act(policy, obs):
    with torch.no_grad():
        q = policy(torch.as_tensor(obs))
        if len(q.shape) == 3:  # Distributional case
            q = torch.sum(q * policy.support, dim=2)
        return torch.argmax(q, dim=1).cpu().numpy().item()

def play(agent1, agent2, env: TronDuoEnv):
    obs, _ = env.reset()
    kek = 0
    while True:
        obs1, obs2 = obs[:, 0], obs[:, 1]
        a1, a2 = cleanrain_act(agent1, obs1), cleanrain_act(agent2, obs2)
        # a1, a2 = rainbow_act(agent1, obs1), rainbow_act(agent2, obs1)  # Use the loaded model to select an action
        # obs, _, done, _, info = env.step(a1)
        obs, _, done, _, info = env.step([a1, a2])
        kek += 1

        if done:
            obs, _ = env.reset()
            print(info.get("result"), f"Total steps: {kek}")
            kek = 0
    
def battle(folder1, folder2):
    # env = TronDuoEnv()
    # env = TronView(TronEnv())
    size = 25
    env = TronView(TronDuoEnv(size))
    env = TorchObservationWrapper(env, device="cpu")

    n_actions = env.action_space.nvec[0]           # should be 3
    obs_shape = env.observation_space.shape[-3:]  # Ignore the stacked observations

    if folder2 == "":
        folder = Path("runs") / folder1
        path1, path2 = folder / "A.pth", folder / "B.pth"
        path1, path2 = get_agent_files(folder1, num_agents=2)
        # agent1, agent2 = make_clean_rainbow(path1, obs_shape), make_clean_rainbow(path2, obs_shape)
        # agent1, agent2 = make_rainbow(path1), make_rainbow(path2)
        agent1, agent2 = make_dqn(path1, obs_shape), make_dqn(path2, obs_shape)

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
    