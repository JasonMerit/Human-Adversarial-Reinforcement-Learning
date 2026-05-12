from pathlib import Path
import torch
import gymnasium as gym
import os

from rich import print

from rl_core.env import TronDuoEnv, TronView, TronEnv
from rl_core.env.wrappers import TorchObservationWrapper
from rl_core.agents.rainbow import DuelingNetwork

# from rl_core.rainbow.common.networks import get_model


def make_dqn(path, obs_shape, n_actions):
    from rl_core.agents.dqn import QNetwork
    return QNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_rainbow(path, obs_shape, n_actions):
    return DuelingNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

# def get_agent_files(agent_folder, num_agents):
#     # Select the appropriate agent class based on the file name
#     # find all files ending in .pth in the given path
#     agent_folder = Path("runs/") / agent_folder
#     print(agent_folder)
#     files = [f for f in os.listdir(agent_folder) if f.endswith(".pth")]
#     print(files)
#     # Sort agents based on indx, e.g. A_5000.pth should be after A_4000.pth
#     files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
#     print(files)
#     return [os.path.join(agent_folder, f) for f in files[:num_agents]]

    
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
        # a1, a2 = cleanrain_act(agent1, obs1), cleanrain_act(agent2, obs2)
        # a1, a2 = rainbow_act(agent1, obs1), rainbow_act(agent2, obs1)  # Use the loaded model to select an action
        a1, a2 = agent1.act(obs1), agent2.act(obs2)
        # obs, _, done, _, info = env.step(a1)
        obs, _, done, _, info = env.step([a1, a2])
        kek += 1

        if done:
            obs, _ = env.reset()
            print(info.get("result"), f"Total steps: {kek}")
            kek = 0
    
def battle(folder1, folder2):
    folder = Path("runs") / folder1
    path1, path2 = folder / "A.pth", folder / "B.pth"
    size = DuelingNetwork.size_from_checkpoint(path1)
    
    env = TronView(TronDuoEnv(size))
    env = TorchObservationWrapper(env, device="cpu")
    n_actions = env.unwrapped.n_actions
    obs_shape = env.unwrapped.obs_shape

    agent1, agent2 = make_rainbow(path1, obs_shape, n_actions), make_rainbow(path2, obs_shape, n_actions)

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
    