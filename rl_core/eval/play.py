from pathlib import Path
import torch
import gymnasium as gym
import os

from rich import print

from rl_core.env.wrappers import TronPlay, TronView

def make_dqn(path, obs_shape, n_actions):
    from rl_core.agents.dqn import QNetwork
    return QNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_rainbow(path, obs_shape, n_actions):
    from rl_core.agents.rainbow import DuelingNetwork
    return DuelingNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def get_agent_files(agent_folder, num_agents):
    kek = Path("runs/") / agent_folder
    return kek / "A.pth", kek / "B.pth"
    
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

def play(env: TronPlay):
    obs, _ = env.reset()
    kek = 0
    while True:
        done = env.step(0)
        kek += 1

        if done:
            obs, _ = env.reset()
            kek = 0
    
def battle(path):
    assert path[:4] == "runs", "Path should be a folder in the 'runs' directory containing the trained model checkpoints."
    
    size = 25
    n_actions = 3
    obs_shape = (3, size, size)
    if path[-4:] != ".pth":
        path = Path(path) / "A.pth"

    agent = make_rainbow(path, obs_shape, n_actions)
    env = TronPlay(agent, size)  # Wrap the environment to play against agent2
    env = TronView(env)

    agent.eval()
    play(env)
        

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("path", type=str, default="", help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    battle(args.path)