from pathlib import Path
import torch, yaml

from rich import print

from rl_core.env import TronDuoEnv, TronView
from rl_core.env.wrappers import TorchObservationWrapper
from rl_core.agents.rainbow import DuelingNetwork
from rl_core.argp import load_args

def make_dqn(path, obs_shape, n_actions):
    from rl_core.agents.dqn import QNetwork
    return QNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_agent(path, obs_shape, n_actions, args):
    agent = DuelingNetwork.from_checkpoint(path, obs_shape, n_actions, args, device="cpu")
    agent.eval()
    return agent

def play(agent1, agent2, env: TronDuoEnv):
    obs, _ = env.reset()
    history = []
    while True:
        obs1, obs2 = obs[:, 0], obs[:, 1]
        a1, a2 = agent1.act(obs1), agent2.act(obs2)
        obs, _, done, _, info = env.step([a1.item(), a2.item()])
        history.append(a1)

        if done:
            obs, _ = env.reset()
            print(info.get("result"), f"Total steps: {len(history)}")
            print(history[:len(history)//2])  # Print first half
            print(history[len(history)//2:])  # Print second half
            break
    
def battle(folder):
    folder = Path("runs") / folder
    assert folder.exists(), f"Folder not found: {folder}"
    args = load_args(folder / "args.yml")
    size = args.size

    # env = TronDuoEnv(size)
    env = TronView(TronDuoEnv(size))
    env = TorchObservationWrapper(env, device="cpu")
    n_actions = env.unwrapped.n_actions
    obs_shape = env.unwrapped.obs_shape

    path1, path2 = folder / "A.pth", folder / "B.pth"
    agent1, agent2 = make_agent(path1, obs_shape, n_actions, args), make_agent(path2, obs_shape, n_actions, args)

    play(agent1, agent2, env)
        

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("folder", type=str, default="", help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    battle(args.folder)