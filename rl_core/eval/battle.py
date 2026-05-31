from pathlib import Path
import torch, yaml

from rich import print

from rl_core.env import TronDuoEnv, TronView
from rl_core.env.wrappers import TorchObservationWrapper
from rl_core.agents.rainbow import DuelingNetwork
from rl_core.MCTS.knegt import KnegtNetwork

def make_dqn(path, obs_shape, n_actions):
    from rl_core.agents.dqn import QNetwork
    return QNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_rainbow(path, obs_shape, n_actions):
    return DuelingNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_knegt(path, obs_shape, n_actions):
    return KnegtNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_agent(path, obs_shape, n_actions):
    # Load the dict and determine type from keys
    net_dict = torch.load(path, weights_only=True, map_location="cpu")
    if "opp_head.0.weight" in net_dict:  # Only knegt has opponent model
        agent = make_knegt(path, obs_shape, n_actions)
    else:
        agent =  make_rainbow(path, obs_shape, n_actions)
    
    agent.eval()
    return agent

def cleanrain_act(policy, obs):
    with torch.no_grad():
        q = policy(torch.as_tensor(obs))
        if len(q.shape) == 3:  # Distributional case
            q = torch.sum(q * policy.support, dim=2)
        return torch.argmax(q, dim=1).cpu().numpy().item()

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
            quit()
    
def battle(folder):
    folder = Path("runs") / folder
    with open(folder / "args.yml", "r") as f:
        args = yaml.safe_load(f)
        size = args['size']

    # env = TronDuoEnv(size)
    env = TronView(TronDuoEnv(size))
    env = TorchObservationWrapper(env, device="cpu")
    n_actions = env.unwrapped.n_actions
    obs_shape = env.unwrapped.obs_shape

    path1, path2 = folder / "A.pth", folder / "B.pth"
    agent1, agent2 = make_agent(path1, obs_shape, n_actions), make_agent(path2, obs_shape, n_actions)

    play(agent1, agent2, env)
        

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("folder", type=str, default="", help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    battle(args.folder)