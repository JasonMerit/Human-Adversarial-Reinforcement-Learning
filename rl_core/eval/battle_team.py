import os
from tqdm.contrib import itertools
from pathlib import Path
from rich import print
import yaml

from rl_core.env import TronDuoEnv, TronView
from rl_core.env.wrappers import TorchObservationWrapper
from .battle import make_agent
from rl_core.argp import load_args

def make_team(path, env):
    args = load_args(Path("runs") / (path + "_0") / "args.yml")

    # Find all folders in runs that start with the given path
    runs = [f for f in os.listdir('runs/') if os.path.isdir(Path('runs/') / f) and f.startswith(os.path.basename(path))]

    obs_shape = env.unwrapped.obs_shape
    n_actions = env.unwrapped.n_actions

    team = []
    for run in runs:
        team.append(make_agent(Path('runs') / run / "A.pth", obs_shape, n_actions, args))
        team.append(make_agent(Path('runs') / run / "B.pth", obs_shape, n_actions, args))

    return team

def play(agent1, agent2, env):
    obs, _ = env.reset()
    while True:
        obs1, obs2 = obs[:, 0], obs[:, 1]
        a1, a2 = agent1.act(obs1), agent2.act(obs2)
        obs, _, done, _, info = env.step([a1.item(), a2.item()])

        if done:
            return info["result"]


def battle_team(team1, team2, env):
    results = [0, 0, 0]
    for agent1, agent2 in itertools.product(team1, team2, desc="Battling Teams", leave=False):
        result = play(agent1, agent2, env)
        results[result] += 1
    
    total = sum(results)
    return {
        "total": total,
        "wins": results[1],
        "draws": results[0],        
        "losses": results[2],   
        "score": (results[1] + 0.5 * results[0]) / total
    }

def main(folder1, folder2):
    with open(Path("runs") / (folder1 + "_0") / "args.yml", "r") as f:
        size1 = yaml.safe_load(f)["size"]
    with open(Path("runs") / (folder2 + "_0") / "args.yml", "r") as f:
        size2 = yaml.safe_load(f)["size"]
    assert size1 == size2, f"Both teams must have the same environment size. Got {size1} and {size2}."

    env = TronDuoEnv(size1)
    # env = TronView(TronDuoEnv(size1))
    env = TorchObservationWrapper(env, device="cpu")

    team1, team2 = make_team(folder1, env), make_team(folder2, env)
    print(f"Team {folder1} (x{len(team1)}) VS. Team {folder2} (x{len(team2)}) in Tron {size1}x{size1}")
    
    results = battle_team(team1, team2, env)
    
    print(f"Results: {results}")
    print(f"Team {args.folder1} score: {results['score']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model in the Tron environment.")
    parser.add_argument("folder1", type=str, default="", help="Path folder of trained model checkpoints.")
    parser.add_argument("folder2", type=str, default="", help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    main(args.folder1, args.folder2)
    
    

    
