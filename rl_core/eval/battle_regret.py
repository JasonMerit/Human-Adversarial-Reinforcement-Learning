from pathlib import Path
import torch, yaml

from rich import print

from rl_core.env import TronDuoEnv, TronView
from rl_core.env.wrappers import TorchObservationWrapper
from rl_core.agents.rainbow import DuelingNetwork
from rl_core.argp import load_args

from rl_core.MCTS.agent_regret import MCTS, Node
from rl_core.MCTS.vec_duo_env import VecTronDuoEnv

def make_dqn(path, obs_shape, n_actions):
    from rl_core.agents.dqn import QNetwork
    return QNetwork.from_checkpoint(path, obs_shape, n_actions, device="cpu")

def make_agent(path, obs_shape, n_actions, args):
    agent = DuelingNetwork.from_checkpoint(path, obs_shape, n_actions, args, device="cpu")
    agent.eval()
    return agent

def play(agent1, agent2, mcts, env: TronDuoEnv):
    # Opponent is DQN, because MCTS assumes p1
    results = [0, 0, 0]
    
    for _ in range(10):
        history = []
        obs, _ = env.reset()
        mcts.reset()
        root = Node(env.state)

        while True:
            obs1, obs2 = obs[:, 0], obs[:, 1]
            # a1, a2 = agent1.act(obs1), agent2.act(obs2)
            a2 = agent2.act(obs2).item()

            a1 = mcts.plan(root, sims=200)
            obs, _, done, _, info = env.step([a1, a2])
            history.append(a1)

            child = root.children[a1, a2]  # Reuse the subtree if it exists
            # if not isinstance(child, Node):
            #     print(f"Child is not of [cyan]Node[/cyan]. Got {type(child)} instead")
            #     quit()
            root = Node(env.state) if child is None else child
            child.parent = None
            # if child is None:
            #     root = Node(env.state)
            # else:
            #     child.parent = None
            #     root = child

            if done:
                print(info.get("result"), f"Total steps: {len(history)}")
                results[info.get("result")] += 1
                # print(history[:len(history)//2])  # Print first half
                # print(history[len(history)//2:])  # Print second half
                print(results)
                break
    
def battle(folder):
    if folder[:4] != "runs":
        folder = Path("runs") / folder
    else:
        folder = Path(folder)
    
    assert folder.exists(), f"Folder not found: {folder}"
    args = load_args(folder)
    size = args.size

    # env = TronDuoEnv(size)
    env = TronView(TronDuoEnv(size))
    env = TorchObservationWrapper(env, device="cpu")
    n_actions = env.unwrapped.n_actions
    obs_shape = env.unwrapped.obs_shape

    path1, path2 = folder / "A.pth", folder / "B.pth"
    agent1, agent2 = make_agent(path1, obs_shape, n_actions, args), make_agent(path2, obs_shape, n_actions, args)

    # Regret setup
    NUM_ENVS = 64
    sim_env = TronDuoEnv(size)
    sim_env.reset()
    sim_envs = VecTronDuoEnv(NUM_ENVS, size)
    mcts = MCTS(sim_env, sim_envs, rollouts=NUM_ENVS)

    play(agent1, agent2, mcts, env)
        

if __name__ == "__main__":
    # Get args
    import argparse
    parser = argparse.ArgumentParser(description="Play a trained model against the Regret matching.")
    parser.add_argument("folder", type=str, default="", help="Path folder of trained model checkpoints.")
    args = parser.parse_args()
    battle(args.folder)
    