
import yaml
from pathlib import Path
from rl_core.eval.battle import make_agent

from rl_core.env import TronDuoEnv, TronView
from rl_core.env.wrappers import TorchObservationWrapper


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
    agent1, agent2 = make_agent(path1, obs_shape, n_actions, args), make_agent(path2, obs_shape, n_actions, args)

    play(agent1, agent2, env)



battle("Bench15_4")