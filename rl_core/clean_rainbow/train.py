# From https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
import random
import time

from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rich import print

from .argp import read_args
from .network import Rainbow
from .utils import TimerRegistry
from rl_core.env import TronDuoEnv

def make_envs(indx, seed):
    def thunk():
        env = TronDuoEnv()
        env.action_space.seed(seed + indx)
        return env
    return thunk

if __name__ == "__main__":
    args = read_args()

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_envs(i, args.seed) for i in range(args.num_envs)])
    agent1 = Rainbow(envs.single_action_space.nvec[0], args, device)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = obs[:, 0]
    for global_step in tqdm(range(args.total_timesteps)):
        # anneal PER beta to 1
        # rb.beta = min(1.0, args.prioritized_replay_beta + global_step * (1.0 - args.prioritized_replay_beta) / args.total_timesteps)

        a = agent1.act(obs)

        actions = np.stack([a, a], axis=1) 
        next_obs, rewards, dones, _, infos = envs.step(actions)
        next_obs = next_obs[:, 0]

        agent1.rb.add(obs, a, rewards, next_obs, dones)

        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                agent1.learn()

            # update target network
            if global_step % args.target_network_frequency == 0:
                agent1.update_target()

    envs.close()
    writer.close()
    TimerRegistry.report()