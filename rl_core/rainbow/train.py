"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import time, random
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn

import torch#, wandb
import numpy as np
from rich import print

from .common import argp
from .common.rainbow import Rainbow
from .common.utils import LinearSchedule

import gymnasium as gym
from rl_core.env import TronDuoEnv

torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

def make_envs(indx, seed):
    def thunk():
        env = TronDuoEnv()
        env.action_space.seed(seed + indx)
        return env
    return thunk

if __name__ == '__main__':
    args, wandb_log_config = argp.read_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up logging & model checkpoints
    # wandb.init(project='rainbow', save_code=True, config=dict(**wandb_log_config, log_version=100),
    #            mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=[args.wandb_tag] if args.wandb_tag else [])
    save_dir = Path("checkpoints") / "KEK"#wandb.run.name
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(save_dir)

    # create decay schedules for dqn's exploration epsilon and per's importance sampling (beta) parameter
    per_beta_schedule = LinearSchedule(0, initial_value=args.prioritized_er_beta0, final_value=1.0, decay_time=args.prioritized_er_time)

    args.parallel_envs = 5
    envs = gym.vector.SyncVectorEnv([make_envs(i, args.seed) for i in range(args.parallel_envs)])
    states, _ = envs.reset()
    # states = states[:, 0]
    print('[green bold]Start')

    rainbow = Rainbow(envs, args)
    # wandb.watch(rainbow.q_policy)

    # print('[blue bold]Running environment =', args.env_name,
    #       '[blue bold]\nwith action space   =', env.action_space,
    #       '[blue bold]\nobservation space   =', env.observation_space,)
        #   '[blue bold]\nand config:', sn(**wandb_log_config))

    episode_count = 0
    returns = deque(maxlen=100)
    discounted_returns = deque(maxlen=10)
    losses = deque(maxlen=10)
    q_values = deque(maxlen=10)
    grad_norms = deque(maxlen=10)
    iter_times = deque(maxlen=10)

    returns_all = []
    q_values_all = []

    for game_frame in range(0, args.training_frames + 1, args.parallel_envs):
        # print("[yellow bold]Game frame: ", game_frame)
        per_beta = per_beta_schedule(game_frame)

        # reset the noisy-nets noise in the policy
        rainbow.reset_noise(rainbow.q_policy)

        # compute actions to take in all parallel envs, asynchronously start environment step
        states = states[:, 0]  # (5, 3, 25, 25)
        actions = rainbow.act(torch.from_numpy(states).float().to(device))
        # assert (actions[0] == actions).all(), "All actions should be the same"
        actions = np.stack([actions.cpu().numpy(), actions.cpu().numpy()], axis=1)  # (5, 2)         # Stack actions to act as opponent for testing
        next_states, rewards, dones, _, infos = envs.step(actions)
        
        # Add to buffer
        for state, action, reward, done, j in zip(states, actions, rewards, dones, range(args.parallel_envs)):
            rainbow.buffer.put(state, action, reward, done, j=j)

        states = next_states

        # Learning
        if rainbow.buffer.burnedin:
            for train_iter in range(args.train_count):
                if train_iter > 0: rainbow.reset_noise(rainbow.q_policy)
                q, loss, grad_norm = rainbow.train(args.batch_size, beta=per_beta)
                losses.append(loss)
                grad_norms.append(grad_norm)
                q_values.append(q)
                q_values_all.append((game_frame, q))

        # Update target network
        if game_frame % args.sync_dqn_target_every == 0 and rainbow.buffer.burnedin:
            rainbow.sync_Q_target()

        # every 1M frames, save a model checkpoint to disk and wandb
        if game_frame % (500_000-(500_000 % args.parallel_envs)) == 0 and game_frame > 0:
            rainbow.save(game_frame, args=args, run_name="KEK", run_id="KEK", target_metric=np.mean(returns), returns_all=returns_all, q_values_all=q_values_all)
            print(f'Model saved at {game_frame} frames.')

    envs.close()
