"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import time, random
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn
import os
import yaml

import torch#, wandb
import numpy as np
from rich import print
from tqdm import tqdm

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
    print(f"=====|  Training with seed {args.seed} on device {device}", "[yellow bold](debug mode)[/yellow bold]" if args.debug else "", "|=====")

    # Logging and saving model
    # log_every_frames = 1
    log_every_frames = max(1, args.total_timesteps // 100)
    save_every_frames = max(1, args.total_timesteps // args.total_checkpoints)
    
    if args.save:
        save_folder = "runs/" + args.exp_name
        i = 0
        while os.path.exists(save_folder + f"_{i}"):
            i += 1
        save_folder += f"_{i}/"
        os.makedirs(save_folder)
        with open(save_folder + "args.yml", "w") as f:
                yaml.dump(vars(args), f)
        print(f"Models will be saved to {save_folder}!")
    else:
        print("Models will NOT be saved!")
    # save_dir = Path("checkpoints") / "KEK"#wandb.run.name
    # save_dir.mkdir(parents=True, exist_ok=True)
    # args.save_dir = str(save_dir)

    # create decay schedules for dqn's exploration epsilon and per's importance sampling (beta) parameter
    eps_schedule = LinearSchedule(0, initial_value=args.init_eps, final_value=args.final_eps, decay_time=args.eps_decay_frac * args.total_timesteps)
    per_beta_schedule = LinearSchedule(0, initial_value=args.prioritized_er_beta0, final_value=1.0, decay_time=args.prioritized_er_time)

    envs = gym.vector.SyncVectorEnv([make_envs(i, args.seed) for i in range(args.num_envs)])
    states, _ = envs.reset()

    agent1 = Rainbow(envs, args, device)
    agent2 = Rainbow(envs, args, device)

    returns_all = []
    q_values_all = []

    print('[green bold]Start')
    start_time = time.time()
    results = [0, 0, 0]  # tie/win/loss counts for agent1
    
    # Set pbar to fill within total_time 
    total_time = 60
    pbar = tqdm(total=total_time) 
    try:
        for game_frame in range(0, args.total_timesteps + 1, args.num_envs):
            # print("[yellow bold]Game frame: ", game_frame)
            eps = eps_schedule(game_frame)
            per_beta = per_beta_schedule(game_frame)

            # reset the noisy-nets noise in the policy
            agent1.reset_noise(agent1.q_policy)
            agent2.reset_noise(agent2.q_policy)

            # compute actions to take in all parallel envs, asynchronously start environment step
            obs1, obs2 = states[:, 0], states[:, 1]  # (5, 3, 25, 25)
            
            actions1 = agent1.act(torch.from_numpy(obs1).float().to(device)).cpu().numpy()
            explore_mask = np.random.rand(args.num_envs) < eps
            actions1[explore_mask] = np.random.randint(0, 3, size=explore_mask.sum())
            
            explore_mask = np.random.rand(args.num_envs) < eps
            actions2 = agent2.act(torch.from_numpy(obs2).float().to(device)).cpu().numpy()
            actions2[explore_mask] = np.random.randint(0, 3, size=explore_mask.sum())

            # assert not (actions1 == actions2).all(), f"Same actions for both agents :( \n{actions1}\n{actions2}"
            # assert not (actions1[0] == actions1).all(), f"Same action for agent1 :( \n{actions1}"
            next_states, rewards, dones, _, infos = envs.step(np.stack([actions1, actions2], axis=1))
            
            # Buffer
            for state, action, reward, done, j in zip(obs1, actions1, rewards, dones, range(args.num_envs)):
                agent1.buffer.put(state, action, reward, done, j=j)
            for state, action, reward, done, j in zip(obs2, actions2, rewards, dones, range(args.num_envs)):
                agent2.buffer.put(state, action, -reward, done, j=j)

            states = next_states

            # Iterate over terminations to log episode results
            for i in np.where(dones)[0]:  # Update results for any env that is done
                results[infos["final_info"][i]['result']] += 1

            # Learning
            if agent1.buffer.burnedin:
                for train_iter in range(args.train_count):
                    if train_iter > 0: # reset noise for each training iteration (following the rainbow paper's implementation where noise is reset every time a batch is sampled from the buffer)
                        agent1.reset_noise(agent1.q_policy)
                        agent2.reset_noise(agent2.q_policy)
                    q, loss, grad_norm = agent1.train(args.batch_size, beta=per_beta)
                    q, loss, grad_norm = agent2.train(args.batch_size, beta=per_beta)

            # Update target network
            if game_frame % args.sync_dqn_target_every == 0 and agent1.buffer.burnedin:
                agent1.sync_Q_target()
                agent2.sync_Q_target()
            
            # Logging
            if game_frame % log_every_frames < args.num_envs and game_frame > 0:
                sps = int(game_frame / (time.time() - start_time))
                elapsed = time.time() - start_time
                progress = game_frame / args.total_timesteps
                eta = elapsed * (1/progress - 1)
                print(f"{progress*100:.1f}% - SPS: {sps} - Results: {results}")
                print(f"{eta/60:.1f} minutes left...", end='\r')

            # every 1M frames, save a model checkpoint to disk and wandb
            if game_frame % save_every_frames < args.num_envs and game_frame > 0:
                agent1.save(save_folder + f"A_{game_frame}.pth", verbose=True)
                agent2.save(save_folder + f"B_{game_frame}.pth", verbose=True)

            if args.debug:
                elapsed = int(time.time() - start_time)
                pbar.update(elapsed - pbar.n)  
                if elapsed >= total_time:  
                    print(f"\nStopping training after {total_time} seconds for testing purposes.")
                    break
    
    finally:
        if args.save:
            with open(save_folder + "results.yml", "w") as f:
                yaml.dump({
                    "steps_taken": game_frame, 
                    "training_time_hours": (time.time() - start_time) / 3600,
                    "results": results,
                    }, f)
            agent1.save(save_folder + f"A_{game_frame}.pth", verbose=True)
            agent2.save(save_folder + f"B_{game_frame}.pth", verbose=True)

        print(f"Finished after {time.time() - start_time:.1f} seconds")

    envs.close()
