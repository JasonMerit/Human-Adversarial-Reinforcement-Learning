# From https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
import random, os, time, shutil
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rich import print
import yaml
from tqdm import trange

from .argp import read_args
from .agents import RainbowAgent
from .MCTS.knegt import KnegtAgent
from .utils import TimerRegistry
from rl_core.MCTS.vec_duo_tron import VecTronDuoEnv

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = read_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    writer = None
    if args.track:
        i = 0
        if args.hpc:
            if "[" in args.exp_name:  # If using job arrays, remove brackets for folder name
                args.exp_name = args.exp_name.split("[")[0]
            base_folder = f"rl_core/HPC/runs/{args.exp_name}"  # Move to HPC folder
        else:
            base_folder = f"runs/{args.exp_name}"

        while True:
            save_folder = f"{base_folder}_{i}/"

            try:
                os.makedirs(save_folder)
                break
            except FileExistsError:
                i += 1
            
        with open(save_folder + "args.yml", "w") as f:
            yaml.dump(vars(args), f)
        
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = SummaryWriter(save_folder)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        if args.save:
            print(f"Models will be saved to {save_folder}")
        else:
            print("Models will NOT be saved!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=====| {args.exp_name} on {device} with seed {args.seed}", "[yellow bold](debug mode)[/yellow bold]" if args.debug else "", "|=====")

    # Handle parallel envs
    total_loops = args.total_timesteps // args.num_envs
    learn_start_loop = args.learning_starts // args.num_envs
    target_every = args.target_network_frequency // args.num_envs
    train_count = args.num_envs // args.train_frequency
    log_every = max(1, total_loops // 100)
    save_every = max(1, total_loops // args.total_checkpoints)
    
    # Env
    envs = VecTronDuoEnv(args.num_envs, args.size, render=args.render)
    obs_shape = envs.obs_shape
    n_actions = envs.n_actions

    obs, infos = envs.reset()
    state = infos["state"]

    print(f"Observation shape: {obs_shape}, Action space: {n_actions}")

     
    if args.knegt:
        agent1 = KnegtAgent(0, obs_shape, n_actions, args, device, writer)
        agent2 = KnegtAgent(1, obs_shape, n_actions, args, device, writer)
    else:
        agent1 = RainbowAgent(0, obs_shape, n_actions, envs.encode, args, device, writer)
        agent2 = RainbowAgent(1, obs_shape, n_actions, envs.encode, args, device, writer)
    

    # Logging
    # TimerRegistry.disable()
    start_time = time.time()
    results = [0, 0, 0]
    total_episodes = 0
    # total_episode_lengths = 0
    episode_lengths = np.zeros(args.num_envs, dtype=int)

    
    ep_lens = deque(maxlen=100)

    for global_step in range(1, total_loops + 1):
    # for global_step in trange(1, total_loops + 1, desc="Training"):
        TimerRegistry.start()
        a1 = agent1.act(obs[:, 0])
        a2 = agent2.act(obs[:, 1])

        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * total_loops, global_step)
        explore_mask = np.random.rand(args.num_envs) < epsilon
        a1[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())
        explore_mask = np.random.rand(args.num_envs) < epsilon
        a2[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())
        actions = np.stack([a1, a2], axis=1) 

        obs, rewards, dones, _, infos = envs.step(actions)
        next_state = infos["state"]

        agent1.add(state, actions, rewards, next_state, dones)
        agent2.add(state, actions, -rewards, next_state, dones)

        state = next_state
        episode_lengths += 1
        TimerRegistry.stop("env_step")

        # Training
        if global_step > learn_start_loop:
            # for _ in range(train_count):
            agent1.learn()
            # if args.debug:
            #     print(f"[green]Success[/green] after {time.time() - start_time:.1f} seconds")
            #     quit()
            agent2.learn()

            # update target network
            if global_step % target_every == 0:
                agent1.update_target()
                agent2.update_target()
            
                # quit()
            # print("one learning step")
        
        # Logging
        for i in np.where(dones)[0]:  # Update results for any env that is done
            results[infos["result"][i]] += 1
            # total_episode_lengths += episode_lengths[i]
            ep_lens.append(episode_lengths[i])
            total_episodes += 1
            episode_lengths[i] = 0
        #     if writer:
        #         writer.add_scalar("charts/draw_percentage", results[0] / total_episodes, total_episodes)
        #         writer.add_scalar("charts/agent1_win_percentage", results[1] / total_episodes, total_episodes)
        #         writer.add_scalar("charts/agent2_win_percentage", results[2] / total_episodes, total_episodes)
        #         writer.add_scalar("charts/avg_episode_length", total_episode_lengths / total_episodes, total_episodes)

        if global_step % log_every == 0:
            sps = int(global_step * args.num_envs / (time.time() - start_time))
            elapsed = time.time() - start_time
            progress = global_step / total_loops
            eta = elapsed * (1/progress - 1)
            # print(f"{progress*100:.1f}% - {epsilon=:.3f}")
            # epi_len = total_episode_lengths / total_episodes if total_episodes > 0 else 0
            avg = sum(ep_lens) / 100
            print(f"{progress*100:.1f}% - SPS: {sps} - epi_len: {avg:.2f} - {eta/60:.1f} minutes left...")
        
        env_step = global_step * args.num_envs
        if args.save and global_step % save_every == 0:
            agent1.save(save_folder + f"A_{env_step}.pth")
            agent2.save(save_folder + f"B_{env_step}.pth")

    TimerRegistry.report()

    if args.track:
        if args.save:
            agent1.save(save_folder + "A.pth")
            agent2.save(save_folder + "B.pth")

        with open(save_folder + "results.yml", "w") as f:
            yaml.dump({
                "results": results, 
                "steps_taken": global_step * args.num_envs,
                "global_steps": global_step,
                "training_time_mins": (time.time() - start_time) / 60,
                }, f)
        writer.close()
        TimerRegistry.export(save_folder + "timers.json")

        if args.hpc:  # Duplicate logs
            shutil.copy(f"rl_core/HPC/Out_{args.job_index}.out", save_folder + "Out.out")
            shutil.copy(f"rl_core/HPC/Err_{args.job_index}.err", save_folder + "Err.err")