# From https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rainbow_atari.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
import random, os, time, shutil

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rich import print
import yaml

from .argp import read_args
from .network import Rainbow
from .utils import TimerRegistry
from rl_core.env import TronDuoEnv, TronView

def make_envs(indx, seed, render):
    def thunk():
        env = TronDuoEnv()
        if render and indx==0:
            env = TronView(env, fps=10000)
        env.action_space.seed(seed + indx)
        return env
    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = read_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Handle parallel envs
    total_loops = args.total_timesteps // args.num_envs
    burnin = args.learning_starts // args.num_envs
    target_every = args.target_network_frequency // args.num_envs
    train_count = args.num_envs // args.train_frequency
    log_every = max(1, total_loops // 100)
    save_every = max(1, total_loops // args.total_checkpoints)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=====| {args.exp_name} on {device}", "[yellow bold](debug mode)[/yellow bold]" if args.debug else "", "|=====")

    writer = None
    if args.track:
        i = 0
        save_folder = f"runs/{args.exp_name}"
        if device.type == "cuda":
            save_folder = os.path.join("rl_core/HPC/", save_folder)
        while os.path.exists(save_folder + f"_{i}"):
            i += 1
        save_folder += f"_{i}/"
        os.makedirs(save_folder)
        with open(save_folder + "args.yml", "w") as f:
            yaml.dump(vars(args), f)
        
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = SummaryWriter(save_folder)
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        if args.save:
            print(f"Models will be saved to {save_folder}")
        else:
            print("Models will NOT be saved!")


    # Envs and agents
    envs = gym.vector.SyncVectorEnv([make_envs(i, args.seed, args.render) for i in range(args.num_envs)])
    n_actions = envs.single_action_space.nvec[0]
    agent1 = Rainbow(n_actions, args, device, writer, "A")
    agent2 = Rainbow(n_actions, args, device, writer, "B")

    # Logging
    results = [0, 0, 0]
    total_episodes = 0
    total_episode_lengths = 0
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    kek = 0

    start_time = time.time()

    obs, _ = envs.reset()
    obs1, obs2 = obs[:, 0], obs[:, 1]
    for global_step in range(1, total_loops + 1):
        a1 = agent1.act(obs1)
        a2 = agent2.act(obs2)

        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * total_loops, global_step)
        explore_mask = np.random.rand(args.num_envs) < epsilon
        a1[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())
        explore_mask = np.random.rand(args.num_envs) < 1  # FULL EXPLORE MODE
        a2[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())

        actions = np.stack([a1, a2], axis=1) 
        next_obs, rewards, dones, _, infos = envs.step(actions)
        next_obs1, next_obs2 = next_obs[:, 0], next_obs[:, 1]

        agent1.rb.add(obs1, a1, rewards, next_obs1, dones)
        # agent2.rb.add(obs2, a2, -rewards, next_obs2, dones)

        obs1, obs2 = next_obs1, next_obs2
        episode_lengths += 1

        # Training
        # if global_step > burnin:
        for _ in range(train_count):
            agent1.learn()
            # agent2.learn()

        # update target network
        if global_step % target_every == 0:
            agent1.update_target()
            # agent2.update_target()
        
        # Logging
        for i in np.where(dones)[0]:  # Update results for any env that is done
            results[infos["result"][i]] += 1
            total_episode_lengths += episode_lengths[i]
            total_episodes += 1
            episode_lengths[i] = 0
            if writer:
                writer.add_scalar("charts/draw_percentage", results[0] / total_episodes, total_episodes)
                writer.add_scalar("charts/agent1_win_percentage", results[1] / total_episodes, total_episodes)
                writer.add_scalar("charts/agent2_win_percentage", results[2] / total_episodes, total_episodes)
                writer.add_scalar("charts/avg_episode_length", total_episode_lengths / total_episodes, total_episodes)

        if global_step % log_every == 0:
            kek += 1
            sps = int(global_step * args.num_envs / (time.time() - start_time))
            elapsed = time.time() - start_time
            progress = global_step / total_loops
            eta = elapsed * (1/progress - 1)
            # print(f"{progress*100:.1f}% - {epsilon=:.3f}")
            print(f"{progress*100:.1f}% - SPS: {sps} - Results: {results} {eta/60:.1f} minutes left...")
        
        # env_step = global_step * args.num_envs
        # if args.save and global_step % save_every == 0:
        #     agent1.save(save_folder + f"A_{env_step}.pth")
        #     agent2.save(save_folder + f"B_{env_step}.pth")

    envs.close()
    print(f"Total logs: {kek}")
    TimerRegistry.report()

    if args.track:
        with open(save_folder + "results.yml", "w") as f:
            yaml.dump({
                "results": results, 
                "steps_taken": global_step * args.num_envs, 
                "training_time_hours": (time.time() - start_time) / 3600,
                }, f)
        writer.close()
        TimerRegistry.export(save_folder + "timers.json")

        if device.type == "cuda":  # Duplicate logs
            shutil.copy("rl_core/HPC/Output.out", save_folder + "Output.out")
            shutil.copy("rl_core/HPC/Error.err", save_folder + "Error.err")
        
        if args.save:
            agent1.save(save_folder + f"A.pth")
            agent2.save(save_folder + f"B.pth")
