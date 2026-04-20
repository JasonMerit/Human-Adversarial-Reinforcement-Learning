# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
import yaml
from dataclasses import dataclass
from typing import Optional

from rich import print

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

from rl_core.agents.buffers import ReplayBuffer

from rl_core.agents.dqn import DQNAgent
from rl_core.clean_rainbow.network import Rainbow
from rl_core.env import TronView, TronDuoEnv, Tron2ChannelEnv, PoLEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: Optional[int] = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    save: bool = True
    """whether to save model into the `runs/{exp_name}` folder"""
    render: bool = False
    """whether to render the environment during training (slows down training!)"""
    total_checkpoints: int = 1
    """the total number of checkpoints to save during training"""
    environment: str = "TronDuo"
    """the environment to train on (TronDuo or Tron2Channel)"""
    debug: bool = False
    """whether to run in debug mode (no saving, more logging, etc.)"""

    # Algorithm specific arguments
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    buffer_size: int = 500_000
    """the replay memory buffer size"""
    target_network_frequency: int = 1000#500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""

    # Jason specific arguments
    pol: bool = False
    """Whether to use the proof of learning environment"""
    size: int = 25
    """the size of the PoL environment (size x size grid)"""
    rain: bool = False
    """Whether to train a Rainbow agent instead of DQN"""
    gamma: float = 0.99
    """the discount factor gamma"""
    noisy: bool = False
    """whether to use noisy linear layers"""
    c51: bool = False
    """whether to use the C51 distributional RL algorithm"""
    # Prioritized replay buffer
    per: bool = False
    """whether to use a prioritized experience replay buffer"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay buffer"""
    prioritized_replay_beta: float = 0.4
    """beta parameter for prioritized replay buffer"""
    prioritized_replay_eps: float = 1e-6
    """epsilon parameter for prioritized replay buffer"""
    # train_frequency: int = 10
    # """the frequency of training"""


def make_env(idx, args):
    def thunk():
        Env = TronDuoEnv if not args.pol else PoLEnv
        env = Env(args.size)
        if args.render and idx == 0:
            env = TronView(env, fps=100000)
        env.action_space.seed(args.seed + idx)
        return env
    return thunk
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    print(f"Setting up {Args.exp_name} experiment...")
    args = tyro.cli(Args)
    if args.seed is None:
        args.seed = np.random.randint(0, 1e6)
    if args.debug:
        args.num_envs = 5
        args.save = False

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Logging and saving model
    if args.save:
        save_folder = "runs/" + args.exp_name
        i = 0
        while os.path.exists(save_folder + f"_{i}"):
            i += 1
        save_folder += f"_{i}/"
        os.makedirs(save_folder)
        with open(save_folder + "args.yml", "w") as f:
                yaml.dump(vars(args), f)
        print(f"Models will be saved to {save_folder}")
    else:
        print("[bold yellow] Models will NOT be saved!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=====|  Training with seed {args.seed} on device {device}", "[yellow bold](debug mode)[/yellow bold]" if args.debug else "", "|=====")

    # Handling multiple parallel envs steps
    learn_every = max(1, args.batch_size // args.num_envs)  # 256 / 64 = 4 steps
    target_every = max(1, args.target_network_frequency // args.num_envs)
    learn_start_loop = args.learning_starts // args.num_envs

    total_loops = args.total_timesteps // args.num_envs
    save_every = max(1, total_loops // args.total_checkpoints)
    log_every = max(1, total_loops // 100)
    eval_every = learn_every * 10  # Evaluate every 10 learning steps
    
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(i, args) for i in range(args.num_envs)])
    n_actions = envs.single_action_space.n if args.pol else envs.single_action_space.nvec[0]  # Either is fine (symmetric environment) 
    obs_shape = envs.single_observation_space.shape[-3:]  # Ignore the stacked observations
    print(f"Observation shape: {obs_shape}, Action space: {n_actions}")

    buffer_obs_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
    buffer_args = {"buffer_size": args.buffer_size, "observation_space": buffer_obs_space, "device": device, "n_envs": args.num_envs}
    agent_args = {"obs_shape": obs_shape, "n_actions": n_actions, "lr": args.learning_rate, "rb": ReplayBuffer(**buffer_args), "batch_size": args.batch_size, "device": device}
    agent1 = DQNAgent(**agent_args) if not args.rain else Rainbow(obs_shape, n_actions, args, device, None, "A")
    # agent1 = DQNAgent(**agent_args)
    # agent2 = DQNAgent(**agent_args)
    
    # PoL Specific
    env_eval = PoLEnv(args.size) if args.pol else None
    shortest_path = float('inf')
    win_combo = 0
    from rl_core.eval.pol_eval import eval

    # Logging
    results = [0, 0, 0]
    start_time = time.time()
    total_episodes = 0
    total_episode_lengths = 0
    episode_lengths = np.zeros(args.num_envs, dtype=int)

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(1, total_loops+1):
        env_step = global_step * args.num_envs
        # obs0, obs1 = obs[:, 0], obs[:, 1]

        a0 = agent1.select_action(obs)
        # a0 = agent1.select_action(torch.tensor(obs0, dtype=torch.float32, device=device))
        # a1 = agent2.select_action(torch.tensor(obs1, dtype=torch.float32, device=device))
        
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, env_step)
        explore_mask = np.random.rand(args.num_envs) < epsilon
        a0[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())
        # explore_mask = np.random.rand(args.num_envs) < epsilon
        # a1[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())

        actions = a0
        # actions = np.stack([a0, a1], axis=1)
        next_obs, rewards, terminations, _, infos = envs.step(actions)


        r0, r1 = rewards, -rewards
        agent1.rb.add(obs, a0, r0, next_obs, terminations, infos)
        # agent1.add_to_buffer(obs, next_obs[:, 0], a0, r0, terminations, infos)
        # agent2.add_to_buffer(obs1, next_obs[:, 1], a1, r1, terminations, infos)

        obs = next_obs
        episode_lengths += 1

        # Training.
        if global_step > learn_start_loop:
            if global_step % learn_every == 0:
                agent1.learn()
                # agent2.learn()

            # update target network
            if global_step % target_every == 0:
                agent1.update_target()
                # agent2.update_target()
            
            if global_step % eval_every == 0:
                eval_result = eval(agent1.q_network, env_eval, device)

                shortest_path = min(shortest_path, eval_result)
                if eval_result == env_eval.size * 2 - 2:  # Shortest path in an empty grid is size*2 - 2
                    win_combo += 1
                    if win_combo == 10:  # If the agent has solved the environment 10 times in a row
                        print("[bold green] Agent has consistently solved the environment!")
                        break
                else:
                    win_combo = 0
        
        for i in np.where(terminations)[0]:  # Update results for any env that is done
            # results[infos["final_info"][i]['result']] += 1
            total_episode_lengths += episode_lengths[i]
            total_episodes += 1
            episode_lengths[i] = 0
        # Saving
        # if args.save and global_step % save_every == 0:
        #     agent1.save(save_folder + f"A_{env_step}.pth")
        #     agent2.save(save_folder + f"B_{env_step}.pth")
        
        # Logging
        # if global_step % log_every == 0:
            # pbar.set_postfix({"Results": results, "SPS": sps})
        if global_step % log_every == 0:
            sps = int(env_step / (time.time() - start_time))
            elapsed = time.time() - start_time
            progress = global_step / total_loops
            eta = elapsed * (1/progress - 1)
            epi_len = total_episode_lengths / total_episodes if total_episodes > 0 else 0
            print(f"{progress*100:.1f}% - SPS: {sps} - epi_len: {epi_len:.2f} - eval_len {shortest_path} (x{win_combo}) - {eta/60:.1f} minutes left...")
            # print(f"{progress*100:.1f}% - SPS: {sps} - epi_len: {epi_len:.2f} - {eta/60:.1f} minutes left...")
            # print(f"{progress*100:.1f}% - SPS: {sps}")
            # print(f"{eta/60:.1f} minutes left...", end='\r')
        
        # if args.debug:
        #     elapsed = int(time.time() - start_time)
        #     pbar.update(elapsed - pbar.n)  
        #     if elapsed >= total_time:  
        #         print(f"\nStopping training after {total_time} seconds for testing purposes.")
        #         break

    if args.save:
        with open(save_folder + "results.yml", "w") as f:
            yaml.dump({
                "results": results, 
                "steps_taken": env_step, 
                "global_steps" : global_step,
                "training_time_hours": (time.time() - start_time) / 3600,
                }, f)
        agent1.save(save_folder + "A.pth", verbose=True)
        # agent2.save(save_folder + "B.pth", verbose=True)

    envs.close()
    print(f"Training completed after {env_step} steps and {(time.time() - start_time) / 3600:.2f} hours!")
    # writer.close()
