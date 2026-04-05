import os
import random
import time
import yaml
from dataclasses import dataclass
from typing import Optional
import copy

import gymnasium as gym
import numpy as np
import torch
import tyro

from rl_core.agents.buffers import ReplayBuffer

from rl_core.agents.dqn import DQNAgent, QNetwork
from rl_core.env import TronView, TronDuoEnv, Tron2ChannelEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: Optional[int] = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = True
    """whether to save model into the `runs/{exp_name}` folder"""
    render: bool = False
    """whether to render the environment during training (slows down training!)"""
    total_checkpoints: int = 10
    """the total number of checkpoints to save during training"""
    environment: str = "TronDuo"

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
    # train_frequency: int = 10
    # """the frequency of training"""
    pooling_frequency: int = 100_000
    """the frequency of adding current agent to the pool and updating opponent"""


def make_env(seed, idx, environment, render=False):
    Env = TronDuoEnv if environment == "TronDuo" else Tron2ChannelEnv
    def thunk():
        env = Env()
        if render and idx == 0:
            env = TronView(env)

        env.action_space.seed(seed)
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv([make_env(args.seed + i, i, args.environment, render=args.render) for i in range(args.num_envs)])

    n_actions = envs.single_action_space.nvec[0]  # Either is fine (symmetric environment)
    obs_shape = envs.single_observation_space.shape[-3:]  # Ignore the stacked observations

    buffer_obs_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
    rb = ReplayBuffer(args.buffer_size, buffer_obs_space, device=device, n_envs=args.num_envs)
    current_agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions, lr=args.learning_rate, rb=rb, batch_size=args.batch_size, device=device)

    print(f"===== Training with seed {args.seed} on device {device} =====")
    if not args.save_model:
        print("Models will NOT be saved!")

    obs, _ = envs.reset(seed=args.seed)

    # Handling multiple parallel envs steps
    learn_every = max(1, args.batch_size // args.num_envs)  # 256 / 64 = 4 steps
    target_every = max(1, args.target_network_frequency // args.num_envs)
    learn_start_loop = args.learning_starts // args.num_envs
    save_every = max(1, (args.total_timesteps // args.num_envs) // args.total_checkpoints)
    pool_every = max(1, (args.pooling_frequency // args.num_envs))  # Add to pool twice as often as saving
    log_interval = 5000


    # Logging and saving model
    if args.save_model:
        save_folder = "runs/" + args.exp_name
        i = 0
        while os.path.exists(save_folder + f"_{i}"):
            i += 1
        save_folder += f"_{i}/"
        os.makedirs(save_folder)
        with open(save_folder + "args.yml", "w") as f:
                yaml.dump(vars(args), f)


    # Initialize pool
    eta = 0.01
    past_agents = []       # list of past policies
    past_qualities = []    # list of qi
    opponent : QNetwork = copy.deepcopy(current_agent.q_network)  # start by playing against self
    opponent_idx = None
    pool_size_limit = 50

    def sample_opponent():
        if len(past_agents) == 0:
            return current_agent, None
        # Softmax over qualities
        probs = np.exp(np.array(past_qualities) - np.max(past_qualities))  # for numerical stability
        probs /= probs.sum()
        idx = np.random.choice(len(past_agents), p=probs)
        return past_agents[idx], idx
    
    sps = 0 # for final logging
    start_time = time.time()

    try:
        total_loops = args.total_timesteps // args.num_envs
        # pbar = tqdm(range(total_loops), desc="Training", miniters=log_interval)
        # for global_step in pbar:
        for global_step in range(1, total_loops+1):
            env_step = global_step * args.num_envs
            obs0, obs1 = obs[:, 0], obs[:, 1]

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, env_step)

            a0 = current_agent.select_action(torch.tensor(obs0, dtype=torch.float32, device=device))
            a1 = opponent.opponent_act(torch.tensor(obs1, dtype=torch.float32, device=device))

            explore_mask = np.random.rand(args.num_envs) < epsilon
            a0[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())
            explore_mask = np.random.rand(args.num_envs) < .2  # Opponent constantly seeking random action 20% of the time
            a1[explore_mask] = np.random.randint(0, n_actions, size=explore_mask.sum())

            actions = np.stack([a0, a1], axis=1)
            next_obs, rewards, terminations, _, infos = envs.step(actions)

            # Iterate over terminations to log episode results
            for i in range(args.num_envs):
                if terminations[i]:
                    if opponent_idx is not None and infos["final_info"][i]['result'] == 1:  # current_agent wins => Punish opponent sampling
                        qi = past_qualities[opponent_idx]
                        pi = np.exp(qi) / np.sum(np.exp(np.array(past_qualities)))
                        past_qualities[opponent_idx] = qi - eta / (len(past_agents) * pi)

            current_agent.add_to_buffer(obs0, next_obs[:, 0], a0, rewards, terminations, infos)
            obs = next_obs

            # Training.
            if global_step > learn_start_loop:
                if global_step % learn_every == 0:
                    current_agent.learn()

                # Update target network 
                if global_step % target_every == 0:
                    current_agent.update_target_network()

                # Pooling
                if global_step % pool_every == 0:
                    # Add current agent to pool
                    past_agents.append(copy.deepcopy(current_agent.q_network))
                    past_qualities.append(max(past_qualities) if len(past_qualities) > 0 else 1.0)
                    if len(past_agents) > pool_size_limit:
                        idx = np.argmin(past_qualities)
                        past_agents.pop(idx)
                        past_qualities.pop(idx)

                    # Decide opponent
                    if np.random.rand() < 0.8:
                        opponent, opponent_idx = copy.deepcopy(current_agent.q_network), None
                    else:
                        opponent, opponent_idx = sample_opponent()

                    
            
            # Saving
            if global_step % save_every == 0 and args.save_model:
                current_agent.save(save_folder + f"agent_{env_step}.pth")
            
            # Logging
            # if global_step % log_interval == 0:
                # pbar.set_postfix({"Results": results, "SPS": sps})
            if global_step % log_interval == 0:
                sps = int(env_step / (time.time() - start_time))
                elapsed = time.time() - start_time
                progress = global_step / total_loops
                eta = elapsed * (1/progress - 1)
                print(f"{progress*100:.1f}% - SPS: {sps} - {eta/60:.1f} minutes left")

    finally:
        if args.save_model:
            with open(save_folder + "results.yml", "w") as f:
                yaml.dump({
                    "steps_taken": env_step, 
                    "training_time_hours": (time.time() - start_time) / 3600,
                    }, f)
            current_agent.save(save_folder + f"agent_{env_step}.pth", verbose=True)

        envs.close()
        print(f"Training completed after {env_step} steps and {(time.time() - start_time) / 3600:.2f} hours! Final SPS: {sps}")
        # writer.close()
