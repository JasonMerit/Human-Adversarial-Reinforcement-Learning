# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from typing import Optional

import yaml

from rl_core.env import TronDuoEnv, TronView
from rl_core.agents.buffers import RolloutBuffer
from rl_core.agents.ppo import PPOAgent

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

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000#10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # Jason
    save: bool = True
    """whether to save the final model"""
    num_checkpoints: int = 10
    """the number of checkpoints to save (computed in runtime)"""
    sync: bool = False
    """whether to use envs synchronously """
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(indx, seed):
    def thunk():
        env = TronDuoEnv()
        env.action_space.seed(seed + indx)
        return env
    return thunk

if __name__ == "__main__":
    print(f"Setting up {Args.exp_name} experiment...")
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)  # 5 * 128 = 640
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.seed is None:
        args.seed = np.random.randint(0, 1e6)
    save_every = args.num_iterations // args.num_checkpoints
    log_every = max(1, args.num_iterations // 100)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"===== Training with seed {args.seed} on device {device} =====")

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
        print(f"Saving models to {save_folder}...")
    else:
        print("Models will NOT be saved!")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    # env setup
    if args.sync:
        envs = gym.vector.SyncVectorEnv([make_env(i, args.seed) for i in range(args.num_envs)])
        print("Using synchronous environments (SyncVectorEnv)")
    else:
        envs = gym.vector.AsyncVectorEnv([make_env(i, args.seed) for i in range(args.num_envs)])
        print("Using asynchronous environments (AsyncVectorEnv)")
    n_actions = envs.single_action_space.nvec[0]  # Either is fine (symmetric environment)
    obs_shape = envs.single_observation_space.shape[-3:]  # Ignore the stacked observations

    buffer_obs_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
    buffer_args = {"buffer_size" : args.num_steps, "observation_space" : buffer_obs_space, "gae_lambda" : args.gae_lambda, "n_envs" : args.num_envs}
    agent1, agent2 = [PPOAgent(obs_shape, n_actions, RolloutBuffer(**buffer_args), device, args) for _ in range(2)]

    obs, _ = envs.reset(seed=args.seed)
    obs1, obs2 = obs[:, 0], obs[:, 1]
    episode_start = np.zeros(args.num_envs)

    global_step = 0
    start_time = time.time()
    
    try:
        for iteration in range(1, args.num_iterations + 1):
            for _ in range(args.num_steps):
                global_step += args.num_envs

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    a1, logprob1, _, value1 = agent1.get_action_and_value(torch.tensor(obs1).to(device))
                    a2, logprob2, _, value2 = agent2.get_action_and_value(torch.tensor(obs2).to(device))

                action = np.stack([a1.cpu().numpy(), a2.cpu().numpy()], axis=1)
                next_obs, reward, done, _, infos = envs.step(action)

                agent1.add(obs1, action[:, 0], reward, episode_start, value1, logprob1)
                agent2.add(obs2, action[:, 1], -reward, episode_start, value2, logprob2)

                obs1, obs2 = next_obs[:, 0], next_obs[:, 1]
                episode_start = done
            
            # Learn after collecting data for num_steps steps
            with torch.no_grad():
                last_values1 = agent1.get_value(torch.tensor(obs1).to(device)).reshape(1, -1)
                last_values2 = agent2.get_value(torch.tensor(obs2).to(device)).reshape(1, -1)
            agent1.learn(last_values1, done)
            agent2.learn(last_values2, done)


            # Saving
            if iteration % save_every == 0 and args.save:
                agent1.save(save_folder + f"A_{global_step}.pth")
                agent2.save(save_folder + f"B_{global_step}.pth")
            
            # Logging
            if iteration % log_every == 0:
                sps = int(global_step / (time.time() - start_time))
                elapsed = time.time() - start_time
                progress = iteration / args.num_iterations
                eta = elapsed * (1/progress - 1)
                print(f"{progress*100:.1f}% - SPS: {sps} - {eta/60:.1f} minutes left")

    finally:
        if args.save:
            with open(save_folder + "results.yml", "w") as f:
                yaml.dump({
                    "steps_taken": global_step, 
                    "training_time_hours": (time.time() - start_time) / 3600,
                    }, f)
            agent1.save(save_folder + f"A_{global_step}.pth", verbose=True)
            agent2.save(save_folder + f"B_{global_step}.pth", verbose=True)

    envs.close()