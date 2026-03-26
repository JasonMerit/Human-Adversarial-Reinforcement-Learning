# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
import yaml
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cleanrl_utils.buffers import ReplayBuffer

from rl_core.agents.dqn import DQNAgent
from rl_core.env import TronView, TronDuoEnv, utils


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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    render: bool = False
    """whether to render the environment during training (slows down training!)"""

    # Algorithm specific arguments
    total_timesteps: int = 1000#1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
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
    train_frequency: int = 10
    """the frequency of training"""


def make_env(seed, idx, render=False):
    def thunk():
        env = TronDuoEnv()
        if render and idx == 0:
            env = TronView(env)

        env.action_space.seed(seed)
        return env

    return thunk
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.seed is None:
        args.seed = np.random.randint(0, 1e6)
    run_name = f"{args.exp_name}_{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv([make_env(args.seed + i, i, render=args.render) for i in range(args.num_envs)])

    action_space = envs.single_action_space
    obs_space = envs.single_observation_space

    buffer_obs_space = gym.spaces.Box(low=0, high=1, shape=(3, 25, 25), dtype=np.float32)
    rb0 = ReplayBuffer(args.buffer_size, buffer_obs_space, device=device, n_envs=args.num_envs)
    rb1 = ReplayBuffer(args.buffer_size, buffer_obs_space, device=device, n_envs=args.num_envs)
    human = DQNAgent(obs_shape=obs_space.shape[-3:], n_actions=action_space.nvec[0], lr=args.learning_rate, rb=rb0, batch_size=args.batch_size, gamma=args.gamma, device=device)
    adversary = DQNAgent(obs_shape=obs_space.shape[-3:], n_actions=action_space.nvec[1], lr=args.learning_rate, rb=rb1, batch_size=args.batch_size, gamma=args.gamma, device=device)

    print(f"===== Training with seed {args.seed} on device {device} =====")
    if not args.save_model:
        print(utils.red("Models will NOT be saved!"))
    start_time = time.time()

    # Jason
    episode_count = 0
    log_interval = 100
    eval_interval = next_eval_episode = 1000

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    # Handling multiple parallel envs steps
    learn_every = max(1, args.train_frequency // args.num_envs)
    target_every = max(1, args.target_network_frequency // args.num_envs)
    learn_start_loop = args.learning_starts // args.num_envs

    # Logging
    results = [0, 0, 0]

    try:
        total_loops = args.total_timesteps // args.num_envs
        pbar = tqdm(range(total_loops), desc="Training", miniters=log_interval)
        for global_step in pbar:
            env_step = global_step * args.num_envs
            obs0, obs1 = obs[:, 0], obs[:, 1]

            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, env_step)
            explore_mask = np.random.rand(args.num_envs) < epsilon

            a0 = human.select_action(torch.tensor(obs0, dtype=torch.float32, device=device))
            a1 = adversary.select_action(torch.tensor(obs1, dtype=torch.float32, device=device))

            a0[explore_mask] = np.random.randint(0, action_space.nvec[0], size=explore_mask.sum())
            a1[explore_mask] = np.random.randint(0, action_space.nvec[1], size=explore_mask.sum())

            # TRY NOT TO MODIFY: execute the game and log data.
            actions = np.stack([a0, a1], axis=1)
            next_obs, rewards, terminations, _, infos = envs.step(actions)

            # Iterate over terminations to log episode results
            for i in range(args.num_envs):
                if terminations[i]:
                    results[infos["final_info"][i]['result']] += 1
                    sps = int(env_step / (time.time() - start_time))
                    pbar.set_postfix({"Results": results, "SPS": sps})

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             episode_count += 1
            #             # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #             # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            #             results[info["result"]] += 1
            #             print(results[info["result"]])                        



            r0, r1 = rewards, -rewards
            human.add_to_buffer(obs0, next_obs[:, 0], a0, r0, terminations, infos)
            adversary.add_to_buffer(obs1, next_obs[:, 1], a1, r1, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > learn_start_loop:
                if global_step % learn_every == 0:
                    human.learn()
                    adversary.learn()

                # update target network
                if global_step % target_every == 0:
                    human.update_target_network()
                    adversary.update_target_network()
            
            # if global_step % log_interval == 0:
            #     sps = int(env_step / (time.time() - start_time))
            #     pbar.set_postfix({"SPS": sps})
                
    finally:
        if args.save_model:
            folder = "runs/self_train"
            i = 0
            while os.path.exists(folder + f"_{i}"):
                i += 1
            folder += f"_{i}/"
            os.makedirs(folder)
            
            with open(folder + "args.yml", "w") as f:
                yaml.dump(vars(args), f)
            with open(folder + "results.yml", "w") as f:
                yaml.dump({"results": results, "steps_taken": env_step}, f)
            human.save(folder + "human.pth")
            adversary.save(folder + "adversary.pth")

        # from rl_core.tron_env.tron_env.utils import StateViewer
        # sv = StateViewer(25, scale=20, fps=5)
        # batch = rb.sample(32)
        # for obs in batch.observations:
        #     sv.view(obs.cpu().numpy())
        envs.close()
        # writer.close()
