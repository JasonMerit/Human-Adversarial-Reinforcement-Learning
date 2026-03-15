
import gymnasium as gym
import torch
import numpy as np

from rl_core.cleanrl.cleanrl_utils.buffers import ReplayBuffer

from rl_core.tron_env.tron_env import TronView, TronDuoEnv
from rl_core.tron_env.tron_env.utils import StateViewer
from rl_core.cleanrl.cleanrl.ppo import Agent

def make_env(seed, idx, render=False):
    def thunk():
        env = TronDuoEnv()
        if render and idx == 0:
            env = TronView(env)

        env.action_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    n_envs = 5
    envs = gym.vector.AsyncVectorEnv([make_env(i, i, render=False) for i in range(n_envs)])
    action_space = envs.single_action_space
    obs_space = envs.single_observation_space

    agent = Agent.from_checkpoint("runs\PPO.pth", obs_space.shape[-3:], action_space.nvec[0])
    agent.eval()

    buffer_obs_space = gym.spaces.Box(low=0, high=1, shape=(3, 25, 25), dtype=np.float32)
    rb0 = ReplayBuffer(10000, buffer_obs_space, n_envs=n_envs, device="cpu")
    rb1 = ReplayBuffer(10000, buffer_obs_space, n_envs=n_envs, device="cpu")
    obs, info = envs.reset()

    print("START SELF PLAY")
    for _ in range(20):
        obs0, obs1 = obs[:, 0], obs[:, 1]

        with torch.no_grad():
            a0 = agent(torch.tensor(obs0, dtype=torch.float32))  # shape: (n_envs,)
            a1 = agent(torch.tensor(obs1, dtype=torch.float32))  # shape: (n_envs,)

        actions = np.stack([a0.cpu().numpy(), a1.cpu().numpy()], axis=1)
        next_obs, rewards, done, _, info = envs.step(actions)

        r0, r1 = rewards, -rewards

        # print("REWARD", r0, r1)
        rb0.add(obs0, next_obs[:, 0], a0, r0, done, info)
        rb1.add(obs1, next_obs[:, 1], a1, r1, done, info)

        obs = next_obs

    # Sample a batch from the replay buffer
    sv = StateViewer(25, scale=20, fps=5)
    batch = rb0.sample(32)
    for obs in batch.observations:
        sv.view(obs.numpy())

