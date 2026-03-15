
import gymnasium as gym
import torch

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
    n_envs = 2
    envs = gym.vector.AsyncVectorEnv([make_env(i, i, render=False) for i in range(n_envs)])
    action_space = envs.single_action_space.spaces[0]
    obs_space = envs.single_observation_space.spaces[0]

    agent = Agent.from_checkpoint("runs\PPO.pth", obs_space.shape, action_space.n)
    agent.eval()

    rb0 = ReplayBuffer(10000, obs_space, action_space, n_envs=n_envs, device="cpu")
    rb1 = ReplayBuffer(10000, obs_space, action_space, n_envs=n_envs, device="cpu")
    obs, info = envs.reset()

    for _ in range(2000):
        obs0 = torch.tensor(obs[0], dtype=torch.float32)
        obs1 = torch.tensor(obs[1], dtype=torch.float32)

        with torch.no_grad():
            a0, a1 = agent(obs0), agent(obs1)

        joint_action = torch.stack([a0, a1], dim=1).cpu().numpy()
        next_obs, reward, done, _, info = envs.step(joint_action)

        r0, r1 = reward, -reward

        rb0.add(obs[0], next_obs[0], a0, r0, done, info)
        rb1.add(obs[1], next_obs[1], a1, r1, done, info)

        obs = next_obs

    # Sample a batch from the replay buffer
    sv = StateViewer(25, scale=20, fps=5)
    batch = rb0.sample(32)
    for obs in batch.observations:
        sv.view(obs.numpy())

