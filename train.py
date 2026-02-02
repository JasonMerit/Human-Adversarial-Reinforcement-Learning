import torch
import torch.nn as nn
from torch.distributions import Categorical

class TronPPONet(nn.Module):
    def __init__(self, obs_shape=(3, 10, 10), n_actions=4):
        super().__init__()
        c, h, w = obs_shape
        assert h == 10 and w == 10

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU(),
        )
        self.act_head = nn.Linear(128, n_actions)
        self.v_head = nn.Linear(128, 1)

    def forward(self, obs, state=None, info=None):
        # obs: Batch of shape (batch, C, H, W) as float32
        x = self.conv(obs)
        x = self.fc(x)
        logits = self.act_head(x)
        value = self.v_head(x).squeeze(-1)
        return logits, value


import tianshou as ts

def make_policy(obs_shape=(3, 10, 10), n_actions=4, device="cpu"):
    net = TronPPONet(obs_shape, n_actions).to(device)
    # Split into actor / critic views for PPOPolicy (can share parameters)
    actor = net
    critic = net

    optim = torch.optim.Adam(net.parameters(), lr=3e-4)

    def dist_fn(logits):
        return Categorical(logits=logits)

    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        eps_clip=0.2,
        value_clip=True,
        vf_coef=0.5,
        ent_coef=0.01,
        reward_normalization=True,
        action_scaling=False,
        action_space=None,  # discrete
    )
    return policy


import gymnasium as gym
import numpy as np
from environment.env import TronEnv
from agents.deterministic import DeterministicAgent

def make_env():
    return TronEnv(DeterministicAgent(), width=10, height=10)

train_num = 8
test_num = 4

train_envs = ts.env.SubprocVectorEnv([make_env for _ in range(train_num)])
test_envs = ts.env.SubprocVectorEnv([make_env for _ in range(test_num)])

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = make_policy(obs_shape=(3, 10, 10), n_actions=4, device=device)

train_collector = ts.data.Collector(policy, train_envs)
test_collector = ts.data.Collector(policy, test_envs)



from tianshou.trainer import OnpolicyTrainer

max_epoch = 100
step_per_epoch = 10000        # total env steps per epoch
repeat_per_collect = 4        # PPO epochs per batch
episode_per_collect = 16      # episodes collected before each update
batch_size = 2048             # minibatch size inside PPO

def stop_fn(mean_rewards):
    # adjust threshold as you like
    return mean_rewards >= 0.9 * 50  # e.g., close to max 50

trainer = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=max_epoch,
    step_per_epoch=step_per_epoch,
    repeat_per_collect=repeat_per_collect,
    episode_per_test=20,
    batch_size=batch_size,
    episode_per_collect=episode_per_collect,
    stop_fn=stop_fn,
)

for epoch_stat in trainer:
    print(f"Epoch {epoch_stat.epoch}, "
          f"reward: {epoch_stat.best_reward}, "
          f"step: {epoch_stat.env_step}")




# After training
env = make_env()
obs = env.reset()
done = False

while not done:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = policy.actor(obs_tensor)
    action = torch.argmax(logits, dim=-1).item()
    obs, reward, done, info = env.step(action)
    env.render()


    
