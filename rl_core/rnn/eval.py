import torch
from tqdm import tqdm
import os

from .train import TrafficLightEnv, RNNPolicy, select_action

# Evaluate model over 100 episodes
env = TrafficLightEnv()
policy = RNNPolicy()
policy.load("rl_core/rnn/rnn_policy.pth")
policy.eval()

total_reward = 0
total_episodes = 200
steps = 0
for episode in tqdm(range(total_episodes), desc="Evaluating"):

    obs = env.reset()
    hidden = torch.zeros(1,1,16)
    while True:

        action, logprob, hidden = select_action(policy, obs, hidden)
        # action = env.action_space.sample()

        obs, reward, done = env.step(action)
        steps += 1
        total_reward += reward

        if done:
            break

print(total_reward / (total_episodes * TrafficLightEnv.total_steps), "average reward per step")
# RNN Results: 0.9548
# Random is .3333