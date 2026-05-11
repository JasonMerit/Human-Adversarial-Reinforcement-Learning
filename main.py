# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os, time, random
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import TronDuoEnv, TronView

actions = np.array([
    [1, 0],
    [2, 0],
    [0, 0],
    [0, 2],
    [1, 0],
    [2, 1],
    [0, 1],
    [0, 0],
    [1, 1],
    [1, 2],
    [0, 0],
    [2, 1],
    [2, 2],
    [0, 0],
    [2, 1],
    [1, 1],
    [1, 0],
    [1, 2],
    [1, 0],
    [0, 0],
    [1, 1],
    [0, 2],
    [1, 1],
    [1, 2],
    [1, 1],
    [2, 2],
])

env = TronDuoEnv()
env = TronView(TronDuoEnv())
env.reset()
history = []
steps = 0
episodes=0
while True:
    # action = env.action_space.sample()
    action = actions[steps % len(actions)] if episodes % 2 else 2 - actions[steps % len(actions)]
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    history.append(action)
    if terminated or truncated:
        env.reset()
        history = []
        episodes += 1





# print(kek)
# actions = np.array([[0, 2], [1, 3], [2, 0], [3, 1], [1, 3], [0, 2]])
