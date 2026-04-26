# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import PoLEnv

seed = 42
num_envs = 4
np.random.seed(seed)
for _ in range(5):
    actions = np.random.randint(0, 4, num_envs)
    print(actions)