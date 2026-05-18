import os, time, random
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import TronDuoEnv, TronView

opp_props = np.ones(3) / 3
opp_policy = lambda x: opp_props

for _ in range(2):
    print(opp_policy(2))