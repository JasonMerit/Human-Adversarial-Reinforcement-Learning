# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import PoLEnv

a = np.int8(128)
print(a, a.dtype)