# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import PoLEnv

children = [None] * 4

if children: print("hi")