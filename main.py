# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os, time
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import PoLEnv

kek = np.random.randint(3, size=(6, 5, 2), dtype=np.int8)

print("Test... ", end="")
time.sleep(1)
print("[green]Pass[/green]")

# print(kek)
# actions = np.array([[0, 2], [1, 3], [2, 0], [3, 1], [1, 3], [0, 2]])
