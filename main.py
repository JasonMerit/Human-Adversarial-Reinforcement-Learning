# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
import os
from pathlib import Path
import yaml
import numpy as np
from rich import print
import gymnasium as gym

from rl_core.env import PoLEnv

# NUM_ENVS = 3
# SIZE = 4
# SEED = 4222

# np.random.seed(SEED)


# def make_env(idx):
#     def thunk():
#         env = PoLEnv(SIZE)
#         env.action_space.seed(idx)
#         return env
#     return thunk

# envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])
# obs, _ = envs.reset()

# for _ in range(3):
#     actions = np.random.randint(1, 3, size=NUM_ENVS)
#     obs, rewards, terminateds, truncateds, infos = envs.step(actions)

# # print(obs)

# coords = np.where(obs[:, 1] == 1)
# print(coords)
# pos = np.stack([coords[1], coords[2]], axis=1).tolist()
# print(pos)
# pos = np.where(obs[:, 1] == 1)
# pos = list(zip(coords[1], coords[2]))
# print(pos)
# quit()
# # print(envs.get_attr("walls"))
# walls = tuple(obs[:, 0].astype(np.int8))
# # walls = tuple(obs[:, 0])
# # print(walls)

# # pos = np.asarray(np.where(obs[:, 1, :] == 1), dtype=np.int8).squeeze()
# # print(pos)
# # envs.call("set_state", obs[0])
# # print("=========")
# # obs = envs.call("_get_state")
# # print(obs)

# # pos = envs.get_attr("pos")
# # walls = envs.get_attr("walls")
# # print(pos)



# # envs.set_attr("pos", pos[0])
# # envs.set_attr("pos", pos[0])
# envs.set_attr("walls", walls)

# envs.set_attr("pos", pos)



# # print(envs.get_attr("walls"))
# obs = envs.call("_get_state")
# print(obs)


for i in range(10):
    pass
else:
    print("break")