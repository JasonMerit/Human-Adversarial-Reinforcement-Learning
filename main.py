# from tron_env.env import TronEnv
# from tron_env.wrappers import TronView

# env = TronEnv()
# env = TronView(env)

# state, _ = env.reset()

# episode = 0
# total_reward = 0.0
# while True:
#     action = env.action_space.sample()  # Replace with your agent's action
    
#     state, reward, done, _, info = env.step(action)

#     if done:
#         episode += 1
#         total_reward += reward
#         print(round(total_reward / episode, 2), end="\r")
#         state, _ = env.reset()

import gymnasium as gym
import tron_env  # REQUIRED — triggers __init__ registration

print("Registered envs:")
print([env_id for env_id in gym.envs.registry.keys() if "Tron" in env_id])

env = gym.make("Tron-v0")

obs, info = env.reset()
print("Reset OK")

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("Step OK")
