# import gymnasium as gym
# gym.register(
#     id="TronEnv-v0",
#     entry_point="environment.tron_env:TronEnv",
# )
# from gymnasium.utils.env_checker import check_env
# env = gym.make("TronEnv-v0")
# check_env(env.unwrapped)

from environment.tron_env import TronEnv, TronView
env = TronEnv(size=10)
env = TronView(env, 1)
env.reset()

done = False
while True:
    action = 0
    state, reward, done, _, info = env.step(action)
    if done:
        env.reset()