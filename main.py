# import gymnasium as gym
# gym.register(
#     id="TronEnv-v0",
#     entry_point="environment.tron_env:TronEnv",
# )
# from gymnasium.utils.env_checker import check_env
# env = gym.make("TronEnv-v0")
# check_env(env.unwrapped)

from environment.tron_env import TronEnv, TronView
from agents.deterministic import DeterministicAgent

seed = 523653
env = TronEnv(DeterministicAgent(1), size=10)
env = TronView(env, 10)
state, _ = env.reset()

agent = DeterministicAgent(3)

done = False
while True:
    action = agent.compute_single_action(state)
    state, reward, done, _, info = env.step(action)
    if done:
        env.reset()
        agent.reset()