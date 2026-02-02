# import gymnasium as gym
# gym.register(
#     id="TronEnv-v0",
#     entry_point="environment.tron_env:TronEnv",
# )
# from gymnasium.utils.env_checker import check_env
# env = gym.make("TronEnv-v0")
# check_env(env.unwrapped)

from environment.env import TronEnv
from environment.wrappers import TronView, TronEgo
from agents.deterministic import DeterministicAgent

seed = 523653
env = TronEnv(DeterministicAgent(1), width=10, height=10)
env = TronView(env, fps=10, scale=70)
env = TronEgo(env)
state, _ = env.reset()

agent = DeterministicAgent(1)

done = False
while True:
    TronView.view(state, scale=70)
    action = TronView.wait_for_keypress()
    # action = agent.compute_single_action(state)
    # action = 1 
    state, reward, done, _, info = env.step(action)
    if done:
        env.reset()
        agent.reset()