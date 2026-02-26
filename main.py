from tron_env.env import TronEnv
from tron_env.wrappers import TronView, TronImage, TronEgo

env = TronEnv()
env = TronView(env)
env = TronImage(env)
env = TronEgo(env)

state, _ = env.reset()

episode = 0
total_reward = 0.0
while True:
    action = env.action_space.sample()  # Replace with your agent's action
    
    state, reward, done, _, info = env.step(action)

    if done:
        episode += 1
        total_reward += reward
        print(round(total_reward / episode, 2), end="\r")
        state, _ = env.reset()
