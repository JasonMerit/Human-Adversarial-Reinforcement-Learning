import gymnasium as gym

env = gym.make("ALE/Pong-v5")  # Gymnasium-preferred ID
print(env.observation_space, env.action_space)
