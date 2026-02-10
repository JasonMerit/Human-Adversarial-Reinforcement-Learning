import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from agents.dqn import QNet

from environment.env import TronEnv
from environment.wrappers import TronView, TronEgo, TronTorch, TronImage
from agents import DeterministicAgent, RandomAgent, HeuristicAgent, DQNAgent

q_net = QNet.load("q_net.pth")

env = TronEnv(DeterministicAgent(is_opponent=False), width=10, height=10)
# env = TronImage(env)
# env = TronEgo(env)
# env = TronTorch(env)
env = TronView(env, 10, 70)

state, _ = env.reset()
agent = DeterministicAgent(is_opponent=False)
# agent = HeuristicAgent()
# agent = DQNAgent("q_net.pth")
# agent = RandomAgent(env.action_space)
agent.bind_env(env)

while True:
    action = agent(state)
    state, reward, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()

        
