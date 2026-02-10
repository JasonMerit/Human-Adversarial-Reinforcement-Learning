import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from agents.dqn import QNet

from environment.env import TronEnv
from environment.wrappers import TronView, TronEgo, TronTorch, TronImage
from agents import DeterministicAgent, RandomAgent, HeuristicAgent, DQNAgent

q_net = QNet.load("q_net.pth")

opp = RandomAgent()
env = TronEnv(opp, width=10, height=10)
env = TronEgo(env)
opp.bind_env(env)
# env = TronImage(env)
# env = TronTorch(env)
env = TronView(env, 10, 70)

state, _ = env.reset()
# agent = DeterministicAgent()
# agent = HeuristicAgent()
# agent = DQNAgent("q_net.pth")
agent = RandomAgent()
agent.bind_env(env)

while True:
    action = agent(state)
    state, reward, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()

        
