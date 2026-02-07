from environment.env import TronEnv
from environment.wrappers import TronView, TronEgo
from agents.deterministic import DeterministicAgent
from train import QNet
from eval import eval

agent = QNet.load("q_net.pth")
env = TronEnv(DeterministicAgent(), width=10, height=10)
env = TronEgo(env)

print(eval(env, agent))