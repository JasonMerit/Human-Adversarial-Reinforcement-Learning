import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from agents.dqn import QNet

from environment.env import TronEnv
from environment.wrappers import TronView, TronEgo
from agents.deterministic import DeterministicAgent
from agents.mcts import HeuristicAgent
from utils.heuristics import chamber_heuristic

q_net = QNet.load("q_net.pth")

env = TronEnv(DeterministicAgent(start_left=True), width=10, height=10)
env = TronView(TronEgo(env), 10, 70)
tron = env.unwrapped.tron

state, _ = env.reset()
# state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
agent = DeterministicAgent(start_left=False)
# agent = HeuristicAgent()

while True:
    # with torch.no_grad():
    #     q_values = q_net(state)
    #     action = q_values.argmax().item()
    
    action = env.action_space.sample()
    # action = agent(state)

    state, reward, done, _, _ = env.step(action)

    if done:
        state, _ = env.reset()

    chamber = chamber_heuristic(tron.walls, tron.bike1.pos, tron.bike2.pos)
    print(chamber)
        
