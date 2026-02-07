import torch
from train import QNet

from environment.env import TronEnv
from environment.wrappers import TronView, TronEgo
from agents.deterministic import DeterministicAgent

q_net = QNet()
q_net.load_state_dict(torch.load("q_net.pth"))

env = TronEnv(DeterministicAgent(start_left=True), width=10, height=10)
env = TronView(TronEgo(env), 10, 70)

state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

while True:
    with torch.no_grad():
        q_values = q_net(state)
        action = q_values.argmax().item()

    next_state, reward, done, _, _ = env.step(action)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    state = next_state_tensor

    if done:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
