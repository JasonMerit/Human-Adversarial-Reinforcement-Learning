import yaml
from environment.wrappers import TronView
from agents import DeterministicAgent, RandomAgent, SemiDeterministicAgent, HeuristicAgent
from environment.env import TronDualEnv
                    

with open("rl_core/config.yml", "r") as f:
    config = yaml.safe_load(f)
size = tuple(config.get("grid"))

env = TronDualEnv(size)
env = TronView(env, fps=10, scale=70)

agent = DeterministicAgent()
# agent = HeuristicAgent()
# agent = RandomAgent()
agent.bind_env(env)
state, _ = env.reset()

done = False
total_reward = 0.0
episodes = 1

from pydantic import BaseModel
import requests

# Request schema
class StateRequest(BaseModel):
    grid: list[list[int]]
    bike1: list[int]
    bike2: list[int]

# Response schema
class ActionResponse(BaseModel):
    action: int

# session = requests.Session()  # keeps TCP connection alive
# def make_request(state):
#     payload = {
#         "grid": state[0].tolist(),
#         "bike1": state[1].tolist(),
#         "bike2": state[2].tolist()
#     }
#     response = session.post("http://127.0.0.1:8000/act", json=payload)
#     return response.json()["action"]


def make_request(state):
    state_request = StateRequest(grid=state[0].tolist(), bike1=state[1].tolist(), bike2=state[2].tolist())
    response = requests.post("http://localhost:8000/act", json=state_request.model_dump())
    return response.json()['action']

# def make_request(state):
#     state_request = StateRequest(grid=state[0].tolist(), bike1=state[1].tolist(), bike2=state[2].tolist())
#     response = requests.post("http://localhost:8000/act", json=state_request.model_dump())
#     return response.json()['action']

while True:
    server_action = make_request(state[1])

    state, reward, done, _, info = env.step((agent(state[0]), server_action))
    if done:
        if reward > 0.9:
            total_reward += reward
        state, _ = env.reset()
        agent.reset()
        episodes += 1
