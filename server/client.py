from pydantic import BaseModel
import requests
from .main import StateRequest, ActionResponse, Trajectory

# state_request = StateRequest(grid=[[0]*10]*11, bike1=[0,0], bike2=[0,0])
# response = requests.post("http://localhost:8000/act", json=state_request.model_dump())
# print(response.status_code)
# if response.status_code == 200:
#     print(response.json())

trajectory = Trajectory(
    actions=[(0, 1), (1, 0), (0, 1)],
    winner=1
)

response = requests.post("http://localhost:8000/trajectory", json=trajectory.model_dump())
print(response.status_code)