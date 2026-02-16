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

state_request = StateRequest(grid=[[0]*10]*11, bike1=[0,0], bike2=[0,0])
response = requests.post("http://localhost:8000/act", json=state_request.model_dump())
print(response.status_code)
if response.status_code == 200:
    print(response.json())