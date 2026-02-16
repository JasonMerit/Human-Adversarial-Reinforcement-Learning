import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from colorama import init
init()

from .agents import SemiDeterministicAgent

app = FastAPI(title="RL Agent API")
agent = SemiDeterministicAgent(.5)


# Request schema
class StateRequest(BaseModel):
    grid: list[list[int]]
    bike1: list[int]
    bike2: list[int]

# Response schema
class ActionResponse(BaseModel):
    action: int


# @app.post("/act", response_model=ActionResponse)
# async def act(request: StateRequest):
#     state = (np.array(request.grid), np.array(request.bike1), np.array(request.bike2))
#     action = agent(state)  
#     return ActionResponse(action=action)

# @app.post("/act", response_model=ActionResponse)
# async def act(request: FlatStateRequest):
#     state = request.state
#     grid = np.array(state[:width*height], dtype=np.int8).reshape(height, width)
#     bike1 = np.array(state[width*height:width*height+2], dtype=np.int8)
#     bike2 = np.array(state[width*height+2:], dtype=np.int8)
#     action = agent((grid, bike1, bike2))
#     return ActionResponse(action=action)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}