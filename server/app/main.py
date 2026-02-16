from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .agents import SemiDeterministicAgent

app = FastAPI(title="RL Agent API")
agent = SemiDeterministicAgent(.5)


# Request schema
class StateRequest(BaseModel):
    state: list[float]  # list of floats or ints


# Response schema
class ActionResponse(BaseModel):
    action: int


@app.post("/act", response_model=ActionResponse)
async def act(request: StateRequest):
    if len(request.state) != 121:
        raise HTTPException(status_code=400, detail="Invalid state length")

    try:
        action = agent.act(request.state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ActionResponse(action=action)
