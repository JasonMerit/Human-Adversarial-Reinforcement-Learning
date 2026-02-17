import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from .database import get_connection, init_db
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


@app.post("/act", response_model=ActionResponse)  # Should be GET
async def act(request: StateRequest):
    state = (np.array(request.grid), np.array(request.bike1), np.array(request.bike2))
    action = agent(state)  
    return ActionResponse(action=action)



class Trajectory(BaseModel):
    actions: List[Tuple[int, int]]
    winner: int

@app.on_event("startup")
def startup():
    init_db()

@app.post("/trajectory")
def store_trajectory(traj: Trajectory):
    with get_connection() as con:
        cur = con.cursor()

        cur.execute(
            "INSERT INTO games (winner, length) VALUES (?, ?)",
            (traj.winner, len(traj.actions))
        )
        game_id = cur.lastrowid

        for t, (a1, a2) in enumerate(traj.actions):
            cur.execute(
                "INSERT INTO steps (game_id, t, action_p1, action_p2) VALUES (?, ?, ?, ?)",
                (game_id, t, a1, a2)
            )

        con.commit()

    return {"game_id": game_id}
