from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from finsense.env import FinSenseEnv
from finsense.models import ActionModel
from finsense.tasks import TASKS
from finsense.graders import grade_episode

app = FastAPI(title="FinSense RL Environment")
env = FinSenseEnv()

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

@app.post("/reset", response_model=Dict[str, Any])
def reset_env(req: ResetRequest = ResetRequest()):
    return env.reset(task_id=req.task_id, seed=req.seed)

@app.post("/step", response_model=StepResponse)
def step_env(action: ActionModel):
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not reset")
    
    obs, rew, done, info = env.step(action)
    return StepResponse(observation=obs, reward=rew, done=done, info=info)

@app.get("/state", response_model=Dict[str, Any])
def get_state():
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not reset")
    return env.get_state()

@app.get("/tasks", response_model=List[str])
def get_tasks():
    return list(TASKS.keys())

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "ok"}
