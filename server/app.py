"""
AI Job Application Tracker — FastAPI Server
============================================
Exposes the environment as HTTP endpoints for Hugging Face Spaces deployment.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any

from environment import JobTrackerEnv
from tasks import TASKS
from rewards import compute_reward

app = FastAPI(
    title="AI Job Application Tracker — OpenEnv",
    description=(
        "An RL-style environment where an AI agent manages job applications.\n\n"
        "**Action examples:**\n"
        "- `set_status:applied` / `set_status:interview` / `set_status:rejected` / `set_status:offer`\n"
        "- `set_priority:high` / `set_priority:medium` / `set_priority:low`\n"
        "- `take_action:follow_up` / `take_action:prepare_interview` / `take_action:accept_offer` / `take_action:ignore`\n"
        "- `next` — advance to the next application"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = JobTrackerEnv()


# ── Request / Response models ────────────────────────────────────────────

class ActionRequest(BaseModel):
    action: str = Field(
        ...,
        examples=["set_status:applied"],
        description="Action to take. Examples: set_status:interview, set_priority:high, take_action:follow_up, next",
    )


class GradeRequest(BaseModel):
    task: str = Field(
        ...,
        examples=["status_classification"],
        description="Task name: status_classification, priority_assignment, or decision_making",
    )
    predicted: Any = Field(
        ...,
        examples=["applied"],
        description="Predicted value (string for task 1 & 2, dict for task 3)",
    )
    actual: Any = Field(
        ...,
        examples=["applied"],
        description="Ground-truth value (string for task 1 & 2, dict for task 3)",
    )


class RunAgentRequest(BaseModel):
    api_key: str = Field(..., description="OpenAI-compatible API key")
    model: str = Field(..., description="Model name (e.g. gpt-3.5-turbo, llama-3.1-8b-instant)")
    base_url: str = Field(..., description="API base URL")


# ── Helpers ──────────────────────────────────────────────────────────────

@app.post("/run_agent")
async def run_agent(req: RunAgentRequest):
    """Run the agent on the environment."""
    from inference import run_inference
    try:
        results = run_inference(
            env=env,
            api_key=req.api_key,
            model=req.model,
            base_url=req.base_url
        )
        return results
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "environment": "job_application_tracker"}


@app.post("/reset")
def reset():
    """Reset the environment and return the first observation."""
    obs = env.reset()
    return {"observation": obs}


@app.get("/state")
def get_state():
    """Return the full internal state."""
    return env.state()


@app.post("/step")
def step(req: ActionRequest):
    """
    Execute an action and return (observation, reward, done, info).

    **Valid actions:**
    - `set_status:applied` / `set_status:interview` / `set_status:rejected` / `set_status:offer`
    - `set_priority:high` / `set_priority:medium` / `set_priority:low`
    - `take_action:follow_up` / `take_action:prepare_interview` / `take_action:accept_offer` / `take_action:ignore`
    - `next` — advance to the next application
    """
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/tasks")
def list_tasks():
    """List available tasks with metadata."""
    return [
        {"name": t.name, "difficulty": t.difficulty, "doc": t.__doc__.strip()}
        for t in TASKS
    ]


@app.post("/grade")
def grade(req: GradeRequest):
    """
    Run a task grader and return the score.

    **Examples:**
    - Task 1: `{"task": "status_classification", "predicted": "applied", "actual": "applied"}` → score 1.0
    - Task 2: `{"task": "priority_assignment", "predicted": "high", "actual": "medium"}` → score 0.5
    - Task 3: `{"task": "decision_making", "predicted": {"status":"applied","priority":"high","action":"follow_up"}, "actual": {"status":"applied","priority":"high","action":"follow_up"}}` → score 1.0
    """
    from tasks import get_task

    try:
        task_cls = get_task(req.task)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown task: '{req.task}'",
                "valid_tasks": ["status_classification", "priority_assignment", "decision_making"],
            },
        )

    try:
        score = task_cls.grade(req.predicted, req.actual)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Grading failed: {str(e)}", "hint": "For decision_making, predicted and actual must be dicts with keys: status, priority, action"},
        )

    return {"task": req.task, "score": score}


@app.post("/evaluate")
def evaluate():
    """
    Run a full evaluation episode.
    Computes per-application reward using agent predictions vs ground truth.
    Works at any point — shows results for all applications.
    """
    if not env.applications:
        return {"error": "No applications loaded. Call /reset first."}

    results = []
    total_reward = 0.0

    for app in env.applications:
        gt = {
            "status": app["status"],
            "priority": app["priority"],
            "action": app["recommended_action"],
        }
        r = compute_reward(
            app["agent_status"],
            app["agent_priority"],
            app["agent_action"],
            gt,
        )
        results.append({
            "company": app["company"],
            "role": app["role"],
            "predicted": {
                "status": app["agent_status"],
                "priority": app["agent_priority"],
                "action": app["agent_action"],
            },
            "ground_truth": gt,
            "reward": r,
        })
        total_reward += r

    return {
        "total_reward": round(total_reward, 4),
        "max_possible_reward": round(1.0 * len(env.applications), 4),
        "per_application": results,
    }


# ── Static files & root redirect ────────────────────────────────────────

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
def root():
    """Redirect to the interactive dashboard."""
    return RedirectResponse(url="/static/index.html")


# ── Run with Uvicorn ─────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == '__main__':
    main()

