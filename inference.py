"""
AI Job Application Tracker — Baseline Inference Script
======================================================
Runs an AI agent (via OpenAI-compatible API) through the environment,
executes all 3 tasks, and prints scores.

Environment variables
---------------------
OPENAI_API_KEY   – API key
MODEL_NAME       – Model to use (default: gpt-3.5-turbo)
API_BASE_URL     – Base URL for the API (default: https://api.openai.com/v1)
"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

from environment import JobTrackerEnv
from tasks import (
    StatusClassificationTask,
    PriorityAssignmentTask,
    DecisionMakingTask,
)
from rewards import compute_reward

# ── Config ───────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")

if not API_KEY:
    print("ERROR: Set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


class RunAgentRequest(BaseModel):
    api_key: str = Field(..., description="OpenAI-compatible API key")
    model: str = Field(..., description="Model name (e.g. gpt-3.5-turbo, llama-3.1-8b-instant)")
    base_url: str = Field(..., description="API base URL")


# ── Endpoints ────────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> str:
    """Send a single prompt to the LLM and return the text response.
    Includes automatic retry with backoff for rate-limit errors."""
    import time

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "retry" in error_str or "resource" in error_str:
                wait_time = (attempt + 1) * 10  # 10s, 20s, 30s, 40s, 50s
                print(f"    ⏳ Rate limited. Waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"    ❌ API Error: {e}")
                raise
    print("    ❌ Max retries reached. Returning fallback.")
    return "applied"  # safe fallback


from prompts import (
    build_prompt_status,
    build_prompt_priority,
    build_prompt_action,
)


# ── Main loop ────────────────────────────────────────────────────────────

# ── Main loop ────────────────────────────────────────────────────────────

def run_inference(env=None, api_key=None, model=None, base_url=None):
    """
    Runs inference on the environment using the provided (or environment) config.
    Returns a dictionary of results.
    """
    from openai import OpenAI
    
    # Use provided or env vars
    _key = api_key or os.environ.get("OPENAI_API_KEY", "")
    _model = model or os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    _url = base_url or os.environ.get("API_BASE_URL", "https://api.openai.com/v1")

    if not _key:
        raise ValueError("API Key is missing")

    client = OpenAI(api_key=_key, base_url=_url)

    def _ask(prompt):
        import time
        for attempt in range(5):
            try:
                res = client.chat.completions.create(model=_model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100)
                return res.choices[0].message.content.strip()
            except Exception as e:
                if any(x in str(e).lower() for x in ["rate", "429", "retry"]):
                    time.sleep((attempt + 1) * 2)
                else: raise
        return "applied"

    if env is None:
        env = JobTrackerEnv()
    
    obs = env.reset()
    results = []
    total_reward = 0.0

    while not env.done:
        app = env.applications[env.current_index]
        gt = {"status": app["status"], "priority": app["priority"], "action": app["recommended_action"]}

        # Predictions
        s = _ask(build_prompt_status(obs)).lower()
        p = _ask(build_prompt_priority(obs)).lower()
        a = _ask(build_prompt_action(obs)).lower()

        # Step through env
        env.step(f"set_status:{s}")
        env.step(f"set_priority:{p}")
        _, reward, _, _ = env.step(f"take_action:{a}")
        env.step("next")
        
        results.append({
            "company": app["company"],
            "role": app["role"],
            "predicted": {"status": s, "priority": p, "action": a},
            "ground_truth": gt,
            "reward": reward
        })
        total_reward += reward
        obs = env.get_observation()

    return {
        "total_reward": round(total_reward, 4),
        "per_application": results,
        "max_possible_reward": float(len(results))
    }


if __name__ == "__main__":
    try:
        res = run_inference()
        print("\n" + "="*30 + "\nRESULTS\n" + "="*30)
        for r in res["per_application"]:
            print(f"{r['company']}: {r['reward']:.2f}")
        print(f"Total: {res['total_reward']}/{res['max_possible_reward']}")
    except Exception as e:
        print(f"Error: {e}")
