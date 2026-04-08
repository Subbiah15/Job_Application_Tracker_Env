"""
AI Job Application Tracker — Core Environment (OpenEnv)
=======================================================
Provides reset(), step(), state(), and get_observation() following the
standard RL-style environment loop.
"""

from __future__ import annotations
import copy
from typing import Any

# ── Deterministic dataset of job applications ────────────────────────────
# Each entry carries ground-truth labels so graders can score the agent.
_APPLICATIONS_TEMPLATE: list[dict[str, Any]] = [
    {
        "company": "Google",
        "role": "SWE Intern",
        "status": "applied",
        "deadline_days": 5,
        "priority": "high",
        "recommended_action": "follow_up",
        "agent_status": None,
        "agent_priority": None,
        "agent_action": None,
    },
    {
        "company": "Microsoft",
        "role": "Data Scientist",
        "status": "interview",
        "deadline_days": 3,
        "priority": "high",
        "recommended_action": "prepare_interview",
        "agent_status": None,
        "agent_priority": None,
        "agent_action": None,
    },
    {
        "company": "Amazon",
        "role": "Backend Engineer",
        "status": "applied",
        "deadline_days": 10,
        "priority": "medium",
        "recommended_action": "follow_up",
        "agent_status": None,
        "agent_priority": None,
        "agent_action": None,
    },
    {
        "company": "Netflix",
        "role": "ML Engineer",
        "status": "offer",
        "deadline_days": 2,
        "priority": "high",
        "recommended_action": "accept_offer",
        "agent_status": None,
        "agent_priority": None,
        "agent_action": None,
    },
    {
        "company": "Meta",
        "role": "Research Scientist",
        "status": "rejected",
        "deadline_days": 0,
        "priority": "low",
        "recommended_action": "ignore",
        "agent_status": None,
        "agent_priority": None,
        "agent_action": None,
    },
    {
        "company": "Apple",
        "role": "iOS Developer",
        "status": "applied",
        "deadline_days": 7,
        "priority": "medium",
        "recommended_action": "follow_up",
        "agent_status": None,
        "agent_priority": None,
        "agent_action": None,
    },
]

# Valid value sets
VALID_STATUSES = {"applied", "interview", "rejected", "offer"}
VALID_PRIORITIES = {"high", "medium", "low"}
VALID_ACTIONS = {"follow_up", "prepare_interview", "accept_offer", "ignore"}

MAX_STEPS = 30  # safety limit


class JobTrackerEnv:
    """OpenEnv-compliant environment for AI job-application management."""

    def __init__(self) -> None:
        self.applications: list[dict[str, Any]] = []
        self.current_index: int = 0
        self.done: bool = True
        self.steps_taken: int = 0

    # ── Core API ─────────────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """Initialize a fresh episode and return the first observation."""
        self.applications = copy.deepcopy(_APPLICATIONS_TEMPLATE)
        self.current_index = 0
        self.done = False
        self.steps_taken = 0
        return self.get_observation()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Parse and execute *action*, return (observation, reward, done, info).

        Supported actions
        -----------------
        set_status:<value>     – e.g. set_status:interview
        set_priority:<value>   – e.g. set_priority:high
        take_action:<value>    – e.g. take_action:follow_up
        next                   – advance to the next application
        """
        if self.done:
            return self.get_observation(), 0.0, True, {"error": "Episode is done. Call reset()."}

        self.steps_taken += 1
        info: dict[str, Any] = {"valid": True, "message": ""}
        reward = 0.0
        app = self.applications[self.current_index]

        # Parse action string
        action = action.strip().lower()

        if action.startswith("set_status:"):
            value = action.split(":", 1)[1].strip()
            if value in VALID_STATUSES:
                app["agent_status"] = value
                # Immediate partial reward for correct status
                if value == app["status"]:
                    reward = 0.3
                else:
                    reward = -0.1
                info["message"] = f"Status set to '{value}'."
            else:
                reward = -0.1
                info["valid"] = False
                info["message"] = f"Invalid status '{value}'. Must be one of {VALID_STATUSES}."

        elif action.startswith("set_priority:"):
            value = action.split(":", 1)[1].strip()
            if value in VALID_PRIORITIES:
                app["agent_priority"] = value
                if value == app["priority"]:
                    reward = 0.3
                elif _priority_adjacent(value, app["priority"]):
                    reward = 0.15
                else:
                    reward = -0.1
                info["message"] = f"Priority set to '{value}'."
            else:
                reward = -0.1
                info["valid"] = False
                info["message"] = f"Invalid priority '{value}'. Must be one of {VALID_PRIORITIES}."

        elif action.startswith("take_action:"):
            value = action.split(":", 1)[1].strip()
            if value in VALID_ACTIONS:
                app["agent_action"] = value
                if value == app["recommended_action"]:
                    reward = 0.4
                else:
                    reward = -0.2
                info["message"] = f"Action '{value}' taken."
            else:
                reward = -0.2
                info["valid"] = False
                info["message"] = f"Invalid action '{value}'. Must be one of {VALID_ACTIONS}."

        elif action == "next":
            # Move to the next application
            if self.current_index < len(self.applications) - 1:
                self.current_index += 1
                info["message"] = "Advanced to next application."
            else:
                self.done = True
                info["message"] = "All applications processed. Episode complete."
        else:
            reward = -0.1
            info["valid"] = False
            info["message"] = f"Unrecognised action: '{action}'."

        # Check termination conditions
        if self.steps_taken >= MAX_STEPS:
            self.done = True
            info["message"] += " Max steps reached."

        return self.get_observation(), reward, self.done, info

    def state(self) -> dict[str, Any]:
        """Return the full internal state (for debugging / grading)."""
        return {
            "applications": copy.deepcopy(self.applications),
            "current_index": self.current_index,
            "done": self.done,
            "steps_taken": self.steps_taken,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return what the agent actually sees for the current application."""
        if self.done or self.current_index >= len(self.applications):
            return {"done": True}
        app = self.applications[self.current_index]
        return {
            "company": app["company"],
            "role": app["role"],
            "status": app["status"],
            "days_left": app["deadline_days"],
        }


# ── Helpers ──────────────────────────────────────────────────────────────

_PRIORITY_ORDER = {"high": 2, "medium": 1, "low": 0}


def _priority_adjacent(a: str, b: str) -> bool:
    """Return True if two priority levels are adjacent (e.g. high↔medium)."""
    return abs(_PRIORITY_ORDER.get(a, -1) - _PRIORITY_ORDER.get(b, -1)) == 1
