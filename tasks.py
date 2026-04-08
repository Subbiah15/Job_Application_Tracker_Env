"""
AI Job Application Tracker — Task Definitions & Graders
========================================================
Three tasks of increasing difficulty, each with a deterministic grader.
"""

from __future__ import annotations
from typing import Any


# ── Task 1: Status Classification (Easy) ─────────────────────────────────

class StatusClassificationTask:
    """
    Classify the application status.

    Output : one of  applied | interview | rejected | offer
    Scoring: exact match → 1.0 , otherwise → 0.0
    """

    name = "status_classification"
    difficulty = "easy"

    @staticmethod
    def grade(predicted: str, actual: str) -> float:
        """Deterministic grader — exact match."""
        return 1.0 if predicted.strip().lower() == actual.strip().lower() else 0.0


# ── Task 2: Priority Assignment (Medium) ──────────────────────────────────

_PRIORITY_ORDER = {"high": 2, "medium": 1, "low": 0}


class PriorityAssignmentTask:
    """
    Assign priority to the application.

    Output : one of  high | medium | low
    Scoring: exact → 1.0 , adjacent → 0.5 , wrong → 0.0
    """

    name = "priority_assignment"
    difficulty = "medium"

    @staticmethod
    def grade(predicted: str, actual: str) -> float:
        """Deterministic grader — exact / adjacent / wrong."""
        p = predicted.strip().lower()
        a = actual.strip().lower()
        if p == a:
            return 1.0
        p_val = _PRIORITY_ORDER.get(p)
        a_val = _PRIORITY_ORDER.get(a)
        if p_val is not None and a_val is not None and abs(p_val - a_val) == 1:
            return 0.5
        return 0.0


# ── Task 3: Decision Making (Hard) ───────────────────────────────────────

class DecisionMakingTask:
    """
    Choose the best next action given the full context.

    The agent must supply status, priority, AND action predictions.

    Scoring (weighted):
        status  correctness → 0.3
        priority correctness → 0.3
        action  correctness → 0.4
    """

    name = "decision_making"
    difficulty = "hard"

    @staticmethod
    def grade(
        predicted: dict[str, str],
        actual: dict[str, str],
    ) -> float:
        """
        Deterministic grader — weighted composite.

        Parameters
        ----------
        predicted : {"status": ..., "priority": ..., "action": ...}
        actual    : {"status": ..., "priority": ..., "action": ...}
        """
        score = 0.0

        # Status component (0.3)
        score += 0.3 * StatusClassificationTask.grade(
            predicted.get("status", ""),
            actual.get("status", ""),
        )

        # Priority component (0.3)
        score += 0.3 * PriorityAssignmentTask.grade(
            predicted.get("priority", ""),
            actual.get("priority", ""),
        )

        # Action component (0.4)
        pred_action = predicted.get("action", "").strip().lower()
        actual_action = actual.get("action", "").strip().lower()
        score += 0.4 * (1.0 if pred_action == actual_action else 0.0)

        return round(score, 4)


# ── Registry of all tasks ────────────────────────────────────────────────

TASKS = [
    StatusClassificationTask,
    PriorityAssignmentTask,
    DecisionMakingTask,
]


def get_task(name: str):
    """Look up a task class by name."""
    for t in TASKS:
        if t.name == name:
            return t
    raise ValueError(f"Unknown task: {name}")
