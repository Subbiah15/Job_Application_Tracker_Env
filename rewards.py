"""
AI Job Application Tracker — Reward Function
=============================================
Continuous reward with partial-progress signals.
"""

from __future__ import annotations
from typing import Any

from tasks import StatusClassificationTask, PriorityAssignmentTask


def compute_reward(
    predicted_status: str | None,
    predicted_priority: str | None,
    predicted_action: str | None,
    ground_truth: dict[str, str],
) -> float:
    """
    Compute a continuous reward for a single application.

    Reward breakdown
    ----------------
    +0.3   correct status
    +0.3   correct priority  (0.15 for adjacent)
    +0.4   correct action
    −0.2   incorrect action
    −0.1   redundant / missing predictions

    Parameters
    ----------
    predicted_status   : agent's status prediction (or None if not set)
    predicted_priority : agent's priority prediction (or None if not set)
    predicted_action   : agent's action prediction (or None if not set)
    ground_truth       : {"status": ..., "priority": ..., "action": ...}

    Returns
    -------
    float  — total reward (can be negative)
    """
    reward = 0.0

    # ── Status ───────────────────────────────────────────────────────
    if predicted_status is not None:
        status_score = StatusClassificationTask.grade(
            predicted_status, ground_truth["status"]
        )
        reward += 0.3 * status_score if status_score > 0 else -0.1
    else:
        reward -= 0.1  # penalty for not predicting

    # ── Priority ─────────────────────────────────────────────────────
    if predicted_priority is not None:
        priority_score = PriorityAssignmentTask.grade(
            predicted_priority, ground_truth["priority"]
        )
        if priority_score == 1.0:
            reward += 0.3
        elif priority_score == 0.5:
            reward += 0.15
        else:
            reward -= 0.1
    else:
        reward -= 0.1

    # ── Action ───────────────────────────────────────────────────────
    if predicted_action is not None:
        if predicted_action.strip().lower() == ground_truth["action"].strip().lower():
            reward += 0.4
        else:
            reward -= 0.2
    else:
        reward -= 0.1

    return round(reward, 4)
