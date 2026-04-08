"""
Tests for the AI Job Application Tracker Environment
=====================================================
"""

import sys
import os

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import JobTrackerEnv, VALID_STATUSES, VALID_PRIORITIES, VALID_ACTIONS
from tasks import StatusClassificationTask, PriorityAssignmentTask, DecisionMakingTask
from rewards import compute_reward


# ═══════════════════════════════════════════════════════════════════════════
#  Environment Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = JobTrackerEnv()
        obs = env.reset()
        assert "company" in obs
        assert "role" in obs
        assert "status" in obs
        assert "days_left" in obs

    def test_reset_sets_not_done(self):
        env = JobTrackerEnv()
        env.reset()
        assert env.done is False
        assert env.current_index == 0
        assert env.steps_taken == 0

    def test_reset_loads_applications(self):
        env = JobTrackerEnv()
        env.reset()
        assert len(env.applications) == 6


class TestEnvironmentStep:
    def test_set_status_valid(self):
        env = JobTrackerEnv()
        env.reset()
        obs, reward, done, info = env.step("set_status:applied")
        assert info["valid"] is True
        assert env.applications[0]["agent_status"] == "applied"

    def test_set_status_invalid(self):
        env = JobTrackerEnv()
        env.reset()
        obs, reward, done, info = env.step("set_status:banana")
        assert info["valid"] is False
        assert reward == -0.1

    def test_set_priority_valid(self):
        env = JobTrackerEnv()
        env.reset()
        obs, reward, done, info = env.step("set_priority:high")
        assert info["valid"] is True
        assert env.applications[0]["agent_priority"] == "high"

    def test_take_action_valid(self):
        env = JobTrackerEnv()
        env.reset()
        obs, reward, done, info = env.step("take_action:follow_up")
        assert info["valid"] is True
        assert env.applications[0]["agent_action"] == "follow_up"

    def test_next_advances_index(self):
        env = JobTrackerEnv()
        env.reset()
        env.step("next")
        assert env.current_index == 1

    def test_episode_ends_after_last_next(self):
        env = JobTrackerEnv()
        env.reset()
        for _ in range(len(env.applications)):
            env.step("next")
        assert env.done is True

    def test_step_after_done_returns_error(self):
        env = JobTrackerEnv()
        env.reset()
        env.done = True
        obs, reward, done, info = env.step("set_status:applied")
        assert done is True
        assert "error" in info

    def test_unrecognised_action(self):
        env = JobTrackerEnv()
        env.reset()
        obs, reward, done, info = env.step("fly_to_moon")
        assert info["valid"] is False
        assert reward == -0.1


class TestEnvironmentState:
    def test_state_returns_full_dict(self):
        env = JobTrackerEnv()
        env.reset()
        s = env.state()
        assert "applications" in s
        assert "current_index" in s
        assert "done" in s
        assert "steps_taken" in s


# ═══════════════════════════════════════════════════════════════════════════
#  Grader Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStatusClassificationGrader:
    def test_exact_match(self):
        assert StatusClassificationTask.grade("applied", "applied") == 1.0

    def test_wrong(self):
        assert StatusClassificationTask.grade("interview", "applied") == 0.0

    def test_case_insensitive(self):
        assert StatusClassificationTask.grade("APPLIED", "applied") == 1.0

    def test_deterministic(self):
        """Same input always produces same output."""
        for _ in range(10):
            assert StatusClassificationTask.grade("offer", "offer") == 1.0


class TestPriorityAssignmentGrader:
    def test_exact_match(self):
        assert PriorityAssignmentTask.grade("high", "high") == 1.0

    def test_adjacent(self):
        assert PriorityAssignmentTask.grade("high", "medium") == 0.5
        assert PriorityAssignmentTask.grade("medium", "low") == 0.5

    def test_wrong(self):
        assert PriorityAssignmentTask.grade("high", "low") == 0.0

    def test_deterministic(self):
        for _ in range(10):
            assert PriorityAssignmentTask.grade("medium", "high") == 0.5


class TestDecisionMakingGrader:
    def test_perfect_score(self):
        pred = {"status": "applied", "priority": "high", "action": "follow_up"}
        actual = {"status": "applied", "priority": "high", "action": "follow_up"}
        assert DecisionMakingTask.grade(pred, actual) == 1.0

    def test_all_wrong(self):
        pred = {"status": "rejected", "priority": "low", "action": "ignore"}
        actual = {"status": "applied", "priority": "high", "action": "follow_up"}
        assert DecisionMakingTask.grade(pred, actual) == 0.0

    def test_partial_score(self):
        pred = {"status": "applied", "priority": "low", "action": "ignore"}
        actual = {"status": "applied", "priority": "high", "action": "follow_up"}
        score = DecisionMakingTask.grade(pred, actual)
        # status correct (0.3), priority wrong (0.0), action wrong (0.0)
        assert score == 0.3

    def test_adjacent_priority(self):
        pred = {"status": "applied", "priority": "medium", "action": "follow_up"}
        actual = {"status": "applied", "priority": "high", "action": "follow_up"}
        score = DecisionMakingTask.grade(pred, actual)
        # status 0.3 + priority 0.3*0.5 + action 0.4 = 0.85
        assert score == 0.85


# ═══════════════════════════════════════════════════════════════════════════
#  Reward Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRewardFunction:
    def test_perfect_reward(self):
        gt = {"status": "applied", "priority": "high", "action": "follow_up"}
        r = compute_reward("applied", "high", "follow_up", gt)
        assert r == 1.0

    def test_all_wrong(self):
        gt = {"status": "applied", "priority": "high", "action": "follow_up"}
        r = compute_reward("rejected", "low", "ignore", gt)
        assert r < 0

    def test_missing_predictions(self):
        gt = {"status": "applied", "priority": "high", "action": "follow_up"}
        r = compute_reward(None, None, None, gt)
        assert r == -0.3  # three missing penalties

    def test_partial_reward(self):
        gt = {"status": "applied", "priority": "high", "action": "follow_up"}
        r = compute_reward("applied", "medium", "follow_up", gt)
        # status +0.3, priority adjacent +0.15, action +0.4 = 0.85
        assert r == 0.85
