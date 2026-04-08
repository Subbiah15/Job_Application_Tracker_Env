---
title: AI Job Application Tracker
emoji: 🧾
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# 🧾 AI Job Application Tracker — OpenEnv

A real-world simulation environment where an AI agent manages job applications like a human job seeker. The agent must **track application status**, **prioritize opportunities**, and **decide next actions** — all evaluated with structured scoring.

---

## 🌍 Why This Project?

Job seekers:
- Apply to many companies simultaneously
- Track status manually (spreadsheets, notes)
- Forget follow-ups and miss deadlines

This environment simulates those real workflows and decisions, allowing an AI agent to learn efficient job application management.

---

## 🏗️ Environment Design

### Standard Loop

```
reset() → initialize environment
step(action) → update state, get reward
state() → return full internal state
```

### Observation (what the agent sees)

```json
{
  "company": "Google",
  "role": "SWE Intern",
  "status": "applied",
  "days_left": 5
}
```

### Action Space

| Action | Example |
|--------|---------|
| Set status | `set_status:interview` |
| Set priority | `set_priority:high` |
| Take action | `take_action:follow_up` |
| Next application | `next` |

**Valid values:**
- Status: `applied`, `interview`, `rejected`, `offer`
- Priority: `high`, `medium`, `low`
- Actions: `follow_up`, `prepare_interview`, `accept_offer`, `ignore`

### Episode Termination
- All applications processed, **or**
- Maximum steps reached (30)

---

## 🧪 Tasks & Graders

### Task 1 — Status Classification (Easy)
- **Goal:** Classify the application status
- **Scoring:** Exact match → 1.0, Wrong → 0.0

### Task 2 — Priority Assignment (Medium)
- **Goal:** Assign priority based on status and urgency
- **Scoring:** Exact → 1.0, Adjacent (e.g. high↔medium) → 0.5, Wrong → 0.0

### Task 3 — Decision Making (Hard)
- **Goal:** Choose the optimal next action
- **Scoring (weighted):**
  - Status correctness → 0.3
  - Priority correctness → 0.3
  - Action correctness → 0.4

All graders are **deterministic** — same input always produces the same score.

---

## 🎯 Reward Function

Continuous reward with partial-progress signals:

| Component | Reward |
|-----------|--------|
| Correct status | +0.3 |
| Correct priority | +0.3 |
| Correct action | +0.4 |
| Incorrect action | −0.2 |
| Missing prediction | −0.1 |

**Range:** −0.4 to +1.0 per application

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

```bash
python -m pytest tests/test_environment.py -v
```

### 3. Start the Server

```bash
python server.py
```

Server runs at `http://localhost:7860`

### 4. API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/state` | GET | Get full state |
| `/step` | POST | Take an action (`{"action": "set_status:applied"}`) |
| `/tasks` | GET | List available tasks |
| `/grade` | POST | Run a grader |
| `/evaluate` | POST | Full episode evaluation |

---

## 🤖 Baseline Inference

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-3.5-turbo"        # optional
export API_BASE_URL="https://api.openai.com/v1"  # optional

python inference.py
```

The script runs an LLM agent through all applications across all 3 tasks and prints per-application and summary scores.

---

## 🐳 Docker

```bash
docker build -t job-tracker-env .
docker run -p 7860:7860 job-tracker-env
```

---

## ☁️ Hugging Face Spaces

This project is ready for deployment as a Docker-based HF Space:

1. Create a new Space with **Docker** SDK
2. Push this repository
3. The Space will build and serve on port `7860`
4. Endpoints `/reset`, `/step`, `/state` are available

---

## 📁 Project Structure

```
├── environment.py        # Core OpenEnv (reset, step, state)
├── tasks.py              # 3 tasks + deterministic graders
├── rewards.py            # Continuous reward function
├── inference.py          # Baseline agent using OpenAI API
├── server.py             # FastAPI server (HF Spaces)
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Docker deployment
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── tests/
    └── test_environment.py
```

---

## 📋 OpenEnv Compliance

- ✅ `reset()` / `step()` / `state()` API
- ✅ 3 tasks with deterministic graders (Easy → Medium → Hard)
- ✅ Continuous reward function with partial-progress signals
- ✅ `openenv.yaml` spec file
- ✅ Inference script with env var configuration
- ✅ Docker & Hugging Face Spaces ready
- ✅ Reproducible results
