---
title: FinSense RL
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# FinSense RL Environment

FinSense is a goal-driven financial decision-making RL environment simulating Indian household budgeting. Given a savings goal (e.g. buying a phone worth ₹20,000), a monthly salary, and daily expenses, an AI agent learns to decide which expenses to allow, reduce, or avoid — balancing stress, risk, and deadlines to hit the target in time.

## Overview

FinSense simulates real-world Indian household financial decision-making. An AI agent manages daily expenses across a fixed-horizon episode, deciding to **allow**, **reduce**, or **avoid** each incoming expense — balancing a savings goal, stress accumulation, and stochastic income shocks.

Unlike toy environments, FinSense models a task that millions of Indians face every month: a fixed salary, unpredictable expenses, and a savings goal that keeps slipping. The environment captures delayed consequences (avoiding food today increases stress), trade-offs (essential vs discretionary spending), and temporal pressure (required savings per day increases as the deadline approaches).

## Why FinSense?

Most existing RL environments use games or abstract simulations. FinSense is grounded in real Indian household economics:
- Expense categories include Swiggy, Zomato, BESCOM electricity, auto rickshaw, EMI payments
- Amounts are in INR and reflect realistic Indian price ranges
- The stress mechanic prevents the trivial "avoid everything" exploit — avoiding essential expenses like rent or medicine increases stress, which penalises the agent
- The `daily_allowance` field tells the agent exactly how much it can spend today and still hit the savings goal

## Environment Design

### Episode Structure
Each episode runs for a fixed number of days (15/30/45 depending on task). At each step, the agent receives one incoming expense and must decide what to do with it. At the end of each day, savings are calculated and the goal is updated. The episode ends when either the goal is achieved or days run out.

### Stress Mechanic (Anti-Exploit)
The biggest flaw in naive financial RL environments is the "avoid everything" exploit — an agent that avoids all expenses trivially saves the most money. FinSense prevents this with a necessity-based stress penalty:

| Necessity Tag | Examples | Stress Penalty if Avoided |
|---------------|----------|--------------------------|
| essential | rent, electricity, medicine, groceries | +0.30 per avoid |
| semi-essential | Swiggy, Uber, transport | +0.10 per avoid |
| discretionary | movies, concerts, fine dining | 0.00 (correct to avoid) |

High stress triggers episode-end penalties, making the "avoid everything" policy suboptimal.

### Daily Allowance System
At each step the agent receives `daily_allowance` — the maximum it can spend today while still achieving its savings goal:

```
daily_allowance = (balance - goal_remaining) / days_left
```

This gives the agent a clear budget signal without requiring it to do complex financial math.

## Observation Space

Each step returns a typed observation with 15 fields:

| Field | Type | Description |
|-------|------|-------------|
| `balance` | float | Current account balance in INR |
| `goal_total` | float | Total savings goal in INR |
| `goal_remaining` | float | Amount still needed to reach goal |
| `days_left` | int | Days remaining in episode |
| `daily_allowance` | float | Max spend today to still hit goal |
| `required_savings_per_day` | float | Savings needed per day from now |
| `stress_level` | float [0-1] | Accumulated stress from avoiding essentials |
| `risk_level` | string | low / medium / high based on balance |
| `expected_fixed_expenses` | float | Total fixed costs remaining |
| `income_shock_active` | bool | Whether a shock event is active |
| `recent_spending` | list[float] | Last 5 transaction amounts |
| `avg_daily_spend` | float | Rolling average daily spend |
| `salary` | float | Starting balance / salary |
| `task_id` | string | Current task identifier |
| `current_expense` | dict | The incoming expense to decide on |

## Action Space

The agent returns one action per step:

```json
{
  "decision": "allow | reduce | avoid",
  "approved_amount": 0.0,
  "reasoning": "optional string"
}
```

| Decision | Effect |
|----------|--------|
| `allow` | Pay full expense amount |
| `reduce` | Pay `approved_amount` (must be less than full amount) |
| `avoid` | Pay nothing (stress penalty if essential) |

## Reward Function

Rewards are dense (per-step) plus sparse (episode-end):

**Per-step:**
```
reward = goal_progress - stress_penalty - risk_penalty
```

**Episode-end:**
```
+50  if goal achieved
-30  if goal not achieved
-20  if stress_level > 0.7
```

This asymmetric design encourages partial progress — an agent that saves 80% of the goal still gets meaningful reward.

## Tasks

| Task | Goal | Days | Initial Balance | Difficulty |
|------|------|------|----------------|------------|
| easy | Save ₹5,000 | 15 | ₹30,000 | Easy — no shocks, simple expenses |
| medium | Save ₹15,000 | 30 | ₹60,000 | Medium — income shocks, mixed expenses |
| hard | Save ₹30,000 | 45 | ₹80,000 | Hard — frequent shocks, tight margins |

### Graders

Each task has a separate deterministic grader returning a float in [0.0, 1.0]:

- **Task 1 (easy):** `0.8 * goal_progress + 0.2 * stress_bonus`
- **Task 2 (medium):** `0.6 * goal_progress + 0.2 * efficiency + 0.2 * risk_control`
- **Task 3 (hard):** `0.5 * goal_progress + 0.25 * stress_control + 0.25 * balance_health`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Take an action, returns obs, reward, done, info |
| `/state` | GET | Get current state without advancing |
| `/tasks` | GET | List available tasks |
| `/health` | GET | Health check |

## Standardized Quick Start (Multi-Mode)

The environment is now fully compliant with OpenEnv's multi-mode deployment standards.

```bash
# 1. Install dependencies
uv sync

# 2. Run the environment server (standardized entry point)
uv run server

# 3. Verify validation
./scripts/validate-submission.sh https://yashashvialva-finsense-rl.hf.space
```

## Legacy API Quick Start (Existing)

```bash
# Reset environment
curl -X POST https://yashashvialva-finsense-rl.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Take a step
curl -X POST https://yashashvialva-finsense-rl.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"decision": "avoid", "approved_amount": 0.0, "reasoning": "discretionary"}'

# Get current state
curl -X GET https://yashashvialva-finsense-rl.hf.space/state
```

## Baseline Scores

Running `meta-llama/Llama-3.1-8B-Instruct` via HF Inference Providers:

| Task | Score | Success |
|------|-------|---------|
| easy | ~0.82 | true |
| medium | ~0.65 | partial |
| hard | ~0.50 | partial |

## Project Structure

```
finsense-rl/
├── server/                   # Multi-mode server package
│   └── app.py                # Main entry point (server.app:main)
├── scripts/                  # Automation & validation scripts
│   └── validate-submission.sh # Submission pre-check tool
├── finsense/                 # Core logic package
│   ├── env.py                # FinSenseEnv class
│   ├── models.py             # Pydantic models
│   ├── server.py             # FastAPI endpoints
│   ├── tasks.py              # Task configurations
│   ├── expense_generator.py  # Seeded engine
│   ├── graders.py            # Per-task graders
│   └── reward.py             # Reward calculator
├── inference.py              # Standardized LLM agent loop
├── openenv.yaml              # Task metadata
├── Dockerfile                # HF Space container config
├── pyproject.toml            # Build metadata & entry points
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

### Using uv (Recommended)
```bash
uv sync
uv run server
```

### Using pip
```bash
pip install -r requirements.txt
python -m server.app
```

## Environment Variables

```bash
HF_TOKEN        # Hugging Face API token (required)
API_BASE_URL    # API endpoint (default: https://router.huggingface.co/v1)
MODEL_NAME      # Model identifier (default: meta-llama/Llama-3.1-8B-Instruct:cerebras)
```

## Validation

To ensure your submission is correct before submitting to the platform, you can run the provided validation script.

### Prerequisites
- Docker installed
- `openenv-core` installed: `pip install openenv-core`

### Running Validation
```bash
# Make the script executable
chmod +x scripts/validate-submission.sh

# Run validation against your Hugging Face Space
./scripts/validate-submission.sh https://your-space.hf.space
```

**Note**: If you are on Windows, run this in a Git Bash or WSL terminal.
```