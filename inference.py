"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import os
import json
import sys
from openai import OpenAI
from finsense.env import FinSenseEnv
from finsense.models import ActionModel, StateModel, Expense
from finsense.graders import grade_episode
from finsense.tasks import TASKS

def get_fallback_action(obs: dict) -> ActionModel:
    """
    Smart rule-based fallback when API fails or LLM returns invalid output.
    """
    exp = obs.get("current_expense") or {}
    necessity = exp.get("necessity_tag", "discretionary")
    amount = float(exp.get("amount", 0))

    if necessity == "essential":
        return ActionModel(decision="allow", approved_amount=amount, reasoning="Fallback: essential allowed")
    elif necessity == "semi-essential":
        return ActionModel(decision="reduce", approved_amount=round(amount * 0.5, 2), reasoning="Fallback: semi-essential reduced")
    else:
        return ActionModel(decision="avoid", approved_amount=0.0, reasoning="Fallback: discretionary avoided")

def build_prompt(obs: dict) -> str:
    """
    Short, token-efficient prompt.
    """
    exp = obs.get("current_expense", {}) or {}
    necessity = exp.get("necessity_tag", "discretionary")
    amount = exp.get("amount", 0)
    name = exp.get("name", "Unknown")

    return f"""Financial agent. Decide on this expense.

Balance: {obs.get('balance', 0):.0f} | Goal Left: {obs.get('goal_remaining', 0):.0f} | Days: {obs.get('days_left', 0)}
Expense: {name} | Amount: {amount:.0f} | Type: {necessity}

Rules:
- essential -> allow full amount
- semi-essential -> reduce by 50%
- discretionary -> avoid

Reply ONLY with JSON: {{"decision": "allow/reduce/avoid", "approved_amount": 0.0, "reasoning": "short"}}"""

def calculate_final_score(env, task_id):
    s = env.state
    task = TASKS.get(task_id)
    
    try:
        state_data = {
            "current_day": task.days - s["days_left"],
            "total_days": task.days,
            "balance": s["balance"],
            "initial_goal": s["goal_total"],
            "current_goal_remaining": s["goal_remaining"],
            "stress_level": s["stress_level"],
            "risk_level": s["risk_level"],
            "seed": 42,
            "task_id": task_id,
            "expected_fixed_expenses": s["expected_fixed_expenses"],
            "income_shock_active": s["income_shock_active"],
            "recent_spending": s["recent_spending"],
            "user_type": "balanced",
            "current_expense_idx": 0,
            "daily_expenses": [],
            "daily_expense_idx": 0,
            "terminated": s["days_left"] <= 0,
            "truncated": False
        }
        state_model = StateModel(**state_data)
        raw = grade_episode(state_model)
        # CRITICAL: always clamp, even if grader returns something unexpected
        return max(0.0, min(1.0, raw))
    except Exception:
        goal_total = s["goal_total"]
        goal_remaining = s["goal_remaining"]
        goal_progress = max(0.0, goal_total - goal_remaining)
        raw = goal_progress / max(1.0, goal_total)
        return max(0.0, min(1.0, raw))
def run_inference(task_id="easy"):
    # Load environment variables
    HF_TOKEN = os.getenv("HF_TOKEN")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:cerebras")

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    env = FinSenseEnv()
    obs = env.reset(task_id=task_id)

    # [START] header
    print(f"[START] task={task_id} env=finsense-rl model={MODEL_NAME}")

    step_num = 0
    all_rewards = []
    done = False
    last_error = "null"

    try:
        while not done:
            step_num += 1
            action_str = "null"
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
                        {"role": "user", "content": build_prompt(obs)}
                    ],
                    max_tokens=80,
                    temperature=0.1
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.split("```")[0].strip()

                action_dict = json.loads(raw)

                # Sanitize
                if action_dict.get("decision") not in ("allow", "reduce", "avoid"):
                    action_dict["decision"] = "avoid"
                if not isinstance(action_dict.get("approved_amount"), (int, float)):
                    action_dict["approved_amount"] = 0.0
                
                # Enforce essential rule
                exp = obs.get("current_expense") or {}
                if exp.get("necessity_tag") == "essential" and action_dict.get("decision") in ("reduce", "avoid"):
                    action_dict["decision"] = "allow"
                    action_dict["approved_amount"] = float(exp.get("amount", 0))

                action = ActionModel(**action_dict)
                action_str = f"{action.decision}({action.approved_amount:.0f})"
                last_error = "null"

            except Exception as e:
                action = get_fallback_action(obs)
                action_str = f"fallback:{action.decision}({action.approved_amount:.0f})"
                last_error = str(e).replace('\n', ' ')

            obs, reward, done, info = env.step(action)
            all_rewards.append(reward)

            # [STEP] line
            done_str = str(done).lower()
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={last_error}")

    except Exception as e:
        last_error = str(e).replace('\n', ' ')
    finally:
        success_bool = obs.get("goal_remaining", 0) <= 0
        success_str = str(success_bool).lower()
        score = calculate_final_score(env, task_id)
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

        # [END] line
        print(f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_inference(task)