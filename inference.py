import os
import json
from openai import OpenAI
from finsense.env import FinSenseEnv
from finsense.models import ActionModel


def get_fallback_action(obs: dict) -> ActionModel:
    """
    Smart rule-based fallback when API fails.
    Essential → allow, semi-essential → reduce 50%, discretionary → avoid.
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
    Short, token-efficient prompt. Saves credits significantly.
    """
    exp = obs.get("current_expense", {}) or {}
    necessity = exp.get("necessity_tag", "discretionary")
    amount = exp.get("amount", 0)
    name = exp.get("name", "Unknown")

    return f"""Financial agent. Decide on this expense.

Balance: {obs.get('balance', 0):.0f} | Goal Left: {obs.get('goal_remaining', 0):.0f} | Days: {obs.get('days_left', 0)}
Expense: {name} | Amount: {amount:.0f} | Type: {necessity}

Rules:
- essential → allow full amount
- semi-essential → reduce by 50%
- discretionary → avoid

Reply ONLY with JSON: {{"decision": "allow/reduce/avoid", "approved_amount": 0.0, "reasoning": "short"}}"""


def run_inference():
    print("[START]")

    api_key = os.getenv("HF_TOKEN", "dummy_token")
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    #model_name = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct:nebius")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
    client = OpenAI(api_key=api_key, base_url=base_url)

    env = FinSenseEnv()
    obs = env.reset(task_id="easy")

    step_num = 0
    all_rewards = []
    done = False

    while not done:
        print("[STEP]")
        step_num += 1

        try:
            response = client.chat.completions.create(
                model=model_name,
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

            # Enforce essential rule — never reduce/avoid essential
            exp = obs.get("current_expense") or {}
            if exp.get("necessity_tag") == "essential" and action_dict.get("decision") in ("reduce", "avoid"):
                action_dict["decision"] = "allow"
                action_dict["approved_amount"] = float(exp.get("amount", 0))
                action_dict["reasoning"] = "Overridden: essential must be allowed"

            action = ActionModel(**action_dict)
            print(f" -> Agent decided to {action.decision} Rs. {action.approved_amount} - Reason: {action.reasoning}")

        except Exception as e:
            print(f" -> Encountered AI error: {e}")
            action = get_fallback_action(obs)

        obs, reward, done, info = env.step(action)
        all_rewards.append(round(reward, 4))

    rewards_str = ",".join(str(r) for r in all_rewards)
    success = obs.get("goal_remaining", 0) <= 0

    print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}")


if __name__ == "__main__":
    run_inference()