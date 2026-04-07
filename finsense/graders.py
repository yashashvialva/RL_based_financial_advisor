from finsense.models import StateModel


def grade_task1(state: StateModel) -> float:
    """
    Task 1 — Easy: Monthly Saver
    Goal: Save Rs.5000 in 15 days
    Grader focuses on: goal completion + low stress
    Score range: strictly within (0, 1), returned as [0.01, 0.99]
    """
    goal_completed = max(0.0, state.initial_goal - state.current_goal_remaining)
    goal_score = min(0.99, max(0.01, goal_completed / max(1.0, state.initial_goal)))
    
    # Stress bonus: max 0.19 when stress is 0, min 0.0 when stress >= 0.7
    stress_level = min(1.0, max(0.0, state.stress_level))  # Clamp stress_level to [0, 1]
    stress_bonus = max(0.0, min(0.19, 0.2 * (1.0 - stress_level / 0.7)))
    
    final_score = goal_score + stress_bonus  # Range: [0.01, 1.18]
    
    # CRITICAL: Ensure strictly between 0 and 1
    return min(0.99, max(0.01, final_score))


def grade_task2(state: StateModel) -> float:
    """
    Task 2 — Medium: Quarter Goal
    Goal: Save Rs.15000 in 30 days with income shocks
    Grader focuses on: goal progress + efficiency + overspend control
    Score range: strictly within (0, 1), returned as [0.01, 0.99]
    """
    goal_completed = max(0.0, state.initial_goal - state.current_goal_remaining)
    progress = min(0.99, max(0.01, goal_completed / max(1.0, state.initial_goal)))
    
    days_used_ratio = min(1.0, state.current_day / max(1, state.total_days))
    efficiency = min(0.99, max(0.01, 1.0 - days_used_ratio))
    
    risk_score = {"low": 1.0, "medium": 0.5, "high": 0.0}.get(state.risk_level, 0.5)
    risk_score = min(0.99, max(0.01, risk_score))
    
    # Weighted combination: 60% progress, 20% efficiency, 20% risk
    final_score = (0.6 * progress) + (0.2 * efficiency) + (0.2 * risk_score)  # Range: [0.01, 0.99]
    
    # CRITICAL: Ensure strictly between 0 and 1
    return min(0.99, max(0.01, final_score))


def grade_task3(state: StateModel) -> float:
    """
    Task 3 — Hard: Multi-Goal Chaos
    Goal: Save Rs.30000 + keep stress low + keep balance healthy
    Grader focuses on: all three simultaneously
    Score range: strictly within (0, 1), returned as [0.01, 0.99]
    """
    goal_completed = max(0.0, state.initial_goal - state.current_goal_remaining)
    savings_score = min(0.99, max(0.01, goal_completed / max(1.0, state.initial_goal)))
    
    # Stress score: max 0.99 when stress is 0, decreases as stress increases
    stress_level = min(1.0, max(0.0, state.stress_level))  # Clamp to [0, 1]
    stress_score = min(0.99, max(0.01, 1.0 - stress_level))
    
    # Balance score: higher balance is better, target is Rs.15000 emergency fund
    emergency_fund_target = 15000.0
    balance_score = min(0.99, max(0.01, state.balance / emergency_fund_target))
    
    # Weighted combination: 50% savings, 25% stress management, 25% balance health
    final_score = (0.5 * savings_score) + (0.25 * stress_score) + (0.25 * balance_score)  # Range: [0.01, 0.99]
    
    # CRITICAL: Ensure strictly between 0 and 1
    return min(0.99, max(0.01, final_score))


def grade_episode(state: StateModel) -> float:
    """
    Router: calls the correct per-task grader based on state.task_id.
    Falls back to task1 grader if task_id is unrecognised.
    """
    graders = {
        "easy": grade_task1,
        "medium": grade_task2,
        "hard": grade_task3,
    }
    grader = graders.get(state.task_id, grade_task1)
    return grader(state)