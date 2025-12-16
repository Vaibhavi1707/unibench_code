import json
from collections import defaultdict

def calculate_ets_at_x(result_files):
    """
    Calculates the Expected Tries to Solve (ETS@x).
    
    Estimates the average number of attempts required to solve an issue,
    bounded by the maximum attempts x.
    
    Lower is BETTER.
    Minimum = 1.0 (Solves everything first try)
    Maximum = x   (Never solves anything)
    """
    x = len(result_files)
    if x == 0: return {}

    # 1. Collect Data
    issue_y_counts = defaultdict(int)
    all_submitted_ids = set()

    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'submitted_ids' in data:
                all_submitted_ids.update(data['submitted_ids'])
            if 'resolved_ids' in data:
                for issue_id in data['resolved_ids']:
                    issue_y_counts[issue_id] += 1
        except Exception:
            pass

    total_issues = len(all_submitted_ids)
    if total_issues == 0: return {}

    total_expected_cost = 0.0
    
    # 2. Calculate Cost per Issue
    for issue_id in all_submitted_ids:
        y = issue_y_counts[issue_id]
        p_hat = y / x
        
        if y == 0:
            # Case: Never solved. Cost is the full budget x.
            # (You could penalize this more, e.g., x+1, but x is standard for 'spent budget')
            cost = x
        elif y == x:
            # Case: Solved every time. 
            # Geometric sum formula simplifies to approx 1, but let's be precise:
            # (1 - 0^x) / 1 = 1.0
            cost = 1.0
        else:
            # Truncated Geometric Expectation
            # Sum_{k=1 to x} k * p * (1-p)^(k-1) ... normalized by probability of success?
            # Actually, the formula (1 - (1-p)^x) / p is the expected successes in x trials.
            # We want expected TRIALS to first success.
            
            # The exact formula for Expected Value of a Truncated Geometric Variable T 
            # (where T is trial number of first success, or x if no success):
            # E[T] = (1/p) * (1 - (1-p)^x) ... Wait, that's slightly off.
            
            # Let's use the explicit sum for clarity and correctness:
            # E[T] = Sum(k * P(T=k)) + x * P(Fail)
            # P(T=k) = p * (1-p)^(k-1)
            # P(Fail) = (1-p)^x
            
            expected_trials = 0.0
            prob_fail_streak = 1.0
            
            for k in range(1, x + 1):
                prob_success_now = prob_fail_streak * p_hat
                expected_trials += k * prob_success_now
                prob_fail_streak *= (1 - p_hat)
            
            # Add the cost of failing all x times (which consumes x tries)
            expected_trials += x * prob_fail_streak
            
            cost = expected_trials

        total_expected_cost += cost

    avg_ets = total_expected_cost / total_issues

    return {
        "ETS_at_x": avg_ets,
        "x_limit": x,
        "interpretation": f"On average, the agent takes {avg_ets:.2f} tries to solve an issue."
    }

# --- Usage ---
# run_files = [
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_1.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_2.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_3.json"
# ]

# run_files = [
#     "/Users/aryan.vi.b/code/cmsc848n_results/Qwen3-Coder-30B-A3B-Instruct.run_swebench_codeact_with_qwen3_coder_pass_1.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/Qwen3-Coder-30B-A3B-Instruct.run_swebench_codeact_with_qwen3_coder_pass_2.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/Qwen3-Coder-30B-A3B-Instruct.run_swebench_codeact_with_qwen3_coder_pass_3.json"
# ]

run_files = [
    "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.run_swebench_wo_agent_qwen3_coder_pass_1.json",
    "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.run_swebench_wo_agent_qwen3_coder_pass_2.json"]

res = calculate_ets_at_x(run_files)
print(f"ETS@{res['x_limit']}: {res['ETS_at_x']:.2f}")
print(res['interpretation'])
