import json
import math
import itertools
from statistics import mean
from collections import defaultdict

def calculate_single_er_score(file_paths):
    """
    Calculates ER score for a specific set of files.
    """
    x = len(file_paths)
    if x == 0: return 0.0
    
    # 1. Aggregate Data
    issue_y = defaultdict(int)
    all_submitted = set()
    
    for fp in file_paths:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            if 'submitted_ids' in data:
                all_submitted.update(data['submitted_ids'])
            if 'resolved_ids' in data:
                for rid in data['resolved_ids']:
                    issue_y[rid] += 1
        except:
            pass
            
    total = len(all_submitted)
    if total == 0: return 0.0
    
    # 2. Calculate Log Score
    log_denom = math.log(1 + x)
    sum_score = 0.0
    
    for iid in all_submitted:
        y = issue_y[iid]
        sum_score += math.log(1 + y) / log_denom
        
    return sum_score / total

def explain_reliability_tiers(all_files):
    """
    Generates the interpretable Tiered Breakdown for the full run (x=3).
    """
    x = len(all_files)
    issue_y = defaultdict(int)
    all_submitted = set()
    
    for fp in all_files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            if 'submitted_ids' in data:
                all_submitted.update(data['submitted_ids'])
            if 'resolved_ids' in data:
                for rid in data['resolved_ids']:
                    issue_y[rid] += 1
        except: pass
    
    total = len(all_submitted)
    if total == 0: return
    
    # Tiers
    failed = 0
    flaky = 0
    robust = 0
    
    for iid in all_submitted:
        y = issue_y[iid]
        if y == 0:
            failed += 1
        elif y == 1:
            flaky += 1
        else: # y >= 2
            robust += 1
            
    print(f"\n--- Reliability Tiers (x={x}) ---")
    print(f"Total Issues: {total}")
    print(f"🔴 Failed (0/{x}): {failed} ({failed/total:.1%})")
    print(f"🟡 Flaky  (1/{x}): {flaky} ({flaky/total:.1%})  <-- Likely luck")
    print(f"🟢 Robust (2+/{x}): {robust} ({robust/total:.1%}) <-- True capability")
    
    # Weighted Score
    wrs = (1.0 * (robust/total)) + (0.5 * (flaky/total))
    print(f"Weighted Reliability Score: {wrs:.2f} (Scale 0-1)")

def calculate_averaged_er_at_k(all_files):
    """
    Calculates ER@1, ER@2, ... ER@x by averaging combinations.
    """
    x_max = len(all_files)
    results = {}
    
    print(f"\n--- Averaged ER@k Metrics ---")
    
    for k in range(1, x_max + 1):
        combinations = list(itertools.combinations(all_files, k))
        scores = []
        
        for combo in combinations:
            score = calculate_single_er_score(combo)
            scores.append(score)
            
        avg_score = mean(scores)
        results[k] = avg_score
        print(f"ER@{k}: {avg_score:.4f}")
        
    return results

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

# 1. Get the metrics curve
calculate_averaged_er_at_k(run_files)

# 2. Get the explanation
explain_reliability_tiers(run_files)


# import json
# import math
# from collections import defaultdict

# def calculate_er_at_x(result_files):
#     """
#     Calculates the ER@x (Log-Robustness) metric.
    
#     Formula: Average of [ log(1 + successes) / log(1 + total_tries) ]
    
#     Args:
#         result_files (list): List of paths to the k (or x) result JSON files.
        
#     Returns:
#         dict: The metric and component stats.
#     """
    
#     # x: Maximum amount of tries allowed (number of run files)
#     x = len(result_files)
    
#     if x == 0:
#         return {"er_at_x": 0.0, "error": "No files provided"}

#     # Track how many times each issue was resolved (y)
#     issue_resolution_counts = defaultdict(int)
    
#     # Track the universe of all unique issues attempted (N)
#     all_submitted_ids = set()
    
#     print(f"Processing x={x} runs...")

#     for file_path in result_files:
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
            
#             # 1. Build the denominator (Total universe of issues)
#             if 'submitted_ids' in data:
#                 all_submitted_ids.update(data['submitted_ids'])
            
#             # 2. Build the numerator (y counts per issue)
#             if 'resolved_ids' in data:
#                 for issue_id in data['resolved_ids']:
#                     issue_resolution_counts[issue_id] += 1
                    
#         except (FileNotFoundError, json.JSONDecodeError) as e:
#             print(f"Warning: Could not process {file_path}: {e}")

#     # Calculate Metric
#     total_issues_n = len(all_submitted_ids)
#     sum_log_scores = 0.0
    
#     # Pre-calculate log(1+x) for normalization
#     log_denominator = math.log(1 + x)
    
#     # Iterate over ALL attempted issues (even those with 0 success)
#     for issue_id in all_submitted_ids:
#         y = issue_resolution_counts[issue_id] # Default is 0 if not in dict
        
#         # S_i = log(1 + y) / log(1 + x)
#         issue_score = math.log(1 + y) / log_denominator
#         sum_log_scores += issue_score

#     # Final Average
#     er_metric = sum_log_scores / total_issues_n if total_issues_n > 0 else 0.0

#     return {
#         "ER_at_x_metric": er_metric,
#         "x_max_tries": x,
#         "total_issues_N": total_issues_n,
#         "distribution": dict(issue_resolution_counts) # Useful to see which issues are 'robust'
#     }

# # --- Usage ---

# run_files = [
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_1.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_2.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_3.json"
# ]

# stats = calculate_er_at_x(run_files)

# print(f"Configuration: x = {stats['x_max_tries']}")
# print(f"Total Issues (N): {stats['total_issues_N']}")
# print(f"ER@{stats['x_max_tries']} Score: {stats['ER_at_x_metric']:.4f}")

# # Example interpretation
# # If score is 0.85, it means your agent is highly robust (often solving issues 2/3 or 3/3 times).
# # If score is 0.30, it means your agent is fragile (solving issues 1/3 times mostly).
