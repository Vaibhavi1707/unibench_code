import json
import itertools
from statistics import mean

def calculate_single_pass_at_k(result_files):
    """
    Helper: Calculates the metric for a specific SET of k files.
    """
    all_submitted_ids = set()
    all_resolved_ids = set()
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'submitted_ids' in data:
                all_submitted_ids.update(data['submitted_ids'])
            if 'resolved_ids' in data:
                all_resolved_ids.update(data['resolved_ids'])
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Issue reading {file_path}")

    total = len(all_submitted_ids)
    resolved = len(all_resolved_ids)
    return resolved / total if total > 0 else 0.0

def report_average_pass_at_k(all_file_paths):
    """
    Generates Pass@k stats for k = 1 to len(all_file_paths).
    """
    max_k = len(all_file_paths)
    final_report = {}

    print(f"--- Calculating Pass@k for k=1 to {max_k} ---")
    
    for k in range(1, max_k + 1):
        # Generate all unique combinations of size k
        # e.g., for k=2 with files [A, B, C], we get (A,B), (A,C), (B,C)
        combinations = list(itertools.combinations(all_file_paths, k))
        
        scores = []
        for combo in combinations:
            score = calculate_single_pass_at_k(combo)
            scores.append(score)
        print(f"Computed {scores}  for k={k}")
        avg_score = mean(scores)
        
        final_report[k] = {
            "average_pass_rate": avg_score,
            "min_pass_rate": min(scores),
            "max_pass_rate": max(scores),
            "num_combinations": len(scores)
        }
        
        print(f"k={k}: {avg_score:.2%} (Avg over {len(scores)} configs)")

    return final_report

# --- usage ---
# all_run_files = [
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_1.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_2.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_3.json"
# ]
# all_run_files = [
#     "/Users/aryan.vi.b/code/cmsc848n_results/Qwen3-Coder-30B-A3B-Instruct.run_swebench_codeact_with_qwen3_coder_pass_1.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/Qwen3-Coder-30B-A3B-Instruct.run_swebench_codeact_with_qwen3_coder_pass_2.json",
#     "/Users/aryan.vi.b/code/cmsc848n_results/Qwen3-Coder-30B-A3B-Instruct.run_swebench_codeact_with_qwen3_coder_pass_3.json"
# ]
all_run_files = [
    "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.run_swebench_wo_agent_qwen3_coder_pass_1.json",
    "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.run_swebench_wo_agent_qwen3_coder_pass_2.json"]


# Run the calculation
stats = report_average_pass_at_k(all_run_files)

# Access specific results if needed
# print(f"Final Pass@1: {stats[1]['average_pass_rate']}")


# import json
# from pathlib import Path

# def calculate_pass_at_k_metric(result_files):
#     """
#     Calculates the ratio of unique issues resolved at least once across k runs.
    
#     Args:
#         result_files (list): A list of file paths (str or Path) to the k JSON result files.
        
#     Returns:
#         dict: A dictionary containing the metric and detailed stats.
#     """
    
#     # Sets to track unique issues across all k runs
#     all_submitted_ids = set()
#     all_resolved_ids = set()
    
#     # Iterate through each run file
#     for file_path in result_files:
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
                
#             # Update the set of all unique issues attempted (submitted)
#             # We assume 'submitted_ids' tracks every issue the agent tried to solve in this run
#             if 'submitted_ids' in data:
#                 all_submitted_ids.update(data['submitted_ids'])
            
#             # Update the set of resolved issues
#             # If an issue is in this list for ANY run, it counts as resolved
#             if 'resolved_ids' in data:
#                 all_resolved_ids.update(data['resolved_ids'])
                
#         except FileNotFoundError:
#             print(f"Warning: File not found: {file_path}")
#         except json.JSONDecodeError:
#             print(f"Warning: Invalid JSON in file: {file_path}")

#     # Calculate the metric
#     total_issues = len(all_submitted_ids)
#     resolved_count = len(all_resolved_ids)
    
#     metric = 0.0
#     if total_issues > 0:
#         metric = resolved_count / total_issues

#     return {
#         "pass_at_any_k_metric": metric,
#         "total_unique_issues_attempted": total_issues,
#         "total_unique_issues_resolved": resolved_count,
#         "resolved_ids": sorted(list(all_resolved_ids))
#     }

# # --- Usage Example ---

# # 1. List your k result files
# # Replace these with your actual filenames
# run_files = [
#     "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_1.json",
#     # "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_2.json",
#     # "/Users/aryan.vi.b/code/cmsc848n_results/openai__gpt-5-nano.pass_3.json"
# ]
# k = len(run_files)
# # 2. Calculate
# # (Ensure the files exist before running this snippet)
# stats = calculate_pass_at_k_metric(run_files)

# print(f"pass_at_any_{k}_metric Metric: {stats['pass_at_any_k_metric']:.2%}")
# print(f"Resolved: {stats['total_unique_issues_resolved']} / {stats['total_unique_issues_attempted']}")
