import json

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

        
def get_patch(data_item):
    return data_item["model_patch"]


def get_ground_truth_patch(data_item):
    return data_item["patch"]


def extract_change_line_numbers(patch_text):
    changed_lines = []
    for line in patch_text.splitlines():
        if line.startswith('@@'):
            # Example hunk line: @@ -1,3 +1,4 @@
            parts = line.split(' ')
            new_file_part = parts[1]  # +1,4
            start_line = int(new_file_part.split(',')[0][1:])  # remove '+' and convert to int
            line_count = int(new_file_part.split(',')[1]) if ',' in new_file_part else 1
            changed_lines.append((start_line, start_line + line_count))
    return changed_lines


def is_overlap(range1, range2):
    return max(range1[0], range2[0]) < min(range1[1], range2[1])


def calculate_iou(range1, range2):
    intersection_start = max(range1[0], range2[0])
    intersection_end = min(range1[1], range2[1])
    intersection = max(0, intersection_end - intersection_start)
    
    union_start = min(range1[0], range2[0])
    union_end = max(range1[1], range2[1])
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    return intersection / union


def count_overlap_lines(pred_change_lines, gt_change_lines):
    overlap_cnt = 0
    start_range_diff = 0
    end_range_diff = 0
    
    for idx in range(len(gt_change_lines)):
        for pred_idx in range(len(pred_change_lines)):
            if is_overlap(pred_change_lines[pred_idx], gt_change_lines[idx]):
                overlap_cnt += 1
                iou = calculate_iou(pred_change_lines[pred_idx], gt_change_lines[idx])
                start_range_diff += abs(pred_change_lines[pred_idx][0] - gt_change_lines[idx][0])
                end_range_diff += abs(pred_change_lines[pred_idx][1] - gt_change_lines[idx][1])
            else:
                iou = 0.0
                start_range_diff += abs(pred_change_lines[pred_idx][0] - gt_change_lines[idx][0])
                end_range_diff += abs(pred_change_lines[pred_idx][1] - gt_change_lines[idx][1])
    return {
        "overlap_count": overlap_cnt,
        "start_range_diff": start_range_diff / len(gt_change_lines) if len(gt_change_lines) > 0 else 0,
        "end_range_diff": end_range_diff / len(gt_change_lines) if len(gt_change_lines) > 0 else 0,
        "total_gt_changes": len(gt_change_lines),
        "total_pred_changes": len(pred_change_lines),
        "average_iou": iou / overlap_cnt if overlap_cnt > 0 else 0.0
    }
    
    
if __name__ == "__main__":
    # input_file = '/scratch/zt1/project/cmsc848n/shared/hsoora/minisweagent_gpt5nano.json'
    input_file = "/scratch/zt1/project/cmsc848n/shared/hsoora/qwen3coder.json"
    output_file = 'fault_line_localization_results_qwen3coder.json'
    
    data = read_json(input_file)
    results = []
    avg_iou = 0.0
    n_overlaps = 0
    for item in data:
        pred_patch = get_patch(item)
        gt_patch = get_ground_truth_patch(item)
        
        pred_change_lines = extract_change_line_numbers(pred_patch)
        # print(f"Predicted change lines for instance {item['instance_id']}: {pred_change_lines}")
        gt_change_lines = extract_change_line_numbers(gt_patch)
        # print(f"Ground truth change lines for instance {item['instance_id']}: {gt_change_lines}")
        
        overlap_info = count_overlap_lines(pred_change_lines, gt_change_lines)
        avg_iou += overlap_info["average_iou"]
        n_overlaps += overlap_info["overlap_count"]
        
        result_item = {
            "id": item["instance_id"],
            "pred_change_lines": pred_change_lines,
            "gt_change_lines": gt_change_lines,
            "overlap_info": overlap_info
        }
        results.append(result_item)
    avg_iou /= len(data) if len(data) > 0 else 1
    print(f"Number of overlaps across all instances: {n_overlaps}")
    avg_overlap = n_overlaps / len(data) if len(data) > 0 else 0
    print(f"Average number of overlaps per instance: {avg_overlap}")
    print(f"Average IoU across all instances: {avg_iou}")
    write_json(results, output_file)