import json

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
        

def put_labels(data_item):
    data_item["label"] = "Yes"
    return data_item

if __name__ == "__main__":
    input_file = "/scratch/zt1/project/cmsc848n/shared/hsoora/SWE-bench/swebench/inference/make_datasets/null_resolution/gt_fixed_patches_lite.json"
    output_file = "/scratch/zt1/project/cmsc848n/shared/hsoora/SWE-bench/swebench/inference/make_datasets/null_resolution/gt_fixed_patches_lite.json"
    
    data = read_json(input_file)
    
    labeled_data = []
    for item in data:
        labeled_data.append(put_labels(item))
        
    write_json(labeled_data, output_file)