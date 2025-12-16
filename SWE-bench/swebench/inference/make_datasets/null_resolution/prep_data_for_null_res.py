import json
from tqdm import tqdm

with open('null_res_prompt.txt', 'r') as f:
    PROMPT = f.read()

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
        
def generate_data_instances(data):
    instances = []
    for item in tqdm(data):
        issue_description = item['problem_statement']
        fixed_code = item['fixed_code']
        
        prompt = PROMPT.format(issue_description=issue_description, 
                               fixed_code=fixed_code)
        
        item["text_inputs"] = prompt
    return data


if __name__ == "__main__":
    input_file = 'gt_fixed_patches.json'
    output_file = 'null_data.json'
    
    data = read_json(input_file)
    instances = generate_data_instances(data)
    write_json(instances, output_file)