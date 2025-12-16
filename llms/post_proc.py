import re

def extract_patch_blocks(text):
    """
    Extracts all blocks that appear between ```patch and ``` delimiters.

    Returns:
        List[str]: a list of extracted patch blocks (strings).
    """
    # Regex explanation:
    # ```patch      literal start
    # (.*?)         non-greedy capture of everything inside
    # ```           literal ending triple-backtick
    # Flags: DOTALL so . matches newlines
    pattern = r"```patch\s*(.*?)\s*```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches

def read_json(file_path):
    import json
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_json(file_path, data):
    import json
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
        
def get_patch_from_item(item):
    """
    Given an item (dict) with a 'text' field, extract the first patch block.

    Returns:
        str or None: the extracted patch block, or None if not found.
    """
    text = item.get('model_patch', '')
    patches = extract_patch_blocks(text)
    if patches:
        return patches[0].strip("\n")
    return None


if __name__ == "__main__":
    import json
    from pathlib import Path

    input_json_path = "/fs/gamma-projects/audio/V2A/vila_hd/llms/llms/dataset/data-00000-of-00001_qwen3_coder_output.json"
    output_json_path = (
        Path(input_json_path).parent / "data-00000-of-00001_qwen3_coder_extracted_patches_v1.json"
    )

    data = read_json(input_json_path)
    print(f"Loaded {len(data)} entries.")

    results = []
    for item in data:
        patch = get_patch_from_item(item)
        if patch is None:
            patch = ""
        new_item = item.copy()
        new_item["model_patch"] = "diff --git " + patch
        new_item["model_name_or_path"] = item.get("model", "")
        results.append(new_item)

    write_json(output_json_path, results)
    print(f"Wrote output with extracted patches to {output_json_path}")