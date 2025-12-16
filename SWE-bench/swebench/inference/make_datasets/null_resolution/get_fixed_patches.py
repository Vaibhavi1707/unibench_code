import json

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def add_lines_list(content):
    content_with_lines = list()
    for ix, line in enumerate(content.split("\n"), start=1):
        content_with_lines.append(f"{ix} {line}")
    return content_with_lines


def add_lines(content):
    return "\n".join(add_lines_list(content))


def make_code_text(files_dict, add_line_numbers=False):
    all_text = ""
    for filename, contents in sorted(files_dict.items()):
        all_text += f"[start of {filename}]\n"
        if add_line_numbers:
            all_text += add_lines(contents)
        else:
            all_text += contents
        all_text += f"\n[end of {filename}]\n"
    return all_text.strip("\n")
 

def get_full_code(data_item):
    return make_code_text(data_item["file_contents"])

def get_fixed_patch(data_item):
    return data_item["patch"]

import re

def apply_unified_diff(original_text, diff_text):
    orig = original_text.splitlines(keepends=True)
    new = []
    i = 0  # pointer in original file

    diff_lines = diff_text.splitlines()

    # Find file hunks
    hunk_re = re.compile(r'^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')

    ptr = 0
    while ptr < len(diff_lines):
        line = diff_lines[ptr]

        # Ignore metadata
        if line.startswith("diff ") or line.startswith("---") or line.startswith("+++"):
            ptr += 1
            continue

        m = hunk_re.match(line)
        if m:
            start_old = int(m.group(1)) - 1   # convert to 0-index
            ptr += 1

            # Copy unchanged lines BEFORE hunk
            while i < start_old:
                new.append(orig[i])
                i += 1

            # Process hunk lines
            while ptr < len(diff_lines) and not diff_lines[ptr].startswith("@@"):
                hline = diff_lines[ptr]
                if hline.startswith(" "):
                    new.append(orig[i])
                    i += 1
                elif hline.startswith("-"):
                    i += 1
                elif hline.startswith("+"):
                    new.append(hline[1:] + "\n")
                ptr += 1

            continue

        ptr += 1

    # copy any remaining original lines
    while i < len(orig):
        new.append(orig[i])
        i += 1

    return "".join(new)



def apply_patch_to_code(code, patch):
    code_lines = code.splitlines()
    patch_lines = patch.splitlines()
    
    for line in patch_lines:
        if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
            continue
        elif line.startswith('+'):
            code_lines.append(line[1:])
        elif line.startswith('-'):
            print(line[1:])
            code_lines.remove(line[1:])
    
    return "\n".join(code_lines)

import subprocess
import tempfile
import os

def apply_patch_with_unix_patch(repo_path, patch_text):
    """
    Apply a unified diff patch to a repo using the Unix `patch` command.

    Parameters:
        repo_path (str): Path to the repository root.
        patch_text (str): The unified diff patch as a string.

    Returns:
        success (bool): True if patch applied cleanly, False otherwise.
        output (str): Combined stdout/stderr from the patch command.
    """

    # Create a temporary patch file
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".patch") as tmp:
        tmp.write(patch_text)
        tmp_path = tmp.name

    try:
        # Run UNIX patch command exactly as SWE-bench does (with -p1)
        result = subprocess.run(
            ["patch", "-p1", "-i", tmp_path],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        success = (result.returncode == 0)
        return success, result.stdout

    finally:
        # Clean up
        os.remove(tmp_path)



def get_fixed_patches(input_jsonl, output_json):
    data = read_jsonl_file(input_jsonl)
    fixed_data = []
    
    for item in data:
        full_code = get_full_code(item)
        patch = get_fixed_patch(item)
        # print("Full code:", full_code)
        # fixed_code = apply_patch_to_code(full_code, patch)
        fixed_code = apply_unified_diff(full_code, patch)
        
        fixed_data.append({
            **item,
            # "full_code": full_code,
            # "patch": patch,
            "fixed_code": fixed_code
        })
    
    write_json(fixed_data, output_json) 
    
get_fixed_patches(
    input_jsonl="/scratch/zt1/project/cmsc848n/shared/hsoora/SWE-bench/swebench/inference/make_datasets/datasets/swebench_verified/SWE-bench_Lite__patch-only__fs-oracle.test.progress.jsonl",
    output_json= "gt_fixed_patches_lite.json"
)