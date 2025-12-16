import re
import difflib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple



def extract_code_block(prompt: str) -> str | None:
    """
    Extract the content inside the first <code>...</code> block
    from a full prompt string.

    Returns:
        The inner text (without the <code> tags), or None if not found.
    """
    # (?i) = case-insensitive, DOTALL = allow newlines
    m = re.search(r"(?i)<code>\s*(.*?)\s*</code>", prompt, re.DOTALL)
    if not m:
        return None
    return m.group(1)


# -------------------------------------------------------------
# 1. Parse SWE-bench-style <code> block into filename → lines
# -------------------------------------------------------------
def parse_repo_text(code_block: str) -> Dict[str, List[str]]:
    """
    Parse a SWE-bench style <code> block into a mapping:
        { "path/to/file.py": ["line1", "line2", ...], ... }

    Expects chunks like:
        [start of astropy/timeseries/core.py]
        1 # Licensed under ...
        2 ...
        [end of astropy/timeseries/core.py]
    """
    files: Dict[str, List[str]] = {}

    # Regex to capture [start of X] ... [end of X]
    pattern = re.compile(
        r"\[start of ([^\]]+)\](.*?)\[end of \1\]",
        re.DOTALL,
    )

    for match in pattern.finditer(code_block):
        filename = match.group(1).strip()
        numbered_block = match.group(2).strip("\n")

        lines = []
        for raw_line in numbered_block.splitlines():
            # Lines look like: "12 some code here"
            # Split off leading number and space
            raw_line = raw_line.rstrip("\n")
            if not raw_line:
                lines.append("")
                continue
            # Find first space after the number
            parts = raw_line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                content = parts[1]
            else:
                # Fallback: no number or malformed -> take whole line
                content = raw_line
            lines.append(content)

        files[filename] = lines

    return files


# -------------------------------------------------------------
# 2. Helpers for patch repair
# -------------------------------------------------------------
def context_line_matches(line: str, file_lines: List[str]) -> bool:
    """
    Check if an unchanged (context) line exists *exactly* in file_lines.
    """
    if line.strip() == "":
        # allow any blank line
        return True
    return any(line == l for l in file_lines)


def repair_hunk(hunk_lines: List[str], file_lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Given a list of lines belonging to a single hunk (excluding @@ header),
    repair malformed context lines by fuzzy-matching them to lines in file_lines.

    Returns:
        fixed_hunk_lines, list_of_error_messages
    """
    fixed: List[str] = []
    errors: List[str] = []

    if file_lines is None:
        # No file content → nothing we can do
        return hunk_lines, ["No file content available for this hunk."]

    for line in hunk_lines:
        if line.startswith(" "):  # context line
            content = line[1:]
            if not context_line_matches(content, file_lines):
                # Try fuzzy match to find closest line in the file
                candidates = difflib.get_close_matches(content, file_lines, n=1, cutoff=0.6)
                if candidates:
                    best = candidates[0]
                    fixed.append(" " + best)
                    errors.append(f"Context adjusted: '{content}' → '{best}'")
                else:
                    # We keep it, but record that it didn't match anything
                    fixed.append(line)
                    errors.append(f"Unmatched context line left as-is: '{content}'")
            else:
                fixed.append(line)
        else:
            # Added (+) or removed (-) lines are left untouched
            fixed.append(line)

    return fixed, errors


# -------------------------------------------------------------
# 3. Repair an entire patch using in-memory repo files
# -------------------------------------------------------------
def repair_patch_with_repo(patch_text: str,
                           repo_files: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    """
    Given a unified diff patch string and a mapping of
    filename -> list of lines (from parse_repo_text),
    repair malformed context lines in each hunk.

    Returns:
        fixed_patch_text, list_of_error_messages
    """
    output: List[str] = []
    errors_all: List[str] = []

    current_file: str = None
    file_lines: List[str] = None
    current_hunk: List[str] = []

    lines = patch_text.splitlines()

    for line in lines:
        # Start of a new file diff
        if line.startswith("diff --git"):
            # Flush previous hunk if any
            if current_hunk and file_lines is not None:
                fixed_hunk, errs = repair_hunk(current_hunk, file_lines)
                output.extend(fixed_hunk)
                errors_all.extend(errs)
                current_hunk = []

            output.append(line)

            # Parse filename from: diff --git a/path b/path
            try:
                parts = line.split()
                # e.g. ['diff', '--git', 'a/astropy/foo.py', 'b/astropy/foo.py']
                a_path = parts[2]
                # strip leading "a/"
                if a_path.startswith("a/"):
                    a_path = a_path[2:]
                current_file = a_path
                file_lines = repo_files.get(current_file)
                if file_lines is None:
                    errors_all.append(f"No file content found for '{current_file}' in repo_text.")
            except Exception as e:
                current_file = None
                file_lines = None
                errors_all.append(f"Failed to parse filename from line: {line} ({e})")

            continue

        # Hunk header
        if line.startswith("@@"):
            # Flush previous hunk
            if current_hunk and file_lines is not None:
                fixed_hunk, errs = repair_hunk(current_hunk, file_lines)
                output.extend(fixed_hunk)
                errors_all.extend(errs)
                current_hunk = []

            output.append(line)
            continue

        # Lines inside a hunk
        if line.startswith(" ") or line.startswith("+") or line.startswith("-"):
            current_hunk.append(line)
            continue

        # Other metadata lines (index, ---/+++ headers, etc.)
        # Just flush any existing hunk, then append
        if current_hunk and file_lines is not None:
            fixed_hunk, errs = repair_hunk(current_hunk, file_lines)
            output.extend(fixed_hunk)
            errors_all.extend(errs)
            current_hunk = []

        output.append(line)

    # Final flush at end of patch
    if current_hunk and file_lines is not None:
        fixed_hunk, errs = repair_hunk(current_hunk, file_lines)
        output.extend(fixed_hunk)
        errors_all.extend(errs)

    fixed_patch = "\n".join(output)
    return fixed_patch, errors_all

def read_json(path: str) -> dict:
    import json
    with open(path, 'r') as f:
        return json.load(f)
    
def repair_pred(data_item):
    gen_patch = data_item['model_patch']
    prompt = data_item['text']
    code_block = extract_code_block(prompt)
    repo_files = parse_repo_text(code_block)
    fixed_patch, errors = repair_patch_with_repo(gen_patch, repo_files)
    data_item['model_patch'] = fixed_patch
    data_item['repair_errors'] = errors
    return data_item

def write_json(path: str, data: dict):
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    # Example: you already have `code_block` (the big <code> ... </code> string)
    # and `patch_text` (the model's unified diff).
    #
    # Here we just sketch placeholders; you will plug in your actual strings.
    # code_block = """<put your [start of ...][end of ...] content here>"""
    # patch_text = """<put your model-generated diff here>"""

    # repo_files = parse_repo_text(code_block)
    # fixed_patch, errors = repair_patch_with_repo(patch_text, repo_files)

    # print("---- FIXED PATCH ----")
    # print(fixed_patch)
    # print("\n---- REPAIR LOG ----")
    # for e in errors:
    #     print(" *", e)
    
    items = read_json("/fs/gamma-projects/audio/V2A/vila_hd/llms/llms/dataset/trial_dataset_with_qwen3_coder_output.json")
    repaired_items = [repair_pred(item) for item in items]
    write_json("/fs/gamma-projects/audio/V2A/vila_hd/llms/llms/dataset/trial_dataset_with_qwen3_coder_output_repaired.json", repaired_items)
