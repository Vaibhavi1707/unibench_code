import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------
model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# -------------------------------------------------------
# INPUT JSON PATH
# -------------------------------------------------------
input_json_path = "/scratch/zt1/project/cmsc848n/shared/hsoora/SWE-bench/swebench/inference/make_datasets/datasets/swebench_verified/SWE-bench_Verified__style-3__fs-oracle/test/data-00000-of-00001.json"

with open(input_json_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries.")

# -------------------------------------------------------
# BUILD CHAT MESSAGES WITH THINKING = TRUE
# -------------------------------------------------------
prompts = [item["text"] for item in data]

messages_batch = [
    [{"role": "user", "content": p}]
    for p in prompts
]

texts = [
    tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    for msg in messages_batch
]

# -------------------------------------------------------
# TOKENIZE BATCH
# -------------------------------------------------------
batch_inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=False
).to(model.device)

# -------------------------------------------------------
# GENERATE
# -------------------------------------------------------
generated = model.generate(
    **batch_inputs,
    max_new_tokens=32768
)

# -------------------------------------------------------
# PARSE THINKING + RESPONSE
# -------------------------------------------------------
THINK_END = 151668  # </think> token ID

results = []
for i, item in enumerate(data):
    # find true input length
    true_len = (batch_inputs.input_ids[i] != tokenizer.pad_token_id).sum().item()
    output_ids = generated[i][true_len:].tolist()

    # parse thinking
    try:
        rev_index = output_ids[::-1].index(THINK_END)
        think_end_idx = len(output_ids) - rev_index
    except ValueError:
        think_end_idx = 0

    thinking_text = tokenizer.decode(
        output_ids[:think_end_idx],
        skip_special_tokens=True
    ).strip("\n")

    result_text = tokenizer.decode(
        output_ids[think_end_idx:],
        skip_special_tokens=True
    ).strip("\n")

    # -------------------------------------------------------
    # EXTEND ORIGINAL ITEM WITH NEW FIELDS
    # -------------------------------------------------------
    new_item = item.copy()
    new_item["thinking"] = thinking_text
    new_item["model_output"] = result_text

    results.append(new_item)

# -------------------------------------------------------
# SAVE TO NEW JSON FILE
# -------------------------------------------------------
output_json_path = (
    Path(input_json_path).parent / "data-00000-of-00001__with_qwen_output.json"
)

with open(output_json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} annotated entries → {output_json_path}")
