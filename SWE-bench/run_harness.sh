#!/bin/bash

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path /scratch/zt1/project/cmsc848n/shared/hsoora/swebench_verified_responses.json \
    --run_id run_codeact_gpt_5_nano_results