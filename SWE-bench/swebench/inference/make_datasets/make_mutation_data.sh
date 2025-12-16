#!/bin/bash
python3 create_text_dataset.py \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --splits test \
    --validation_ratio 0.0 \
    --output_dir ./datasets/swebench_verified/mutation_data \
    --retrieval_file /scratch/zt1/project/cmsc848n/shared/hsoora/SWE-bench/swebench/inference/make_datasets/datasets/swebench_verified/oracle_retrieval.json