#!/bin/bash
set -e

MODEL_PATH=$1
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_PATH")
PORT=${2:-8080}

echo "PORT=$PORT"
echo "MODEL_PATH=$MODEL_PATH"
echo "MODEL_DIR=$MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Create container cache dir — required for vLLM + TorchInductor
CACHE_DIR="$MODEL_DIR/.vllm_cache_vl"
mkdir -p "$CACHE_DIR"

echo "CACHE_DIR=$CACHE_DIR"

# Pull correct image name if not already present
if [ ! -f vllm-openai_latest.sif ]; then
    echo "Pulling vLLm Singularity image..."
    singularity pull vllm-openai_latest.sif docker://vllm/vllm-openai:latest
fi

# Make sure you started model server (vLLM or llama.cpp) and code execution engine before running this!


# Run vLLM
singularity exec --nv \
  --home "$CACHE_DIR" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --bind "$MODEL_DIR":/model \
  vllm-openai_latest.sif \
    python3 -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port "$PORT" \
      --model /model/"$MODEL_NAME" \
      --served-model-name "$MODEL_NAME"

