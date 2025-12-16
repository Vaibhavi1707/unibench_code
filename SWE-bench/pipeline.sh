#!/bin/bash

model_load_loc=$1
model=$2
agent=$3
INFERENCE_DIR="TMP/inference"
RESULTS_DIR="TMP/results"

if [ ! -d "$INFERENCE_DIR" ]; then
  mkdir -p $INFERENCE_DIR
fi

if [ ! -d "$RESULTS_DIR" ]; then
  mkdir -p $RESULTS_DIR
fi

if [ -f "$RESULTS_DIR/$agent-$model.json" ] ; then
  echo "Results already exist. Skipping inference."
  exit 0
fi

if [[ "$model" != "*gpt*" ]] && [[ "$model_load_loc" == "remote" ]]; then
    echo "Only gpt models are supported for remote loading."
    exit 1
fi

if [ "$model_load_loc" == "local" ]; then
  ./start_vllm_server.sh $2
    sleep 10
echo "VLLM server started."

if [ "$agent" == "mini-swe-agent" ]; then
    echo "Running mini-swe-agent with model $model"
    ./run_mini_swe_agent.sh $model lite

else if [[ "$agent" == "code-act" ]]; then
    echo "Running code-act agent with model $model"
    ./run_code_act_agent.sh $model lite

else if [[ "$agent" == "" ]]; then
    echo "Running model $model without agent"
    ./run_no_agent.sh $model lite

else
    echo "Agent not supported for remote models."
    exit 1
fi
fi
fi

echo "Inference done. Running evaluation harness on results."
./run_harness.sh $INFERENCE_DIR/$agent-$model.json

echo "Running harness complete. Run aggregate results to see summary."
python3 aggregate_results.py --input_file $RESULTS_DIR/$agent-$model.jsonl