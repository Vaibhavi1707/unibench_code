export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENAI_API_KEY="helloworld"
export OPENAI_API_BASE="http://gpu-b11-5:8080/v1"
python3 demo_$1.py --model_name Qwen3-Coder-30B-A3B-Instruct --openai_base_url $OPENAI_API_BASE  --jupyter_kernel_url http://gpu-b11-5:8081/execute
