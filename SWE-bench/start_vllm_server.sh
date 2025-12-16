#!/bin/bash
vllm serve $1 \
  --tensor-parallel-size 4