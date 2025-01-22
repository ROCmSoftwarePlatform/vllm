#!/bin/bash

export FUSED_MOE_PERSISTENT=1 
export VLLM_MOE_PADDING=128 
export VLLM_MOE_SHUFFLE=1 
export TRITON_HIP_USE_NEW_STREAM_PIPELINE=1 
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

## ---- Mixtral fp8 tuning ---- ##

python benchmark_moe.py --model /data/Mixtral-8x7B-Instruct-v0.1-FP8/  -tp 1 --dtype fp8_w8a8 --tune
python benchmark_moe.py --model /data/Mixtral-8x7B-Instruct-v0.1-FP8/  -tp 2 --dtype fp8_w8a8 --tune
python benchmark_moe.py --model /data/Mixtral-8x7B-Instruct-v0.1-FP8/  -tp 4 --dtype fp8_w8a8 --tune
python benchmark_moe.py --model /data/Mixtral-8x7B-Instruct-v0.1-FP8/  -tp 8 --dtype fp8_w8a8 --tune

python benchmark_moe.py --model /data/Mixtral-8x22B-Instruct-v0.1-FP8/  -tp 1 --dtype fp8_w8a8 --tune
python benchmark_moe.py --model /data/Mixtral-8x22B-Instruct-v0.1-FP8/  -tp 2 --dtype fp8_w8a8 --tune
python benchmark_moe.py --model /data/Mixtral-8x22B-Instruct-v0.1-FP8/  -tp 4 --dtype fp8_w8a8 --tune
python benchmark_moe.py --model /data/Mixtral-8x22B-Instruct-v0.1-FP8/  -tp 8 --dtype fp8_w8a8 --tune


## ---- Mixtral fp16 tuning ---- ##

python benchmark_moe.py --model /data/AI-ModelScope/Mixtral-8x7B-Instruct-v0___1/  -tp 1  --tune
python benchmark_moe.py --model /data/AI-ModelScope/Mixtral-8x7B-Instruct-v0___1/  -tp 2  --tune
python benchmark_moe.py --model /data/AI-ModelScope/Mixtral-8x7B-Instruct-v0___1/  -tp 4  --tune
python benchmark_moe.py --model /data/AI-ModelScope/Mixtral-8x7B-Instruct-v0___1/  -tp 8  --tune

python benchmark_moe.py --model /data/huggingFace/Mixtral-8x22B-v0.1/  -tp 1  --tune
python benchmark_moe.py --model /data/huggingFace/Mixtral-8x22B-v0.1/  -tp 2  --tune
python benchmark_moe.py --model /data/huggingFace/Mixtral-8x22B-v0.1/  -tp 4  --tune
python benchmark_moe.py --model /data/huggingFace/Mixtral-8x22B-v0.1/  -tp 8  --tune
 
