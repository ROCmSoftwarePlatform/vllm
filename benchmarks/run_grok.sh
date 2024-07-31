#!/bin/bash

#batches="1 2 4 8 16 32 64 128"
#export PYTORCH_TUNABLEOP_VERBOSE=true
batches="8"

export VLLM_FP8_REDUCE_CONV=1
export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export NCCL_MIN_NCHANNELS=112

for batch in $batches
do
    debug_blas=0 dev_cnt_per_grp=8 device_cnt=8 enable_fp8=1 enable_tuneop=0 benchmark_item=latency num_prompts=$batch ./benchmark_sweep.sh
done
