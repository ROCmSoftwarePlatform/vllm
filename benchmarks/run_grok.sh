#!/bin/bash

#batches="1 2 4 8 16 32 64 128"
#export PYTORCH_TUNABLEOP_VERBOSE=true
batches="8"

#VLLM_FP8_REDUCE_CONV VLLM_FP8_ACT_PADDING VLLM_TUNE_GEMM VLLM_FP8_WEIGHT_PADDING VLLM_UNTUNE_FILE
#export TUNE_FP8=1

#export TORCH_BLAS_PREFER_HIPBLASLT=1

export PATH=/docker/home/kk/ws/vllm-mlperf/benchmarks/nccl/rccl-perf-coll/mpich/install/bin:$PATH
export LD_LIBRARY_PATH=/docker/home/kk/ws/vllm-mlperf/benchmarks/nccl/rccl-perf-coll/rccl/build/release/lib:/docker/home/kk/ws/vllm-mlperf/benchmarks/nccl/rccl-perf-coll/mpich/install/lib:$LD_LIBRARY_PATH

export VLLM_FP8_REDUCE_CONV=1
#export VLLM_FP8_ACT_PADDING=1
export VLLM_FP8_WEIGHT_PADDING=1
export NCCL_MIN_NCHANNELS=112
export VLLM_USE_TRITON_FLASH_ATTN=False

export FUSED_MOE_PERSISTENT=1 
export VLLM_MOE_PADDING=1 
export VLLM_MOE_SHUFFLE=1 
export TRITON_HIP_USE_NEW_STREAM_PIPELINE=1

for batch in $batches
do
    dev_cnt_per_grp=8 device_cnt=8 enable_fp8=1 enable_tuneop=0 benchmark_item=latency num_prompts=$batch ./benchmark_sweep.sh
    #debug_blas=0 dev_cnt_per_grp=2 device_cnt=2 enable_fp8=1 enable_tuneop=0 benchmark_item=latency+accuracy_check num_prompts=$batch ./benchmark_sweep.sh
done
