#!/bin/bash

#platform=${1:-"mi300x"}
device_cnt=${device_cnt:-4}
dev_cnt_per_grp=${dev_cnt_per_grp:-4}
n_dev_grp=$((device_cnt/dev_cnt_per_grp))

enable_prof=${enable_prof:-0}
enable_trace=${enable_trace:-0}
enable_rpd=${enable_rpd:-0}
enable_tuneop=${enable_tuneop:-1}
enable_flash=${enable_flash:-0}
enable_fp8=${enable_fp8:-0}
debug_blas=${debug_blas:-0}
debug_nccl=${debug_nccl:-0}
max_iters=${max_iters:-10}
gbs_list=${gbs_list:-`echo 8 16 32 64 128`}

tune_gemm=${tune_gemm:-0}

benchmark_item=${benchmark_item:-"throughput"}

# throughput test
echo $benchmark_item
if [[ "${benchmark_item}" == "throughput" ]]; then
    i_tokens_list=(64 100)
    o_tokens_list=(64 128 200)
    num_prompts=${num_prompts:-1000}
else
    # latency test
    #i_tokens_list=(512 1024 2048 3072 4096 6144 8192)
    #o_tokens_list=(1 32 128 256)
    i_tokens_list=(4096)
    o_tokens_list=(4096)
    num_prompts=${num_prompts:-1}
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/log"}
PROF_DIR=${PROF_DIR:-"$SCRIPT_DIR/profiler"}
BLAS_DIR=${BLAS_DIR:-"$SCRIPT_DIR/blas"}

mkdir -p $LOG_DIR
mkdir -p $PROF_DIR
mkdir -p $BLAS_DIR

PROF=""
PROF_OPTIONS=""

ROCM=0
if command -v rocm-smi 1>/dev/null 2>&1; then
    ROCM=1
fi

if [[ "${ROCM}" -gt 0 ]]; then
    platform=${1:-"mi300x"}
else
    platform=${1:-"h100"}
fi

rocm_version=`cat /opt/rocm/.info/version`

gemm_tune_csv=${gemm_tune_csv:-"`pwd`/tunableop-config_${rocm_version}.csv"}

echo $gemm_tune_csv

export PYTORCH_TUNABLEOP_ENABLED=${enable_tuneop}
export PYTORCH_TUNABLEOP_TUNING=${tune_gemm}
export PYTORCH_TUNABLEOP_FILENAME="${gemm_tune_csv}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=1

if [[ ${enable_fp8} -eq 0 ]]; then
    TAG_PREFIX=vllm-benchmark-${benchmark_item}_grok_bf16_${platform}
else
    TAG_PREFIX=vllm-benchmark-${benchmark_item}_grok_f8_${platform}
fi

if [[ ${enable_prof} -eq 1 ]]; then
    if [[ "${ROCM}" -gt 0 ]]; then
        TAG_PREFIX+="_rocprof"
        PROF_OPTIONS+="--profile"
    else
        TAG_PREFIX+="_nsys"
    fi
fi

if [[ ${enable_flash} -eq 1 ]]; then
    TAG_PREFIX+="_fa"
    export VLLM_USE_TRITON_FLASH_ATTN=False
fi

if [[ ${enable_rpd} -eq 1 ]]; then
    TAG_PREFIX+="_rpd"
fi

if [[ ${debug_blas} -eq 1 ]]; then
    TAG_PREFIX+="_blas_dbg"
    #export ROCBLAS_LAYER=7
    #export HIPBLASLT_LOG_LEVEL=5
    export HIPBLASLT_LOG_MASK=32
fi

if [[ ${debug_nccl} -eq 1 ]]; then
    TAG_PREFIX+="_nccl_dbg"
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,COLL
fi
for i_token in "${i_tokens_list[@]}"; do
    for o_token in "${o_tokens_list[@]}"; do
        CURRENTDATE=`date +"%Y-%m-%d-%T"`
        for first_dev in `seq 0 ${dev_cnt_per_grp} $((device_cnt-1))`;do
            DEV_LIST=""
            for dev in `seq ${first_dev} $((first_dev + dev_cnt_per_grp - 1))`;do
                DEV_LIST="${DEV_LIST}${dev},"
            done

            TAG=${TAG_PREFIX}_gbs${num_prompts}_ilen${i_token}_olen${o_token}_dev${DEV_LIST}_${CURRENTDATE}
            if [[ ${enable_prof} -eq 1 ]]; then
                if [[ "${ROCM}" -gt 0 ]]; then
                    PROF="rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace --stats --output-format CSV -d $PROF_DIR/$TAG -- "
                    #PROF="rocprof --stats -o $PROF_DIR/$TAG.csv "
                else
                    PROF="nsys nvprof -o ${PROF_DIR}/nsys_${TAG} "
                fi
            fi

            if [[ ${debug_blas} -eq 1 ]]; then
                export ROCBLAS_LOG_PROFILE_PATH=${BLAS_DIR}/blas_cases_${TAG}.yaml
                export ROCBLAS_LOG_BENCH_PATH=${BLAS_DIR}/blas_cases_${TAG}_bench.log
            fi

            if [[ "${ROCM}" -gt 0 ]]; then
                #cmd="ROCR_VISIBLE_DEVICES=${DEV_LIST} HIP_FORCE_DEV_KERNARG=1 ${PROF}"
                cmd="HIP_FORCE_DEV_KERNARG=1 ${PROF}"
            else
                cmd="CUDA_VISIBLE_DEVICES=${DEV_LIST} ${PROF}"
            fi
            #cmd="${cmd} torchrun --standalone --nproc_per_node=${dev_cnt_per_grp}"
            #cmd="${cmd} benchmark_${benchmark_item}.py --model bigcode/starcoder2-15b --dtype float16"

            #cmd="${cmd} python benchmark_${benchmark_item}-v2.py --model bigcode/starcoder2-15b --dtype float16"
            #cmd="${cmd} python benchmark_${benchmark_item}-v2.py --model bigcode/starcoder2-15b --dtype float16 --max-num-batched-tokens 65536"
            #cmd="${cmd} python benchmark_${benchmark_item}-v2.py --model bigcode/starcoder2-15b --dtype float16 --max-model-len 512"
            #cmd="${cmd} python benchmark_${benchmark_item}-v2.py --model bigcode/starcoder2-15b --dtype float16 --max-model-len 512 --max-num-batched-tokens 65536"
            #cmd="${cmd} python benchmark_${benchmark_item}.py --model \"/dockerx/data/huggingface/hub/models--hpcai-tech--grok-1/snapshots/babfc31aa6cffb15aa89202aaa01ac47bd1925a2/\" --dtype float16 --trust-remote-code"
            #cmd="${cmd} python benchmark_${benchmark_item}.py --model hpcai-tech/grok-1 --dtype float16 --trust-remote-code"
	    if [[ ${enable_fp8} -eq 0 ]]; then
                cmd="${cmd} python benchmark_${benchmark_item}.py --model \"/dockerx/data/huggingface/hub/models--hpcai-tech--grok-1/snapshots/babfc31aa6cffb15aa89202aaa01ac47bd1925a2/\" --dtype float16 --trust-remote-code"
	    else
                cmd="${cmd} python benchmark_${benchmark_item}.py --model \"/dockerx/data/huggingface/hub/models--hpcai-tech--grok-1/snapshots/babfc31aa6cffb15aa89202aaa01ac47bd1925a2/\" --dtype float16 --trust-remote-code --quantization=\"fp8\" --quantized-weights-path=\"quantized/quark/w_fp8_a_fp8_o_fp8/num_calib_128/moe.safetensors\""
                #cmd="${cmd} python benchmark_${benchmark_item}.py --model \"/dockerx/data/huggingface/hub/models--hpcai-tech--grok-1/snapshots/babfc31aa6cffb15aa89202aaa01ac47bd1925a2/\" --dtype float16 --trust-remote-code --quantization=\"fp8\" --quantized-weights-path=\"quantized/quark/w_fp8_a_fp8_o_fp8/num_calib_128/moe.safetensors\" --kv-cache-dtype fp8 --quantization-param-path /dockerx/data/huggingface/hub/models--hpcai-tech--grok-1/snapshots/babfc31aa6cffb15aa89202aaa01ac47bd1925a2/quantized/quark/w_fp8_a_fp8_o_fp8/num_calib_128/kvcache/kv_cache_scales.json"
	    fi

	    # For llama
            #cmd="${cmd} python benchmark_${benchmark_item}.py --model \"/dockerx/data/llama2/llama-2-70b-chat-hf/\" --dtype float16"
            #cmd="${cmd} python benchmark_${benchmark_item}.py --model \"/dockerx/data/llama2/llama-2-70b-chat-hf/\" --dtype float16 --quantization=\"fp8\" --quantized-weights-path=\"quantized/llama.safetensors\""
            #cmd="${cmd} torchrun --standalone --nnodes=1 --nproc-per-node=${dev_cnt_per_grp} benchmark_${benchmark_item}.py --model \"/dockerx/data/mlperf/model/Llama-2-70b-chat-hf/\" --dtype float16 --quantization=\"fp8\" --quantized-weights-path=\"quantized/quark_share/modelzoo/llama2_70b_wfp8_afp8_ofp8_nomerge/json-safetensors/llama.safetensors\""

            if [[ ${enable_rpd} -eq 1 ]]; then
                cmd="${cmd} --enable-prof"
            fi

            if [[ "${benchmark_item}" == "throughput" ]]; then
                cmd="${cmd} --num-prompts ${num_prompts} --input-len ${i_token} --output-len ${o_token} -tp ${dev_cnt_per_grp} --worker-use-ray"
            else
                cmd="${cmd} --batch-size ${num_prompts} --input-len ${i_token} --output-len ${o_token} -tp ${dev_cnt_per_grp} --worker-use-ray"
            fi

            echo "${cmd}" | tee "${LOG_DIR}/r_${TAG}.log"
            if [[ ${debug_blas} -eq 1 || ${debug_nccl} -eq 1 ]]; then
                eval "${cmd}" >> ${LOG_DIR}/r_${TAG}.log 2>&1
            else
                eval "${cmd}" 2>&1 | tee -a "${LOG_DIR}/r_${TAG}.log"
            fi
        done
        act_python3=`ps -aux| grep "vllm-benchmark-" | sed 's/.*grep.*//g' | grep -e "\S" | wc -l`
        while [[ ${act_python3} -gt 0 ]]; do act_python3=`ps -aux| grep "vllm-benchmark-" | sed 's/.*grep.*//g' | grep -e "\S" | wc -l`; continue;done
        if [[ ${enable_rpd} -eq 1 ]]; then
            mv trace.rpd ${PROF_DIR}/r_${TAG}.rpd
        fi
    done
done
