import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
from tqdm import tqdm

from vllm.model_executor.layers.fused_moe import (fused_moe,
                                                  get_config_file_name)

from vllm import envs

padding_size = 128 if envs.VLLM_MOE_PADDING else 0

def permute_weight(x: torch.Tensor) -> torch.Tensor:
    ## Hardcode BLOCK_K and BLOCK_N

    BK = 256
    BN = 128
    x_ = x
    x_ = x_.view(x.shape[0],
                 x.shape[1]//BN, BN//16, 16,
                     x.shape[2]//BK, BK//64, 4, 16)
    x_ = x_.permute(0,1,5,2,6,4,3,7)
    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2]);
    return x_

def main(model, tp_size, gpu, dtype: str):
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu)
    method = fused_moe
    for bs in [8
           # 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
           # 2048, 3072, 4096
    ]:
        run_grid(bs,
                 model=model,
                 method=method,
                 gpu=gpu,
                 tp_size=tp_size,
                 dtype=dtype)


def run_grid(bs, model, method, gpu, tp_size, dtype: str):
    if model == '8x7B':
        d_model = 4096
        model_intermediate_size = 14336
        num_layers = 32
    elif model == '8x22B':
        d_model = 6144
        model_intermediate_size = 16384
        num_layers = 56
    elif model == 'grok1':
        d_model = 6144
        model_intermediate_size = 32768
        num_layers = 64
    else:
        raise ValueError(f'Unsupported Mixtral model {model}')
    num_total_experts = 8
    top_k = 2
    # tp_size = 2
    num_calls = 100

    num_warmup_trials = 1
    num_trials = 1

    configs = []

    waves_per_eu_range = [0]
    # Remove 32 because of triton compiling error
    matrix_instr_nonkdim_range = [16]
    kpack_range = [1, 2]

    for block_size_n in [32, 64, 128, 256]:
        for block_size_m in [16, 32, 64, 128, 256]:
            for block_size_k in [64, 128, 256]:
                for group_size_m in [1, 16, 32, 64]:
                    for num_warps in [4, 8]:
                        for num_stages in [2, 3, 4, 5]:
                            for waves_per_eu in waves_per_eu_range:
                                for (matrix_instr_nonkdim
                                     ) in matrix_instr_nonkdim_range:
                                    for kpack in kpack_range:
                                        configs.append({
                                            "BLOCK_SIZE_M": block_size_m,
                                            "BLOCK_SIZE_N": block_size_n,
                                            "BLOCK_SIZE_K": block_size_k,
                                            "GROUP_SIZE_M": group_size_m,
                                            "num_warps": num_warps,
                                            "num_stages": num_stages,
                                            "waves_per_eu": waves_per_eu,
                                            "matrix_instr_nonkdim":
                                            matrix_instr_nonkdim,
                                            "kpack": kpack,
                                        })

    best_config = None
    best_time_us = 1e20

    print(f'{tp_size=} {bs=}')

    for config in tqdm(configs):
        # warmup
        try:
            print(config)
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_calls,
                    bs=bs,
                    d_model=d_model,
                    num_total_experts=num_total_experts,
                    top_k=top_k,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    method=method,
                    config=config,
                    dtype=dtype,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # trial
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                num_total_experts=num_total_experts,
                top_k=top_k,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                method=method,
                config=config,
                dtype=dtype,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

                #print("Current BEST: BLOCK_SIZE_M=", block_size_m,
                #    " ,BLOCK_SIZE_N=", block_size_n,
                #    " ,BLOCK_SIZE_K=", block_size_k,
                #    " ,GROUP_SIZE_M=", group_size_m,
                #    " ,num_warps=", num_warps,
                #    " ,num_stages=", num_stages,
                #    " ,waves_per_eu=", waves_per_eu,
                #    " ,matrix_instr_nonkdim=",
                #    matrix_instr_nonkdim,
                #    " ,kpack=", kpack,
                #)

                tqdm.write(
                    f'{kernel_dur_us=:.1f} {model_dur_ms=:.1f}'
                    f' {bs=} {tp_size=} {top_k=} {num_total_experts=} '
                    f'{d_model=} {model_intermediate_size=} {num_layers=}')

    print("best_time_us", best_time_us)
    print("best_config", best_config)

    # holds Dict[str, Dict[str, int]]
    filename = get_config_file_name(num_total_experts,
                                    model_intermediate_size // tp_size,
                                    "float8" if dtype == "float8" else None)
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def run_timing(num_calls: int, bs: int, d_model: int, num_total_experts: int,
               top_k: int, tp_size: int, model_intermediate_size: int, method,
               config, dtype: str) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda:0",
        dtype=torch.float16,
    )

    w1 = torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model+padding_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2 = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size+padding_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None

    if dtype == "float8":
        w1 = w1.to(torch.float8_e4m3fnuz)
        w2 = w2.to(torch.float8_e4m3fnuz)
        w1_scale = torch.ones(num_total_experts,
                              device=hidden_states.device,
                              dtype=torch.float32)
        w2_scale = torch.ones(num_total_experts,
                              device=hidden_states.device,
                              dtype=torch.float32)
        a1_scale = torch.ones(1,
                              device=hidden_states.device,
                              dtype=torch.float32)
        a2_scale = torch.ones(1,
                              device=hidden_states.device,
                              dtype=torch.float32)

    

    gating_output = F.softmax(torch.rand(
        (num_calls, bs, num_total_experts),
        device=hidden_states.device,
        dtype=torch.float32,
    ), dim=-1)

    if envs.VLLM_MOE_SHUFFLE:
        w1_shuffled = permute_weight(w1.data)
        w2_shuffled = permute_weight(w2.data)
    else:
        w1_shuffled = w1
        w2_shuffled = w2

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        hidden_states = method(
            hidden_states=hidden_states,
            w1=w1_shuffled,
            w2=w2_shuffled,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            gating_output=gating_output[i],
            topk=2,
            renormalize=True,
            inplace=True,
            override_config=config,
            use_fp8=dtype == "float8",
        )
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='benchmark_mixtral_moe',
        description='Benchmark and tune the fused_moe kernel',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['float8', 'float16'],
        help='Data type used for fused_moe kernel computations',
    )
    parser.add_argument('--model',
                        type=str,
                        default='8x7B',
                        choices=['8x7B', '8x22B', 'grok1'],
                        help='The Mixtral model to benchmark')
    parser.add_argument('--tp-size',
                        type=int,
                        default=2,
                        help='Tensor paralleli size')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help="GPU ID for benchmarking")
    args = parser.parse_args()
    sys.exit(main(args.model, args.tp_size, args.gpu, args.dtype))
