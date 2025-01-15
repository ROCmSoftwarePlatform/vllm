import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import (get_config_file_name,
                                                  invoke_fused_moe_kernel,
                                                  moe_align_block_size)


def main(args):
    os.environ["HIP_VISIBLE_DEVICES"] = args.GPUID

    for bs in [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            8192,
            16384,
            18432,
            20480,
    ]:
        run_grid(bs, model=args.model, TP=args.TP)


## Utilize method from rocm/Triton tuning script
def get_full_tuning_space():
    configs = []

    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [128]
    block_k_range = [128]
    # split_k_range = [1] #, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [8]
    group_m_range = [1]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [0]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16]
    kpack_range = [2]

    for block_m in block_m_range:
        for block_n in block_n_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        # for split_k in split_k_range:
                        for num_stages in num_stage_range:
                            for waves_per_eu in waves_per_eu_range:
                                for (matrix_instr_nonkdim
                                     ) in matrix_instr_nonkdim_range:
                                    for kpack in kpack_range:
                                        configs.append({
                                            "BLOCK_SIZE_M": block_m,
                                            "BLOCK_SIZE_N": block_n,
                                            "BLOCK_SIZE_K": block_k,
                                            "GROUP_SIZE_M": group_m,
                                            "num_warps": num_warps,
                                            "num_stages": num_stages,
                                            "waves_per_eu": waves_per_eu,
                                            "matrix_instr_nonkdim":
                                            matrix_instr_nonkdim,
                                            "kpack": kpack,
                                        })

    return configs


## Utilize method from rocm/Triton tuning script
def prune_configs(M, N, K, configs):
    return configs


def union_of_list_of_dicts(l1, l2):
    result = []
    temp_list = l1.copy()
    temp_list.extend(l2)
    for myDict in temp_list:
        if myDict not in result:
            result.append(myDict)

    return result


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def run_grid(bs, model, TP):
    if model == '8x7B':
        d_model = 4096
        model_intermediate_size = 14336
    elif model == '8x22B':
        d_model = 6144
        model_intermediate_size = 16384
    else:
        raise ValueError(f'Unsupported Mixtral model {model}')

    num_total_experts = 8
    top_k = 2
    tp_size = TP
    num_calls = 100

    num_warmup_trials = 1
    num_trials = 1

    full_configs = get_full_tuning_space()
    M1 = bs * 2
    N1 = model_intermediate_size * 2 // tp_size
    K1 = d_model
    prune_configs_1 = prune_configs(M1, N1, K1, full_configs)

    M2 = bs * 2
    N2 = d_model
    K2 = model_intermediate_size // tp_size
    prune_configs_2 = prune_configs(M2, N2, K2, full_configs)

    configs = union_of_list_of_dicts(prune_configs_1, prune_configs_2)
    print(f"{bs=} || {len(full_configs)=} | {len(prune_configs_1)=} | \
            {len(prune_configs_2)=} | {len(configs)=}")

    best_config = None
    best_time_us = 1e20

    for config in tqdm(configs):
        # warmup
        try:
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_calls,
                    bs=bs,
                    d_model=d_model,
                    num_total_experts=num_total_experts,
                    top_k=top_k,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    config=config,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # benchmark
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                num_total_experts=num_total_experts,
                top_k=top_k,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                config=config,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            # model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

            # print(f'{kernel_dur_us=:.1f} {model_dur_ms=:.1f}'
            #       f' {bs=} {tp_size=} {top_k=} {num_total_experts=} '
            #       f'{d_model=} {model_intermediate_size=} {num_layers=}')

    # print("best_time_us", best_time_us)
    # print("best_config", best_config)

    # holds Dict[str, Dict[str, int]]
    filename = get_config_file_name(num_total_experts,
                                    model_intermediate_size // tp_size,
                                    dtype=None)
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename) as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def run_timing(
    num_calls: int,
    bs: int,
    d_model: int,
    num_total_experts: int,
    top_k: int,
    tp_size: int,
    model_intermediate_size: int,
    config,
) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda",
        dtype=torch.float16,
    )

    w1 = torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model + 128),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2 = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gating_output = F.softmax(
        torch.rand(
            # (num_calls, bs, num_total_experts), # THIS
            (bs, num_total_experts),
            device=hidden_states.device,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    ###### Stuff from fused moe ######

    assert (hidden_states.shape[0] == gating_output.shape[0]
            ), "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[2] - 128, "Hidden size mismatch"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    M, _ = hidden_states.shape
    E, N, _ = w1.shape
    topk_ = 2
    topk_weights = torch.empty(M,
                               topk_,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk_,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk_,
                                        dtype=torch.int32,
                                        device=hidden_states.device)
    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    intermediate_cache1 = torch.empty(
        (M, topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)

    ##################################

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        invoke_fused_moe_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            None,  # a1_scale
            None,  # w1_scale
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk_ids.shape[1],
            config,
            compute_type=(tl.bfloat16 if hidden_states.dtype == torch.bfloat16
                          else tl.float16),
            use_fp8_w8a8=False,
            use_int8_w8a16=False)

        ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            None,  # a2_scale
            None,  # w2_scale
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=(tl.bfloat16 if hidden_states.dtype == torch.bfloat16
                          else tl.float16),
            use_fp8_w8a8=False,
            use_int8_w8a16=False)

    end_event.record()
    end_event.synchronize()
    # print(f"intermediate 0 shape = {intermediate_cache1.shape}")
    # print(f"intermediate 1 shape = {intermediate_cache2.shape}")
    # print(f"intermediate 2 shape = {intermediate_cache3.shape}")
    # print(f"config = {config}")
    # print(f"sorted token ids = {sorted_token_ids}")
    # print(f"sorted token ids shape = {sorted_token_ids.shape}")
    # print(f"expert ids = {expert_ids}")
    # print(f"num_tokens_post_padded = {num_tokens_post_padded}")

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_mixtral_moe_rocm",
        description="Tune the fused_moe kernel for mixtral.")
    parser.add_argument(
        "--TP",
        type=int,
        choices=[16, 8, 4, 2, 1],
        help="Specify the TP value that the actual model will run on",
        required=True,
    )
    parser.add_argument(
        "--GPUID",
        type=str,
        help="This script uses single GPU. Specify the GPU to use for tuning",
        default="0",
    )
    parser.add_argument('--model',
                        type=str,
                        choices=['8x7B', '8x22B'],
                        help='The Mixtral model to benchmark')

    args = parser.parse_args()

    print(f"Running tuning for {args.model} model")
    print(f"TP is set to: {args.TP}")
    print(f"GPU-ID being used for tuning: {args.GPUID}")
    sys.exit(main(args))
