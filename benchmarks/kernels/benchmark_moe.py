import argparse
import time
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple, TypedDict

import ray
import torch
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser, is_navi

FP8_DTYPE = torch.float8_e4m3fnuz if current_platform.is_rocm(
) and not is_navi() else torch.float8_e4m3fn


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    padding_size = 0
    if envs.VLLM_MOE_PADDING and not (use_fp8_w8a8 or use_int8_w8a16):
        padding_size = 128  # fp16 padding size
    if envs.VLLM_FP8_PADDING and use_fp8_w8a8:
        padding_size = 256  # fp8 padding size. Ignoring int8 for now

    if use_int8_w8a16:
        w1 = torch.randint(-127,
                           127, (
                               num_experts,
                               shard_intermediate_size,
                               hidden_size,
                           ),
                           dtype=torch.int8)
        w2 = torch.randint(-127,
                           127, (
                               num_experts,
                               hidden_size,
                               shard_intermediate_size // 2,
                           ),
                           dtype=torch.int8)
    else:
        w1 = torch.randn(num_experts,
                         shard_intermediate_size,
                         hidden_size + padding_size,
                         dtype=init_dtype)[..., :-padding_size]
        w2 = torch.randn(num_experts,
                         hidden_size,
                         shard_intermediate_size // 2,
                         dtype=init_dtype)
    gating_output = torch.randn(num_iters,
                                num_tokens,
                                num_experts,
                                dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size),
                               dtype=torch.float32)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8:
        w1_scale = torch.randn(num_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)

        w1 = w1.to(FP8_DTYPE)
        w2 = w2.to(FP8_DTYPE)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        from vllm.model_executor.layers.fused_moe import override_config
        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                input_gating,
                topk,
                renormalize=True,
                inplace=True,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def get_rocm_tuning_space(use_fp16):
    # small search space, no pruning required
    # bypassLDS: block_n/num_warps=16 for perf
    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [128] if use_fp16 else [64]
    block_k_range = [128] if use_fp16 else [256]

    num_warps_range = [8] if use_fp16 else [4]
    group_m_range = [1]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [2]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16]
    kpack_range = [2]

    param_ranges = {
        "BLOCK_SIZE_M": block_m_range,
        "BLOCK_SIZE_N": block_n_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
        "waves_per_eu": waves_per_eu_range,
        "matrix_instr_nonkdim": matrix_instr_nonkdim_range,
        "kpack": kpack_range,
    }

    return param_ranges


def get_configs_compute_bound(use_fp16) -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []

    if current_platform.is_rocm():
        param_ranges = get_rocm_tuning_space(use_fp16)
    else:
        # Reduced search space for faster tuning.
        # TODO(woosuk): Increase the search space and use a performance model to
        # prune the search space.
        block_m_range = [16, 32, 64, 128, 256]
        block_n_range = [32, 64, 128, 256]
        block_k_range = [64, 128, 256]
        num_warps_range = [4, 8]
        group_m_range = [1, 16, 32, 64]
        num_stage_range = [2, 3, 4, 5]

        param_ranges = {
            "BLOCK_SIZE_M": block_m_range,
            "BLOCK_SIZE_N": block_n_range,
            "BLOCK_SIZE_K": block_k_range,
            "GROUP_SIZE_M": group_m_range,
            "num_warps": num_warps_range,
            "num_stages": num_stage_range,
        }

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)
    return configs


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        current_platform.seed_everything(seed)
        self.seed = seed
        # get the device id to allocate the tensors and kernels explicitly
        # on the respective GPU ID
        self.device_id = int(ray.get_gpu_ids()[0])

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
    ) -> Tuple[Dict[str, int], float]:
        current_platform.seed_everything(self.seed)
        dtype_str = get_config_dtype_str(dtype,
                                         use_int8_w8a16=use_int8_w8a16,
                                         use_fp8_w8a8=use_fp8_w8a8)
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        op_config = get_moe_configs(num_experts, shard_intermediate_size // 2,
                                    dtype_str)
        if op_config is None:
            config = get_default_config(num_tokens, num_experts,
                                        shard_intermediate_size, hidden_size,
                                        topk, dtype_str)
        else:
            config = op_config[min(op_config.keys(),
                                   key=lambda x: abs(x - num_tokens))]
        kernel_time = benchmark_config(config, num_tokens, num_experts,
                                       shard_intermediate_size, hidden_size,
                                       topk, dtype, use_fp8_w8a8,
                                       use_int8_w8a16)
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        search_space: List[Dict[str, int]],
    ) -> Dict[str, int]:
        best_config = None
        best_time = float("inf")

        with torch.cuda.device(self.device_id):
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_config(config,
                                                   num_tokens,
                                                   num_experts,
                                                   shard_intermediate_size,
                                                   hidden_size,
                                                   topk,
                                                   dtype,
                                                   use_fp8_w8a8,
                                                   use_int8_w8a16,
                                                   num_iters=20)
                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M":
        config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N":
        config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K":
        config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M":
        config["GROUP_SIZE_M"],
        "num_warps":
        config["num_warps"],
        "num_stages":
        config["num_stages"],
        **({
            "waves_per_eu": config["waves_per_eu"]
        } if "waves_per_eu" in config else {}),
        **({
            "matrix_instr_nonkdim": config["matrix_instr_nonkdim"]
        } if "matrix_instr_nonkdim" in config else {}),
        **({
            "kpack": config["kpack"]
        } if "kpack" in config else {}),
    }


def save_configs(configs: Dict[int, BenchmarkConfig], num_experts: int,
                 shard_intermediate_size: int, hidden_size: int, topk: int,
                 dtype: torch.dtype, use_fp8_w8a8: bool,
                 use_int8_w8a16: bool) -> None:
    dtype_str = get_config_dtype_str(dtype,
                                     use_int8_w8a16=use_int8_w8a16,
                                     use_fp8_w8a8=use_fp8_w8a8)

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(num_experts, shard_intermediate_size // 2,
                                    dtype_str)

    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model)
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size

    hidden_size = config.hidden_size
    dtype = torch.float16 if current_platform.is_rocm() else config.torch_dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"

    if args.batch_size is None:
        batch_sizes = [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
            2048, 3072, 4096, 8192, 16384, 18432, 20480
        ]
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        is_fp16 = not (use_fp8_w8a8 or use_int8_w8a16)
        search_space = get_configs_compute_bound(is_fp16)
        print(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        configs = _distribute(
            "tune", [(batch_size, E, shard_intermediate_size, hidden_size,
                      topk, dtype, use_fp8_w8a8, use_int8_w8a16, search_space)
                     for batch_size in batch_sizes])
        best_configs = {
            M: sort_config(config)
            for M, config in zip(batch_sizes, configs)
        }
        save_configs(best_configs, E, shard_intermediate_size, hidden_size,
                     topk, dtype, use_fp8_w8a8, use_int8_w8a16)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark", [(batch_size, E, shard_intermediate_size, hidden_size,
                           topk, dtype, use_fp8_w8a8, use_int8_w8a16)
                          for batch_size in batch_sizes])

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--tp-size", "-tp", type=int, default=2)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["auto", "fp8_w8a8", "int8_w8a16"],
                        default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    main(args)
