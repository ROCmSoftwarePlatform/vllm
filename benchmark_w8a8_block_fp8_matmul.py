import argparse

import itertools

import json

import multiprocessing as mp

import os

import triton.language as tl

default_config = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "kpack": 1,
    "matrix_instr_nonkdim": 16,
}
default_values = tuple(default_config.values())


def run_benchmark(update_callback, a, b, a_per_token, b_per_block, group_n,
                  group_k, out_dtype, config_choices_list):
    import torch
    import triton
    import triton.language as tl
    from vllm.model_executor.layers.quantization.utils.fp8_utils \
         import  w8a8_block_fp8_matmul
    M, K = a.shape
    N = b.shape[0]
    quantiles = [0.5, 0.2, 0.8]
    warmup = 20
    rep = 100
    num_choices = len(config_choices_list)

    tune_benchmark_obj = triton.testing.Benchmark(
        x_names=[
            "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
            "num_warps", "kpack", "matrix_instr_nonkdim"
        ],
        x_vals=config_choices_list,
        x_log=True,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=None,
        args={},
    )

    @triton.testing.perf_report(tune_benchmark_obj)
    def bench_config(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
                     num_warps, kpack, matrix_instr_nonkdim, provider):
        config = {
            "BLOCK_SIZE_M": BLOCK_SIZE_M,
            "BLOCK_SIZE_N": BLOCK_SIZE_N,
            "BLOCK_SIZE_K": BLOCK_SIZE_K,
            "GROUP_SIZE_M": GROUP_SIZE_M,
            "num_warps": num_warps,
            "kpack": kpack,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
        }
        bench_function = lambda: w8a8_block_fp8_matmul(a,
                                                       b,
                                                       a_per_token,
                                                       b_per_block,
                                                       [group_n, group_k],
                                                       out_dtype,
                                                       tune_config=config)
        ms, min_ms, max_ms = triton.testing.do_bench(bench_function,
                                                     quantiles=quantiles,
                                                     warmup=warmup,
                                                     rep=rep)

        golden_function = lambda: w8a8_block_fp8_matmul(a,
                                                       b,
                                                       a_per_token,
                                                       b_per_block,
                                                       [group_n, group_k],
                                                       out_dtype,
                                                       use_default_config=True)
        result = bench_function()
        golden = golden_function()

        if not torch.allclose(golden, result, rtol=1e-5, atol=1e-5):
            ms = max_ms = min_ms = 1000000.0

        update_callback()
        return ms, max_ms, min_ms

    result_data_frames = bench_config.run(return_df=True)
    return result_data_frames


def tune(update_callback, start_callback, partition_func, event_queue,
         output_file):
    import torch
    import triton
    import triton.language as tl
    choices_NK = [
        #   N K
        (1536, 1536),
        (1536, 7168),
        (1536, 7168),
        (2048, 512),
        (2304, 7168),
        (24576, 7168),
        (256, 7168),
        (3072, 1536),
        (3072, 7168),
        (32768, 512),
        (36864, 7168),
        (4096, 512),
        (4608, 7168),
        (512, 7168),
        (576, 7168),
        (7168, 1024),
        (7168, 1152),
        (7168, 128),
        (7168, 16384),
        (7168, 18432),
        (7168, 2048),
        (7168, 2304),
        (7168, 256),
        (7168, 8192),
        (7168, 8192),
        (8192, 1536),
    ]

    choices_M = [
        1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048,
        3072, 4096
    ]

    # [(M, K, N), ...]
    shapes = [(t[0], t[1][1], t[1][0])
              for t in list(itertools.product(choices_M, choices_NK))]

    block_m_choices = [32, 64, 128, 256]
    block_n_choices = [32, 64, 128, 256]
    block_k_choices = [32, 64, 128, 256]
    group_m_choices = [1, 8, 16, 32]
    num_warps_choices = [4]
    kpack_choices = [1]#[1, 2]
    matrix_instr_nonkdim_choices = [16]#[16, 32]

    config_choices = lambda: itertools.product(
        block_m_choices, block_n_choices, block_k_choices, group_m_choices,
        num_warps_choices, kpack_choices, matrix_instr_nonkdim_choices)
    out_dtype = torch.float16

    torch.manual_seed(0)

    a_element_size = 1
    b_element_size = 1

    group_n = 128
    group_k = 128

    results = {}
    for shape in shapes:
        M, K, N = shape
        a = torch.randn((M, K), dtype=torch.float16,
                        device='cuda').to(torch.float8_e4m3fnuz)
        b = torch.randn((N, K), dtype=torch.float16,
                        device='cuda').to(torch.float8_e4m3fnuz)
        a_per_token = torch.randn((M, triton.cdiv(K, group_k)),
                                  dtype=torch.float16,
                                  device='cuda')
        b_per_block = torch.randn(
            (triton.cdiv(N, group_n), triton.cdiv(K, group_k)),
            dtype=torch.float16,
            device='cuda')
        config_choices_list = list(
            filter(get_pruner(M, N, K, a_element_size, b_element_size),
                   config_choices()))
        work_list = partition_func(config_choices_list)
        result_configs = run_benchmark(update_callback, a, b, a_per_token,
                                       b_per_block, group_n, group_k,
                                       out_dtype, work_list)

        collect_results(result_configs, M, N, K, results)

    json_output = json.dumps(results, sort_keys=True, indent=4)
    if (output_file):
        with open(output_file, 'w') as f:
            f.write(json_output)
    else:
        print(json_output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int)
    parser.add_argument('-f', '--first_gpu', type=int)
    args = parser.parse_args()
    print(f"#jobs = {args.jobs}")

    first_gpu = 0 if args.first_gpu is None else args.first_gpu

    num_jobs = min(args.jobs, 8 - first_gpu)

    mp.set_start_method('spawn')

    event_queue = mp.Queue()

    listener = mp.Process(target=listener_function,
                          args=(
                              event_queue,
                              num_jobs,
                          ))
    listener.start()

    worker_infos = []
    for pid in range(num_jobs):
        os.environ["HIP_VISIBLE_DEVICES"] = f"{pid+first_gpu}"
        parent_conn, child_conn = mp.Pipe()
        output_file = f"results_{pid}.txt"
        worker = mp.Process(target=worker_function,
                            args=(pid, num_jobs, child_conn, event_queue,
                                  output_file))
        worker.start()
        worker_infos.append({'worker': worker, 'connection': parent_conn})

    for info in worker_infos:
        info['worker'].join()

    event_queue.put(None)
    listener.join()


if __name__ == '__main__':
    main()
