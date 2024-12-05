from itertools import product


## Utilize method from rocm/Triton tuning script
def get_full_tuning_space(use_fp8):
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    if use_fp8:
        block_k_range.remove(16)  # BLOCK_K=16 not supported for fp8
    # split_k_range = [1] #, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [0]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [] if use_fp8 else [16, 32]
    kpack_range = [] if use_fp8 else [1, 2]

    param_ranges = {
        "BLOCK_SIZE_M": block_mn_range,
        "BLOCK_SIZE_N": block_mn_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
        "waves_per_eu": waves_per_eu_range,
    }

    if not use_fp8:
        param_ranges["matrix_instr_nonkdim"] = matrix_instr_nonkdim_range
        param_ranges["kpack"] = kpack_range

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)

    return configs


## Utilize method from rocm/Triton tuning script
def prune_configs(M, N, K, configs, is_fp8=False):
    pruned_configs = []
    elemBytes_a = 1 if is_fp8 else 2  # Assuming fp16 or fp8 cases only
    elemBytes_b = 1 if is_fp8 else 2  # Assuming fp16 or fp8 cases only

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")

        if not is_fp8:
            matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
            if matrix_instr_nonkdim > mfma:
                continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elements per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = 1  # config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if not is_fp8:
            if (matrix_instr_nonkdim > BLOCK_SIZE_M
                    or matrix_instr_nonkdim > BLOCK_SIZE_N):
                continue
            if (matrix_instr_nonkdim >= M
                    and matrix_instr_nonkdim != BLOCK_SIZE_M):
                continue
            if (matrix_instr_nonkdim >= N
                    and matrix_instr_nonkdim != BLOCK_SIZE_N):
                continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a +
               BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b)
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


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
