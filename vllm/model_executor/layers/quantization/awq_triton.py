import torch

import triton
import triton.language as tl

import argparse

@triton.jit
def awq_dequantize_kernel(qweight_ptr,   # quantized matrix
                          scales_ptr,    # scales, per group
                          zeros_ptr,     # zeros, per group
                          split_k_iters, # Not used
                          thx,           # Not used
                          thy,           # Not used
                          group_size,    # Should always be 128
                          result_ptr,    # Output matrix
                          num_cols,      # input num cols in qweight
                          num_rows,      # input num rows in qweight
                          reverse_awq_order_ptr,
                          BLOCK_SIZE_X: tl.constexpr,
                          BLOCK_SIZE_Y: tl.constexpr):
    # Setup the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    offsets = num_cols  * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols 

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (8 * num_cols * result_offsets_y[:, None]
                      + result_offsets_x[None, :])

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks)

    # Load the AWQ reverse order offsets.
    reverse_awq_order_offsets = tl.arange(0, 8)
    reverse_awq_order_tensor =  tl.load(reverse_awq_order_ptr +
            reverse_awq_order_offsets)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = (pid_y * BLOCK_SIZE_Y // group_size
                      + tl.arange(0, BLOCK_SIZE_Y) // group_size)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y  = (pid_y * BLOCK_SIZE_Y // group_size
                       + tl.arange(0, BLOCK_SIZE_Y) // group_size)
    scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8
                        + tl.arange(0, BLOCK_SIZE_X * 8))
    scale_offsets = (num_cols * 8 * scale_offsets_y[:, None] +
                    scale_offsets_x[None, :])
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks)

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(tl.float16)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_SIZE_K'] * args['SPLIT_K']) == 0,
    })
@triton.jit
def awq_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    zeros_ptr, scales_ptr,
    M, N, K,
    awq_group_size, reverse_awq_order_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_zn, stride_zk,
    stride_sn, stride_sk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,  EVEN_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_zn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_sn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    zeros_ptrs = (zeros_ptr + (offs_k[:, None] // awq_group_size) * stride_zk 
                 + offs_zn[None, :] * stride_zn)
    scales_ptrs = (scales_ptr + (offs_k[:, None] // awq_group_size) * stride_sk 
                  + offs_sn[None, :] * stride_sn)
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    reverse_awq_order_offsets = tl.arange(0, 8)
    reverse_awq_order_tensor =  tl.load(reverse_awq_order_ptr +
            reverse_awq_order_offsets)
    shifts = reverse_awq_order_tensor * 4

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            zeros = tl.load(zeros_ptrs)
            scales = tl.load(scales_ptrs)
        else:
            a = tl.load(a_ptrs,
                        mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            #TODO: Make zeros and scales loading work for the 
            # AWQ group size
            zeros = tl.load(zeros_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            scales = tl.load(scales_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)

        # Dequantize the B matrix.
        shifts = shifts[None, :].broadcast_to(b.shape[0] * b.shape[1], 8)
        shifts = shifts_b.reshape(b.shape[0], b.shape[1] * 8)
        b = b.reshape(b.shape[0] * b.shape[1], 1
              ).broadcast_to(b.shape[0] * b.shape[1], 8
              ).reshape(b.shape[0], b.shape[1] * 8)
        zeros = zeros.reshape(zeros.shape[0] * zeros.shape[1], 1
              ).zerosroadcast_to(zeros.shape[0] * zeros.shape[1], 8
              ).reshape(zeros.shape[0], zeros.shape[1] * 8)
        b = b >> shifts & 0xF
        zeros = zeros >> shifts & 0xF
        b = (b - zeros) * scales
        b = b.to(tl.float16)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk



    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)

# Example input: 
#   qweight.size=torch.Size([3584, 576]),
#   qweight.dtype = torch.int32,
#   scales.size=torch.Size([28, 4608]),
#   scales.dtype=torch.float16,
#   zeros.size=torch.Size([28, 576]),
#   zeros.dtype=torch.int32
#   split_k_iters=0
#   thx=0
#   thy=0
def awq_dequantize_triton(qweight: torch.Tensor,
                         scales: torch.Tensor,
                         zeros: torch.Tensor,
                         split_k_iters: int, # Not used
                         thx: int, # Not used
                         thy: int # Not used
                         ) -> torch.Tensor:
    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device = qweight.device,
                         dtype = torch.float16)

    block_size_x = 32
    block_size_y = 32

    reverse_awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7],
            dtype = torch.uint8, device = qweight.device)

    Y = qweight.shape[0] # num rows
    X = qweight.shape[1] # num cols
    group_size = 128
    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']), triton.cdiv(Y,
                    META['BLOCK_SIZE_Y']), )
    awq_dequantize_kernel[grid](qweight, scales, zeros, split_k_iters, 
            thx, thy, group_size, result, X, Y, reverse_awq_order,
            BLOCK_SIZE_X = block_size_x, BLOCK_SIZE_Y = block_size_y)

    return result


def reverse_awq_order(t: torch.Tensor):
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    t = t[:, reverse_order_tensor] & 0xF
    return t 


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
def awq_dequantize_torch(qweight: torch.Tensor,
                         scales: torch.Tensor,
                         qzeros: torch.Tensor,
                         split_k_iters: int,
                         thx: int,
                         thy: int) -> torch.Tensor:
    print(f"awq_dequantize_torch:qweight.shape = {qweight.shape}"
          f", qzeros.shape={qzeros.shape}")
    bits = 4
    group_size = 128
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    
    iweights = torch.bitwise_right_shift(
        qweight[:, :, None],
        shifts[None, None, :]).to(torch.int8)

    iweights = iweights.view(iweights.shape[0], -1)

    # iweights = reverse_awq_order(iweights)
    # return (iweights & 0xF).to(torch.float16)

    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)

    zeros = zeros.view(qzeros.shape[0], -1)

    zeros = reverse_awq_order(zeros)
    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)


    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    print(f"awq_dequantize_torch:iweights.shape = {iweights.shape},"
          f"zeros.shape={zeros.shape}, "
          f"scales.shape={scales.shape}")

    # return iweights.to(torch.float16)
    return (iweights - zeros) * scales

def test_dequantize():
    print("=" * 10 + " TESTING DEQUANTIZE" + "=" * 10)
    use_triton = True 
    use_torch = True

    qweight_rows = 3584
    qweight_cols = 576
    group_size = 128
    small_test_size = True
    if small_test_size:
        qweight_rows = 256
        qweight_cols = 128
    print(f"qweight_rows = {qweight_rows}, qweight_cols = {qweight_cols}")
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32
    split_k_iters=0
    thx=0
    thy=0
    device='cuda'
    torch.manual_seed(0)

    qweight = torch.randint(0,10000000, (qweight_rows,
                         qweight_cols),
                         dtype=qweight_dtype,
                         device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0, 10000000, (zeros_rows,
                          zeros_cols),
                          dtype=zeros_dtype,
                          device=device)
    print(f"qweight = {qweight}")
    if use_triton:
      iweights_triton = awq_dequantize_triton(
        qweight, scales, zeros, split_k_iters, thx, thy)
      print(f"Triton result:iweights_triton = {iweights_triton}")
      print(f"Any infs in triton result? --> {not torch.all(False == torch.isinf(iweights_triton))}")

    if use_torch:
      iweights_torch = awq_dequantize_torch(
        qweight, scales, zeros, split_k_iters, thx, thy)
      print(f"Torch result:iweights_torch = {iweights_torch}")

    if use_torch and use_triton:
        diff = iweights_torch - iweights_triton
        error = torch.sum(torch.sqrt(diff * diff))
        print(f"error = {error}")

def awq_gemm_triton(input: torch.Tensor, qweight: torch.Tensor,
                   qzeros: torch.Tensor, scales: torch.Tensor,
                   split_k_iters: int) -> torch.Tensor:
  weights = awq_dequantize_torch(qweight, scales, qzeros,
                                       split_k_iters, 0, 0)

  grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) 
                       * triton.cdiv(N, META['BLOCK_SIZE_N']),
                       split_k_iters,)
  M, K = input.shape
  N = qweight.shape[1] * 8
  awq_group_size = 128
  block_size_m = 32
  block_size_n = 32
  block_size_k = 32
  reverse_awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7],
          dtype = torch.uint8, device = qweight.device)

  result = torch.empty((M, N), dtype = torch.float16, device = input.device)

  # A = input, B = qweight, C = result
  # A = M x K, B = K x N, C = M x N
  awq_gemm_kernel[grid](
    input, qweight, result,
    qzeros, scales,
    M, N, K,
    awq_group_size, reverse_awq_order,
    input.stride(0), input.stride(1),
    qweight.stride(0), qweight.stride(1),
    result.stride(0), result.stride(1), 
    qzeros.stride(0), qzeros.stride(1),
    scales.stride(0), scales.stride(1),
    BLOCK_SIZE_M = block_size_m, BLOCK_SIZE_N = block_size_n,
    BLOCK_SIZE_K = block_size_k,
    SPLIT_K = split_k_iters)

  return result

# input   - [N, K]
# qweight - [K, M // 8]
# qzeros  - [K // G, M // 8]
# scales  - [K // G, M]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_torch(input: torch.Tensor, qweight: torch.Tensor,
                   qzeros: torch.Tensor, scales: torch.Tensor,
                   split_k_iters: int) -> torch.Tensor:
    weights = awq_dequantize_torch(qweight, scales, qzeros, split_k_iters, 0, 0)
    return torch.matmul(input, weights)

def test_gemm():
    print("=" * 10 + " TESTING GEMM " + "=" * 10)

    split_k_iters = 1
    group_size = 128
    device = "cuda"

    # input.size = torch.Size([1, 3584]),
    # input.dtype = torch.float16
    # qweight.size = torch.Size([3584, 448]),
    # qweight.dtype = torch.int32
    # qzeros.size = torch.Size([28, 3584]),
    # qzeros.dtype = torch.float16
    # scales.size = torch.Size([28, 448]),
    # scales.dtype = torch.int32
    # split_k_iters = 8
    input_rows = 1
    input_cols = 3584
    input_dtype = torch.float16
    qweight_rows = input_cols
    qweight_cols = 448
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols
    qzeros_dtype = torch.int32

    input = torch.rand((input_rows, input_cols),
                       dtype = input_dtype)
    qweight = torch.randint(0, torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols))
    qzeros = torch.randint(0, torch.iinfo(torch.int32).max,
                            (qzeros_rows, qzeros_cols))
    scales = torch.rand((scales_rows, scales_cols),
                        dtype = scales_dtype)

    use_triton = True
    use_torch = True

    torch.manual_seed(0)

    if use_torch:
        output_torch = awq_gemm_torch(input, qweight, qzeros, scales,
                                      split_k_iters)
        print(f"output_torch = {output_torch}")

    if use_triton:
        output_triton = awq_gemm_triton(input, qweight, qzeros, scales,
                                        split_k_iters)
        print(f"output_triton = {output_triton}")

def main():
    parser = argparse.ArgumentParser(description="awq_triton test driver",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test")
    known_args, unknown_args = parser.parse_known_args()
    if known_args.test is not None:
        if known_args.test == "dequantize":
            test_dequantize()
        elif known_args.test == "gemm":
            test_gemm()
        else:
            print(f"Unknown test {known_args.test}")
    else:
        print("No test provided.")
        parser.print_help()


if __name__ == '__main__':
    main()
