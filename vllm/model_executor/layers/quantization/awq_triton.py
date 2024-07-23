import torch

import triton
import triton.language as tl


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
                          awq_order_ptr,
                          BLOCK_SIZE_X: tl.constexpr,
                          BLOCK_SIZE_Y: tl.constexpr):
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    reverse_awq_order_offsets = tl.arange(0, 8)
    reverse_awq_order_tensor =  tl.load(reverse_awq_order_ptr +
            reverse_awq_order_offsets)

    awq_order_offsets = tl.arange(0, 8)
    awq_order_tensor =  tl.load(awq_order_ptr + awq_order_offsets)

    print(f"pid_y = {pid_y}, pid_x = {pid_x}")
    # print(f"BLOCK_SIZE_Y = {BLOCK_SIZE_Y}, BLOCK_SIZE_X = {BLOCK_SIZE_X}")

    # qweight offsets for qweight_ptr
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    # offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    offsets = num_cols  * offsets_y[:, None] + offsets_x[None, :]

    print(f"offsets_x.shape = {offsets_x.shape}")

    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    # result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X) * 8
    # result_offsets_x = (offsets_x.reshape(BLOCK_SIZE_X, 8) 
        # + reverse_awq_order_tensor[None, :]).reshape(BLOCK_SIZE_X * 8)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    # result_offsets_x = (offsets_x.reshape((BLOCK_SIZE_X, 8)) * 8
            # + reverse_awq_order_tensor[None, :])
    # result_offsets = (num_cols * result_offsets_y[:, None] +
            # result_offsets_x[None, :])
    # print(f"offsets_x = {offsets_x}")
    result_offsets = 8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    # print(f"result_offsets_x = {result_offsets_x}")
    # print(f"result_offsets = {result_offsets}")

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols 

    # print(f"masks_y = {masks_y}")
    # print(f"masks_x = {masks_x}")
    masks = masks_y[:, None] & masks_x[None, :]

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    iweights = tl.load(qweight_ptr + offsets, masks)
    print(f"iweights.shape = {iweights.shape}")
    print(f"iweights = {iweights}")

    iweights = iweights.view(BLOCK_SIZE_Y * BLOCK_SIZE_X, 8)

    shifts = reverse_awq_order_tensor * 4#tl.arange(0, 8)  * 4
    shift_weights = shifts[None, :].broadcast_to(BLOCK_SIZE_Y * BLOCK_SIZE_X, 8)
    print(f"shift_weights= {shift_weights}")

    iweights = iweights >> shift_weights & 0xF
    print(f"iweights = {iweights}")

    iweights = iweights.reshape(BLOCK_SIZE_Y, BLOCK_SIZE_X * 8)

    zero_offsets_y = (pid_y * BLOCK_SIZE_Y // group_size
                      + tl.arange(0, BLOCK_SIZE_Y) // group_size)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]
    print(f"zero_offsets = {zero_offsets}")
    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
    print(f"-->zeros={zeros}")
    zeros = zeros.view(BLOCK_SIZE_Y * BLOCK_SIZE_X, 8)
    zeros = zeros >> shift_weights & 0xF
    zeros = zeros.reshape(BLOCK_SIZE_Y, BLOCK_SIZE_X * 8)
    print(f"zeros={zeros}")


    scale_offsets_y  = (pid_y * BLOCK_SIZE_Y // group_size
                       + tl.arange(0, BLOCK_SIZE_Y) // group_size)
    scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8
                        + tl.arange(0, BLOCK_SIZE_X * 8))
    scale_offsets = (num_cols * 8 * scale_offsets_y[:, None] +
                    scale_offsets_x[None, :])
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    scales = tl.load(scales_ptr + scale_offsets, scale_masks)

    print(f"scales = {scales}")
    print(f"iweights.shape = {iweights.shape}, zeros.shape = {zeros.shape}")
    iweights = (iweights - zeros) * scales
    tl.store(result_ptr + result_offsets, iweights, masks)
    return
    # print(f"offsets_y = {offsets_y}")
    # print(f"offsets_x = {offsets_x}")

    # Scale offsets for scales_ptr
    scale_offsets_y = (pid_y * BLOCK_SIZE_Y 
                      + tl.arange(0, BLOCK_SIZE_Y) // group_size)
    scale_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X) * 8
    scale_offsets = (num_cols * scale_offsets_y[:, None] +
                     scale_offsets_x[None,:])

    # Zero offsets for scales_ptr
    zero_offsets_y = (pid_y * BLOCK_SIZE_Y 
                      + tl.arange(0, BLOCK_SIZE_Y) //group_size)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    # Output offsets for result_ptr
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X) * 8
    result_offsets = (num_cols * result_offsets_y[:, None] +
            result_offsets_x[None, :])
    # print(f"result_offsets = {result_offsets}")

    # print(f"offsets = {offsets}")

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols 

    # print(f"masks_y = {masks_y}")
    # print(f"masks_x = {masks_x}")
    masks = masks_y[:, None] & masks_x[None, :]

    iweights = tl.load(qweight_ptr + offsets, masks)

    zeros = tl.load(zeros_ptr + zero_offsets, masks)

    # There are 8 values packed per int, loop over them and
    # do block-wise computations w.r.t the order and write
    # the results out to result_ptr w.r.t. the reverse order.
    for i in range(8):
        shift = i

        # Use reverse_awq_order to write result in reverse_awq_order.
        reverse_awq_order = 0
        if i == 0:
            reverse_awq_order = 0
        elif i == 1:
            reverse_awq_order = 4
        elif i == 2:
            reverse_awq_order = 1 
        elif i == 3:
            reverse_awq_order = 5
        elif i == 4:
            reverse_awq_order = 2
        elif i == 5:
            reverse_awq_order = 6
        elif i == 6:
            reverse_awq_order = 3
        elif i == 7:
            reverse_awq_order = 7

        # Use awq_order to load scales in awq_order.
        awq_order = 0
        if i == 0:
            awq_order = 0
        elif i == 1:
            awq_order = 2 
        elif i == 2:
            awq_order = 4 
        elif i == 3:
            awq_order = 6
        elif i == 4:
            awq_order = 1
        elif i == 5:
            awq_order = 3
        elif i == 6:
            awq_order = 5
        elif i == 7:
            awq_order = 7

        # Load the scales in AWQ order so that the equation:
        #  (iweights_shift - zeros_shift) * scales
        # computes the correct values.
        scales = tl.load(scales_ptr + scale_offsets + awq_order, masks)

        # Shift and extract the packed value, but its still in AWQ order.
        iweights_shifted = ((iweights >> shift) & 0xF)
        zeros_shifted = ((zeros >> shift) & 0xF)

        print(f"iweights_shifted = {iweights_shifted}")

        # Compute the dequantized results and write them in reverse
        # AWQ order.
        tl.store(result_ptr + result_offsets + reverse_awq_order,
                 # (iweights_shifted - zeros_shifted) * scales,
                 iweights_shifted,
                 masks)

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
                         split_k_iters: int,
                         thx: int,
                         thy: int) -> torch.Tensor:
    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device = qweight.device,
                         dtype = torch.float16)

    reverse_awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7],
            dtype = torch.int32, device = qweight.device)
    awq_order = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7],
            dtype = torch.int32, device = qweight.device)
    Y = qweight.shape[0] # num rows
    X = qweight.shape[1] # num cols
    group_size = 128
    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']), triton.cdiv(Y,
                    META['BLOCK_SIZE_Y']), )
    awq_dequantize_kernel[grid](qweight, scales, zeros, split_k_iters, 
            thx, thy, group_size, result, X, Y, reverse_awq_order, awq_order,
            BLOCK_SIZE_X = 32, BLOCK_SIZE_Y = 64)

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


# qweightss [R     , C // 8], int32
# scales  - [R // G, C     ], float16
# zeros   - [R // G, C // 8], int32
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

    return (iweights - zeros) * scales

def main():
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
    # AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    qweight = torch.randint(-1412623820,-1412623820 + 1, (qweight_rows,
                         qweight_cols),
                         dtype=qweight_dtype,
                         device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(305441741, 305441741 + 1, (zeros_rows,
                          zeros_cols),
                          dtype=zeros_dtype,
                          device=device)
    print(f"zeros.shape = {zeros.shape}")
    # zeros = torch.zeros(zeros_rows,
                       # zeros_cols,
                       # dtype=zeros_dtype,
                       # device=device)
    print(f"qweight = {qweight}")
    if use_triton:
      iweights_triton = awq_dequantize_triton(
        qweight, scales, zeros, split_k_iters, thx, thy)
      print(f"Triton result:iweights_triton = {iweights_triton}")

    if use_torch:
      iweights_torch = awq_dequantize_torch(
        qweight, scales, zeros, split_k_iters, thx, thy)
      print(f"Torch result:iweights_torch = {iweights_torch}")

if __name__ == '__main__':
    main()
