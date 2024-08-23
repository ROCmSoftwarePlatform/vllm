import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
#from vllm.model_executor.models.mixtral import MixtralMoE

from vllm import envs

padding_size = 128 if envs.VLLM_MOE_PADDING else 0


def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)

def permute_weight(x: torch.Tensor) -> torch.Tensor:
    ## Hardcode BLOCK_K and BLOCK_N

    BK = 128
    BN = 128
    x_ = x.clone()
    x_ = x_.view(x.shape[0],
                 x.shape[1]//BN, BN//16, 16,
                     x.shape[2]//BK, BK//32, 4, 8)
    x_ = x_.permute(0,1,5,2,6,4,3,7)
    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2]);
    return x_


#@pytest.mark.parametrize("m", [512, 222, 33, 1])
#@pytest.mark.parametrize("n", [2048, 256, 1024])
#@pytest.mark.parametrize("k", [128, 511, 1024])
#@pytest.mark.parametrize("e", [8, 64])
#@pytest.mark.parametrize("topk", [2, 6])
#@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k+padding_size), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n+padding_size), device='cuda', dtype=dtype) / 10

    

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    torch_output = torch_moe(a, w1, w2, score, topk)
    print(f"torch_output: {torch_output}")

    if envs.VLLM_MOE_SHUFFLE:
        w1_shuffled = permute_weight(w1.data)
        w2_shuffled = permute_weight(w2.data)
    else:
        w1_shuffled = w1
        w2_shuffled = w2
    
    triton_output = fused_moe(a, w1_shuffled, w2_shuffled, score, topk, renormalize=False)
    print(f"triton_output: {triton_output}")
    #triton_col_output = fused_moe_col_major(a, w1, w2, score, topk, renormalize=False)
    #print(f"triton_col_output: {triton_col_output}")

    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    #assert torch.allclose(triton_col_output, torch_output, atol=1e-2, rtol=0)

if __name__ == '__main__':

    print( "test" )

    test_fused_moe(8, 4096, 8192, 8, 2, torch.float16)
