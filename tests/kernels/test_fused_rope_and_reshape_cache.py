import random
from itertools import accumulate, product
from time import perf_counter
from typing import List, Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import CacheConfig
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope

from .allclose_default import get_default_atol, get_default_rtol

NUM_TOKENS = [42]  # Arbitrary values for testing

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024]
# We assume fp8 is always enabled for testing.
KV_CACHE_DTYPE = ["auto", "fp8"]

BLOCK_SIZES = [16]

DTYPES = [torch.bfloat16, torch.float16] # FIXME torch.half
HEAD_SIZES = [128]
NUM_HEADS = [32] # Arbitrary values for testing
NUM_KV_HEADS = [8] # Arbitrary values for testing
ROTARY_DIMS = [None]  # None means rotary dim == head size FIXME: 32
BATCH_SIZES = [8]  # Arbitrary values for testing
SEQ_LENS = [1024]  # Arbitrary values for testing
IS_NEOX_STYLE = [True, False]
SEEDS = [0]
CUDA_DEVICES = [0] # FIXME


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fused_rotary_embedding_with_no_cache(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    is_neox_style: bool,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:

    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    if rotary_dim is None:
        rotary_dim = head_size

    _ROPE_DICT.clear()
    cache_config = CacheConfig(16, 1.0, 1, "auto")
    rope = get_rope(head_size, rotary_dim, max_position, base, 
                    is_neox_style, rope_scaling={"type": "llama3", 
                                                "low_freq_factor": 1.0, 
                                                "high_freq_factor": 2.0,
                                                "original_max_position_embeddings": 1024},
                    fused_with_kv_cache_op=True,
                    cache_config = cache_config)
    rope = rope.to(dtype=dtype)

    #------------------Simulate------------------------------

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn(batch_size,
                        seq_len,
                        num_kv_heads * head_size,
                        dtype=dtype)
    value = torch.randn_like(key)

    ref_query, ref_key = rope.forward_native(positions, query, key)

    #----------------------Actual-Run------------------------

    kv_scale = 1.0
    key_cache, value_cache = torch.empty(0, 0, 0, 0, 0), torch.empty(0, 0, 0, 0)
    slot_mapping = torch.empty(0, dtype=torch.long)

    rope.forward(
        positions, query, key, value, 
        key_cache, value_cache,
        slot_mapping, kv_scale, kv_scale)

    #----------------------Assert----------------------------
    assert torch.allclose(query, ref_query, atol=0.001, rtol=0.1)
    assert torch.allclose(key, ref_key, atol=0.001, rtol=0.1)




@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fused_rotary_embedding_with_reshape_cache(
    kv_cache_factory,
    block_size: int,
    num_blocks: int,
    kv_cache_dtype: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    is_neox_style: bool,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:

    torch.set_printoptions(precision=4)
    torch.set_printoptions(threshold=5000)

    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    if rotary_dim is None:
        rotary_dim = head_size

    _ROPE_DICT.clear()
    cache_config = CacheConfig(block_size, 1.0, 1, kv_cache_dtype)
    rope = get_rope(head_size, rotary_dim, max_position, base, 
                    is_neox_style, 
                    rope_scaling={"rope_type": "llama3", 
                                "factor": 1.0,
                                "low_freq_factor": 1.0, 
                                "high_freq_factor": 2.0,
                                "original_max_position_embeddings": 1024},
                    fused_with_kv_cache_op=True,
                    cache_config = cache_config)
    rope = rope.to(dtype=dtype)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), batch_size * seq_len)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # print(f"{key_cache.shape=}")
    # print(f"{value_cache.shape=}")

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Using default kv_scale
    kv_scale = 1.0

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    # positions = positions.fill_(1)
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn(batch_size,
                        seq_len,
                        num_kv_heads * head_size,
                        dtype=dtype)
    value = torch.randn_like(key)

    #------------------Simulate------------------------------
    time_start = perf_counter()

    ref_query, ref_key = rope.forward_cuda(positions, query.clone(), key.clone())
    ops.reshape_and_cache(ref_key.view(-1, num_kv_heads, head_size),
                          value.view(-1, num_kv_heads, head_size),
                          cloned_key_cache, 
                          cloned_value_cache, slot_mapping,
                          kv_cache_dtype, kv_scale, kv_scale)

    time_end = perf_counter()
    # report the duration
    print(f'Non fused call {(time_end - time_start) * 1000} ms')

    #----------------------Actual-Run------------------------
    time_start = perf_counter()

    rope.forward(
        positions, query, key, value, 
        key_cache, value_cache,
        slot_mapping, kv_scale, kv_scale)

    time_end = perf_counter()
    # report the duration
    print(f'Fused run {(time_end - time_start) * 1000} ms')

    #----------------------Assert----------------------------
    atol, rtol = 1e-05, 1e-05
    if kv_cache_dtype == "fp8":
        atol, rtol = 0.001, 0.001


    # print(f"{query[0, 0, 0:128].view(-1, 8)}")
    # print(f"{query[0, 0, 128:256].view(-1, 8)}")
    # print(f"{query[0, 0, 256:384].view(-1, 8)}")
    # print(f"{query[0, 0, 384:512].view(-1, 8)}")
    # print(f"{query[0, 0, 512:640].view(-1, 8)}")
    # print(f"{ref_query[0, 0, 0:128].view(-1, 8)}")

    assert torch.allclose(query, ref_query, atol=atol, rtol=rtol)
    assert torch.allclose(key, ref_key, atol=atol, rtol=rtol)
    assert torch.allclose(key_cache, cloned_key_cache, atol=atol, rtol=rtol)
    assert torch.allclose(value_cache, cloned_value_cache, atol=atol, rtol=rtol)
