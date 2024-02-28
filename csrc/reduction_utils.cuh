/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cuda_compat.h"

#ifndef _warpSize
#define _log2warpSize 6
#define _warpSize     (1 << _log2warpSize)
#else
#error("This is an oopsie")
#endif
namespace vllm {
template<typename T, int startMask = (_warpSize >> 1)>
__inline__ __device__ T warpReduceSum(T val) {
  static_assert((startMask & (startMask - 1)) == 0,
    "startMask is not a positive power of 2!");
  #pragma unroll
  for (int mask = startMask; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[_warpSize];
  int lane = threadIdx.x & (_warpSize - 1);
  int wid = threadIdx.x >> _log2warpSize;
  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / (float) _warpSize)) ? shared[lane] : (T)(0.0f);
  constexpr int maxActiveLanes = 1024 >> _log2warpSize;
  val = warpReduceSum<T, maxActiveLanes>(val);
  return val;
}

} // namespace vllm
#undef _log2warpSize
#undef _warpSize