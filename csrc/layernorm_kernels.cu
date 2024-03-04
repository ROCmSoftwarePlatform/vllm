#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"
#include "attention/dtype_float16.cuh"
#include "attention/dtype_bfloat16.cuh"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

template<typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  extern __shared__ __align__(sizeof(scalar_t)) char _shmem[];
  scalar_t* __restrict__ shmem = reinterpret_cast<scalar_t*>(_shmem);

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * hidden_size + idx;
    float x = (float) input[id] + (float) residual[id];
    variance += x * x;
    residual[id] = shmem[idx] = (scalar_t) x;
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) shmem[idx];
    input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

// Function specialization in the case of FP16
// Additional optimization: Use packed operations.
template<>
__global__ void fused_add_rms_norm_kernel(
  c10::Half* __restrict__ input,           // [..., hidden_size]
  c10::Half* __restrict__ residual,        // [..., hidden_size]
  const c10::Half* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  extern __shared__ __align__(sizeof(__half2)) char _shmem[];
  const int half_hidden_size = hidden_size / 2;
  __shared__ float s_variance;
  float variance = 0.0f;
  // These are declared `__restrict__` because they are not aliased in practice
  __half2* __restrict__ shmem = reinterpret_cast<__half2*>(_shmem);
  __half2* __restrict__ input2 = reinterpret_cast<__half2*>(input);
  __half2* __restrict__ residual2 = reinterpret_cast<__half2*>(residual);
  const __half2* __restrict__ weight2 = reinterpret_cast<const __half2*>(weight);

  for (int idx = threadIdx.x; idx < half_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * half_hidden_size + idx;
    __half2 x = input2[id] + residual2[id];
    residual2[id] = x;
    shmem[idx] = x * weight2[idx];
    float2 z = __half22float2(x);
    variance += z.x * z.x + z.y * z.y;
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    if (hidden_size % 2 == 1) {
      // If hidden size is odd, last element has not been processed yet
      int idx = hidden_size - 1;
      __half x = input[idx] + residual[idx];
      residual[idx] = x;
      reinterpret_cast<__half*>(_shmem)[idx] = x * reinterpret_cast<const __half*>(weight)[idx];
      variance += (float) x;
    }
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < half_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * half_hidden_size + idx;
    float2 z = __half22float2(shmem[idx]) * s_variance;
    input2[id] = __float22half2_rn(z);
  }
  if (hidden_size % 2 == 1 && threadIdx.x == 0) {
    // If hidden size is odd, last element has not been processed yet
    float x = s_variance * (float) reinterpret_cast<__half*>(_shmem)[hidden_size - 1];
    input[hidden_size - 1] = __float2half_rn(x);
  }
}

} // namespace vllm

void rms_norm(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}

void fused_add_rms_norm(
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "fused_add_rms_norm_kernel",
    [&] {
      vllm::fused_add_rms_norm_kernel<scalar_t>
      <<<grid, block, sizeof(scalar_t) * hidden_size, stream>>>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}
