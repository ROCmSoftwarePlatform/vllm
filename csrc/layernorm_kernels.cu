#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"
#include "attention/dtype_float16.cuh"

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

// Consider creating a similar overload/"partial specialization" for scalar_t == bf16
template<typename scalar_t, int hidden_size>
__global__ typename std::enable_if<std::is_same<scalar_t, c10::Half>::value, void>::type 
  fused_add_rms_norm_kernel(
  c10::Half* __restrict__ input,           // [..., hidden_size]
  c10::Half* __restrict__ residual,        // [..., hidden_size]
  const c10::Half* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens) {
  static_assert(hidden_size % 2 == 0);
  constexpr int half_hidden_size = hidden_size / 2;
  __shared__ float s_variance;
  float variance = 0.0f;
  __shared__ __half2 shmem[half_hidden_size];
  __half2* __restrict__ input2 = reinterpret_cast<__half2*>((void*)input);
  __half2* __restrict__ residual2 = reinterpret_cast<__half2*>((void*)residual);
  const __half2* __restrict__ weight2 = reinterpret_cast<const __half2*>((void*)weight);
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < half_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * half_hidden_size + idx;
    __half2 x = input2[id] + residual2[id];
    residual2[id] = x;
    shmem[idx] = x;
    float2 z = __half22float2(x);
    variance += z.x * z.x + z.y * z.y;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < half_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * half_hidden_size + idx;
    float2 z = __half22float2(shmem[idx]) * s_variance;
    input2[id] = __float22half2_rn(z) * weight2[idx];
  }
}


template<typename scalar_t, int hidden_size>
__global__ void fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens) {
  __shared__ float s_variance;
  float variance = 0.0f;
  __shared__ scalar_t shmem[hidden_size];
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx] + (float) residual[blockIdx.x * hidden_size + idx];
    residual[blockIdx.x * hidden_size + idx] = (scalar_t) x;
    variance += x * x;
    shmem[idx] = (scalar_t) x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) shmem[idx];
    input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
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

#define LAUNCH_FUSED_ADD_RMS_NORM(HIDDEN_SIZE)  \
  VLLM_DISPATCH_FLOATING_TYPES(                 \
    input.scalar_type(),                        \
    "fused_add_rms_norm_kernel",                \
    [&] {                                       \
      vllm::fused_add_rms_norm_kernel<scalar_t, HIDDEN_SIZE><<<grid, block, 0, stream>>>(   \
        input.data_ptr<scalar_t>(),             \
        residual.data_ptr<scalar_t>(),          \
        weight.data_ptr<scalar_t>(),            \
        epsilon,                                \
        num_tokens);                            \
    });

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
  const cudaStream_t& stream = at::cuda::getCurrentCUDAStream();
  #ifdef _MATT_DEBUG
  printf("Launching with %d blocks\n", num_tokens);
  #endif
  switch (hidden_size) {
    case 4096:
      LAUNCH_FUSED_ADD_RMS_NORM(4096);
      break;
    case 5120:
      LAUNCH_FUSED_ADD_RMS_NORM(5120);
      break;
    case 8192:
      LAUNCH_FUSED_ADD_RMS_NORM(8192);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", hidden_size);
      break;
  }
}
