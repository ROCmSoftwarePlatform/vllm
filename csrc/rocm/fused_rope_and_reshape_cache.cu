#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "attention/attention_dtypes.h"
#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
  #include "quantization/fp8/amd/hip_float8.h"
  #include "quantization/fp8/amd/quant_utils.cuh"

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

namespace {

template <typename scalar_t, int width>
struct __align__(16) vec_t;

template <typename scalar_t, int width>
__device__ void apply_rope(scalar_t* __restrict__ arr_ptr,
                          const scalar_t* __restrict__ sin_ptr,
                          const scalar_t* __restrict__ cos_ptr,
                          int rot_offset, int embed_dim, 
                          const bool IS_NEOX,
                          vec_t<scalar_t, width>& __restrict__ out_xvec,
                          vec_t<scalar_t, width>& __restrict__ out_yvec);

template <typename scalar_t, int width>
struct __align__(16) vec_t {
  scalar_t data[width];

  __device__ vec_t() = default;
  __device__ vec_t(const vec_t<scalar_t, width>& __restrict__ other) {
#pragma unroll
    for (int i = 0; i < width; ++i) data[i] = other.data[i];
  }

  __device__ vec_t operator*(const vec_t& __restrict__ other) const {
    vec_t<scalar_t, width> tmp{*this};
#pragma unroll
    for (int i = 0; i < width; ++i) tmp.data[i] *= other.data[i];
    return tmp;
  }

  __device__ vec_t operator*(const float& scale) const {
    vec_t<scalar_t, width> tmp{*this};
#pragma unroll
    for (int i = 0; i < width; ++i) tmp.data[i] *= scale;
    return tmp;
  }

  __device__ vec_t operator+(const vec_t& __restrict__ other) const {
    vec_t<scalar_t, width> tmp{*this};
#pragma unroll
    for (int i = 0; i < width; ++i) tmp.data[i] += other.data[i];
    return tmp;
  }

  __device__ vec_t operator-(const vec_t& __restrict__ other) const {
    vec_t<scalar_t, width> tmp{*this};
#pragma unroll
    for (int i = 0; i < width; ++i) tmp.data[i] -= other.data[i];
    return tmp;
  }

  __device__ vec_t<scalar_t, width>& operator=(const vec_t& __restrict__ other) {
#pragma unroll
    for (int i = 0; i < width; ++i) data[i] = other.data[i];
    return *this;
  }

  __device__ vec_t<scalar_t, width>& operator+=(const vec_t& __restrict__ other) {
#pragma unroll
    for (int i = 0; i < width; ++i) data[i] += other.data[i];
    return *this;
  }

  __device__ scalar_t& operator [](const size_t& idx) {
    return data[idx];
  }

  __device__ scalar_t operator [](const size_t& idx) const {
    return data[idx];
  }

  friend
  __device__ inline void apply_rope<scalar_t, width>(
                                    scalar_t* __restrict__ arr_ptr,
                                    const scalar_t* __restrict__ sin_ptr,
                                    const scalar_t* __restrict__ cos_ptr,
                                    int rot_offset, int embed_dim,
                                    const bool IS_NEOX,
                                    vec_t<scalar_t, width>& __restrict__ out_xvec,
                                    vec_t<scalar_t, width>& __restrict__ out_yvec);
};


template <typename scalar_t, int width>
__device__ inline void apply_rope(scalar_t* __restrict__ arr_ptr,
                                  const scalar_t* __restrict__ cos_ptr,
                                  const scalar_t* __restrict__ sin_ptr,
                                  int rot_offset, int embed_dim,
                                  const bool IS_NEOX,
                                  vec_t<scalar_t, width>& __restrict__ out_xvec,
                                  vec_t<scalar_t, width>& __restrict__ out_yvec) {
    const vec_t<scalar_t, width> xvec = 
          *reinterpret_cast<vec_t<scalar_t, width> *>(arr_ptr + rot_offset);
    const vec_t<scalar_t, width> yvec = 
          *reinterpret_cast<vec_t<scalar_t, width> *>(arr_ptr + embed_dim + rot_offset);

    if (IS_NEOX) {
      const vec_t<scalar_t, width> cos = 
            *reinterpret_cast<const vec_t<scalar_t, width> *>(cos_ptr + rot_offset);
      const vec_t<scalar_t, width> sin = 
            *reinterpret_cast<const vec_t<scalar_t, width> *>(sin_ptr + rot_offset);
#pragma unroll
      for (int i = 0; i < width; ++i) {
        out_xvec[i] = xvec[i] * cos[i] - yvec[i] * sin[i];
        out_yvec[i] = yvec[i] * cos[i] + xvec[i] * sin[i];
      }
    } else {
      const vec_t<scalar_t, width> xcos = 
            *reinterpret_cast<const vec_t<scalar_t, width> *>(cos_ptr + rot_offset / 2);
      const vec_t<scalar_t, width> xsin = 
            *reinterpret_cast<const vec_t<scalar_t, width> *>(sin_ptr + rot_offset / 2);
#pragma unroll
      for (int i = 0; i < width / 2; ++i) {
        int x_i = 2 * i;
        int y_i = 2 * i + 1;
        out_xvec[x_i] = xvec[x_i] * xcos[i] - xvec[y_i] * xsin[i];
        out_xvec[y_i] = xvec[y_i] * xcos[i] + xvec[x_i] * xsin[i];
      }

      const vec_t<scalar_t, width> ycos = 
          *reinterpret_cast<const vec_t<scalar_t, width> *>(cos_ptr + (embed_dim + rot_offset) / 2);
      const vec_t<scalar_t, width> ysin = 
          *reinterpret_cast<const vec_t<scalar_t, width> *>(sin_ptr + (embed_dim + rot_offset) / 2);

#pragma unroll
      for (int i = 0; i < width / 2; ++i) {
        int x_i = 2 * i;
        int y_i = 2 * i + 1;
        out_yvec[x_i] = yvec[x_i] * ycos[i] - yvec[y_i] * ysin[i];
        out_yvec[y_i] = yvec[y_i] * ycos[i] + yvec[x_i] * ysin[i];
      }
    }
}

template <typename scalar_t, int width, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__device__ void store_value_into_cache(
                          cache_t* __restrict__ cache,
                          const int head_idx, const int head_offset,
                          const int head_size, const int num_kv_heads,
                          const int64_t block_idx, const int block_size,
                          const int64_t block_offset, const int x,
                          vec_t<scalar_t, width>& val,
                          const float kv_scale) {
  const int x_idx = head_offset / x;
  const int x_offset = head_offset % x;

  const int64_t tgt_idx = 
      block_idx * num_kv_heads * (head_size / x) * block_size * x +
      head_idx * (head_size / x) * block_size * x + 
      x_idx * block_size * x +
      block_offset * x + x_offset;

  if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
    *reinterpret_cast<vec_t<scalar_t, width> *>(cache + tgt_idx) = val;
  } else {
    *reinterpret_cast<vec_t<scalar_t, width> *>(cache + tgt_idx) =
        vllm::fp8::scaled_convert<cache_t, scalar_t, kv_dt>(val, kv_scale);
  }
}

} // anonymous namespace

namespace vllm {

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt, bool IS_NEOX>
__global__ void __launch_bounds__ (512) fused_rotary_embedding_and_reshape_cache_kernel_vec(
        scalar_t* __restrict__ query,        // [batch_size, seq_len, num_heads, head_size] or 
                                             // [num_tokens, num_heads, head_size]
        scalar_t* __restrict__ key,          // [num_tokens, num_heads, head_size]
        const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
        cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
        cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
        const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
        const int64_t* __restrict__ positions,  // [batch_size, seq_len] or [num_tokens]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t query_stride, const int key_stride, const int value_stride, 
        const int num_heads, const int num_kv_heads, const int head_size, 
        const int rot_dim, const int block_size, const int x, 
        const float k_scale, const float v_scale) {
  
  const int width = 16 / sizeof(scalar_t);
  using vec_t = vec_t<scalar_t, width>;

  // Each thread block is responsible for "width" tokens.
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim / width;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = (i * width) / embed_dim;
    const int rot_offset = (i * width) % embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;

    vec_t& out_xvec = *reinterpret_cast<vec_t *>(query + token_head + rot_offset);
    vec_t& out_yvec = *reinterpret_cast<vec_t *>(query + token_head + embed_dim + rot_offset);

    apply_rope(query + token_head, cos_ptr, sin_ptr, rot_offset,
              embed_dim, IS_NEOX, out_xvec, out_yvec);
  }

  const int64_t slot_idx = block_size == 0 ? 0: slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int nk = num_kv_heads * embed_dim / width;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = (i * width) / embed_dim;
    const int rot_offset = (i * width) % embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;

    // FIXME check why do we need to modify key
    vec_t& out_xvec = *reinterpret_cast<vec_t *>(key + token_head + rot_offset);
    vec_t& out_yvec = *reinterpret_cast<vec_t *>(key + token_head + embed_dim + rot_offset);

    apply_rope(key + token_head, cos_ptr, sin_ptr, rot_offset,
              embed_dim, IS_NEOX, out_xvec, out_yvec);

    if (block_size != 0) {
      store_value_into_cache<scalar_t, width, cache_t, kv_dt>
                            (key_cache, head_idx, rot_offset,
                            head_size, num_kv_heads,
                            block_idx, block_size, block_offset, 
                            x, out_xvec, k_scale);
      store_value_into_cache<scalar_t, width, cache_t, kv_dt>
                            (key_cache, head_idx, embed_dim + rot_offset,
                            head_size, num_kv_heads,
                            block_idx, block_size, block_offset, 
                            x, out_yvec, k_scale);
    }
  }

  const int nv = num_kv_heads * head_size;
  for (int i = threadIdx.x; block_size && i < nv; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;

    const int64_t src_value_idx = token_idx * value_stride + i;
    scalar_t tgt_value = value[src_value_idx];

    const int64_t tgt_idx =
        block_idx * num_kv_heads * head_size * block_size +
        head_idx * head_size * block_size + 
        head_offset * block_size +
        block_offset;

    if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
      value_cache[tgt_idx] = tgt_value;
    } else {
      value_cache[tgt_idx] =
          vllm::fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, v_scale);
    }
  }
}

} // namespace vllm


 // SRC_DTYPE is the stored data type of qkv.
 // CACHE_T is the data type of in KV cache.
 // KV_DT is actual type of data in KV cache
 // IS_NEOX flag to compute positional encodding.
 #define CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE(QKV_T, CACHE_T, KV_DT, IS_NEOX)            \
    vllm::fused_rotary_embedding_and_reshape_cache_kernel_vec<QKV_T, CACHE_T, KV_DT, IS_NEOX>   \
        <<<grid, block, 0, stream>>>(                                                       \
                reinterpret_cast<QKV_T*>(query.data_ptr()),                                 \
                reinterpret_cast<QKV_T*>(key.data_ptr()),                                   \
                reinterpret_cast<QKV_T*>(value.data_ptr()),                                 \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                           \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                         \
                reinterpret_cast<QKV_T*>(cos_sin_cache.data_ptr()),                         \
                positions.data_ptr<int64_t>(),                                              \
                slot_mapping.data_ptr<int64_t>(),                                           \
                query_stride, key_stride, value_stride,                                     \
                num_heads, num_kv_heads, head_size,                                         \
                rot_dim, block_size, x, k_scale, v_scale);


  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename SRC_DTYPE, str CACHE_T, IS_NEOX>.
  #define DISPATCH_ROPE_BY_KV_CACHE_DTYPE(SRC_DTYPE, CACHE_T, IS_NEOX, FN)            \
    if (CACHE_T == "auto") {                                                          \
      if (SRC_DTYPE == at::ScalarType::Half) {                                        \
        FN(c10::Half, c10::Half, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX);           \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                             \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX);   \
      } else {                                                                        \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);        \
      }                                                                               \
    }


void fused_rotary_embedding_and_reshape_cache(
        torch::Tensor& positions,     // [batch_size, seq_len] or [num_tokens]
        torch::Tensor& query,   // [batch_size, seq_len, num_heads * head_size] or
                                // [num_tokens, num_heads * head_size]
        torch::Tensor& key,     // [batch_size, seq_len, num_kv_heads * head_size] or
                                // [num_tokens, num_kv_heads * head_size]
        torch::Tensor& value,   // [batch_size, seq_len, num_heads * head_size] or
                                // [num_tokens, num_heads * head_size]
        torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
        torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
        const std::string& kv_cache_dtype,
        torch::Tensor& cos_sin_cache, // [max_position, rot_dim]
        torch::Tensor& slot_mapping,  // [num_tokens]
        const int64_t head_size,
        const double k_scale,
        const double v_scale,
        bool is_neox) {

  assert(query.scalar_type() == key.scalar_type());

  int64_t num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  int64_t value_stride = value.stride(-2);

  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  auto width = 16 / key.element_size();
  dim3 grid(num_tokens / width);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512) / width, width);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (is_neox) {
    DISPATCH_ROPE_BY_KV_CACHE_DTYPE(key.scalar_type(),
        kv_cache_dtype, true, CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE)
  } else {
    DISPATCH_ROPE_BY_KV_CACHE_DTYPE(key.scalar_type(),
        kv_cache_dtype, false, CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE)
  }
}