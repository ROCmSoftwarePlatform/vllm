#pragma once

#include <torch/all.h>

void LLMM_Silu(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
               const int64_t rows_per_block);

void LLMM1(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
           const int64_t rows_per_block);

void wvSpltK(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
             const int64_t N_in, const int64_t CuCount);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens,
    int64_t block_size, int64_t max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale, int64_t partition_size);

void fused_rotary_embedding_and_reshape_cache(
        torch::Tensor& positions, torch::Tensor& query,
        torch::Tensor& key, torch::Tensor& value, 
        torch::Tensor& key_cache, torch::Tensor& value_cache,
        const std::string& kv_cache_dtype,
        torch::Tensor& cos_sin_cache,torch::Tensor& slot_mapping,
        const int64_t head_size, const double k_scale, const double v_scale,
        bool is_neox);
