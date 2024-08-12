#pragma once

#include <torch/extension.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

#ifdef USE_ROCM
void ck_quant_group_gemm(torch::Tensor& out, torch::Tensor& A, torch::Tensor& B,
                torch::Tensor& B_Scale, torch::Tensor& Ms);
//void ck_quant_group_gemm(torch::Tensor& out, torch::Tensor& A, torch::Tensor& B,
//                torch::Tensor& B_Scale, torch::Tensor& Ms, torch::Tensor& Bias,
//                int num_ms, int num_experts, bool has_bias, bool do_time,
//                bool has_b_scales, bool has_gelu_act);
#endif
