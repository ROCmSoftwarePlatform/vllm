// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp>

//group GEMM
#include <ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_gemm_fixed_nk.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_gemm_multi_abd_fixed_nk.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop_multiply.hpp>

//vanilla GEMM
#include <ck/tensor_operation/gpu/device/device_gemm_splitk.hpp>
#include <ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp>
#include <ck/tensor_operation/gpu/device/device_gemm_multiple_abd.hpp>
#include <ck/library/tensor_operation_instance/gpu/gemm_multi_abd.hpp>

//Host utilities
#include <ck/library/utility/device_memory.hpp>
#include <ck/library/utility/host_tensor.hpp>
#include <ck/library/utility/host_tensor_generator.hpp>
#include <ck/library/utility/literals.hpp>
#include <ck/library/reference_tensor_operation/cpu/reference_gemm.hpp>
#include <ck/library/utility/check_err.hpp>

//#include "common.h"
//#include "pybind11_kernel_helpers.h"

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
        size = mem_size;
    }

    void* GetDeviceBuffer() { return p_mem_; }
    size_t GetDeviceBufferSize() { return size; }

    ~SimpleDeviceMem() {
        (void)hipFree(p_mem_);
    }

    void* p_mem_;
    size_t size;
};

using CK_I8   = int8_t;
using CK_BF16 = ck::bhalf_t;
using CK_F32  = float;
using CK_F16  = ck::half_t;
using CK_F8   = ck::f8_t;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Multiply      = ck::tensor_operation::element_wise::Multiply;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;


#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <iostream>

template<typename tensor_type>
int calc_unique_and_counts(tensor_type* d_array, std::vector<tensor_type> &h_unique, std::vector<tensor_type> &h_counts, int N, cudaStream_t stream) {

    // Wrap the raw pointer with a Thrust device vector
    thrust::device_vector<tensor_type> d_vec(d_array, d_array + N);

    // Sort the device vector
    //thrust::sort(d_vec.begin(), d_vec.end());

    // Remove duplicates and calculate unique counts
    thrust::device_vector<tensor_type> d_unique(N);
    thrust::device_vector<tensor_type> d_counts(N);

    auto end_unique = thrust::unique_copy(thrust::cuda::par.on(stream), d_vec.begin(), d_vec.end(), d_unique.begin());
    auto end_counts = thrust::reduce_by_key(thrust::cuda::par.on(stream), d_vec.begin(), d_vec.end(),
                                            thrust::make_constant_iterator(1),
                                            d_unique.begin(), d_counts.begin()).second;

    cudaStreamSynchronize(stream);
    // Resize vectors to remove unused elements
    d_unique.resize(thrust::distance(d_unique.begin(), end_unique));
    d_counts.resize(thrust::distance(d_counts.begin(), end_counts));

    // Output the results
    h_unique.resize(d_unique.size());
    h_counts.resize(d_unique.size());
    thrust::copy(d_unique.begin(), d_unique.end(), h_unique.begin());
    thrust::copy(d_counts.begin(), d_counts.end(), h_counts.begin());
        
    return (int)d_unique.size();
}

void calc_ms_and_uniq_m(int num_experts, std::vector<int> &ms_host_w_zeros, std::vector<int> &ms_host, std::vector<int> &unique_m, cudaStream_t stream) {
    //auto start = std::chrono::steady_clock::now();
    cudaStreamSynchronize(stream);
    //calc_unique_and_counts(ms_dev_ptr, unique_m, ms_dev_ptr, num_index, stream);
    for (int i = 0; i < num_experts; i++) {
        if (ms_host_w_zeros[i] != 0) {
            unique_m.push_back(i);
            ms_host.push_back(ms_host_w_zeros[i]);
        }
    }
}

template<ck::index_t NumDTensor = 0, typename A0DataType, typename B0DataType, typename A0Layout, typename B0Layout, typename ELayout>
__global__ void compute_group_gemm_kerne_arg(
    void *_group_gemm_arg_dev_ptr,
    int group_count,
    A0DataType* a_dev_ptr,
    B0DataType* b_dev_ptr,
    A0DataType* b_scale_dev_ptr,
    A0DataType* bias_dev_ptr,
    A0DataType* e_dev_ptr, int num_experts, int* ms_dev_ptr, int n, int k) {

    const int tid = threadIdx.x;
    const int gid = blockDim.x * blockIdx.x + tid;
    ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>* grouped_gemm_kernel_args = static_cast<ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>*>(_group_gemm_arg_dev_ptr);
    extern __shared__ ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor> s_grouped_gemm_kernel_args[];
    extern __shared__ int prefixIndxSum[];
    extern __shared__ int prefixMsSum[];
    int m = (tid < num_experts) ? ms_dev_ptr[tid] : 0;
    int m_base = m;

    if (tid < num_experts) {
        s_grouped_gemm_kernel_args[tid] = ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>();
    }

    // Perform exclusive scan within the block
    if (warpSize > num_experts) {
        for (int offset = 1; offset < warpSize; offset *= 2) {
            int t_m_base = __shfl_up(m_base, offset);

            if (threadIdx.x >= offset) {
                m_base += t_m_base;
            }
        }
        __syncthreads();
    } else {
        prefixMsSum[threadIdx.x] = m_base;
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            if (threadIdx.x >= offset) {
                int t_m_base = prefixMsSum[threadIdx.x - offset];
                m_base += t_m_base;
                __syncthreads();
                prefixMsSum[threadIdx.x] = m_base;
                __syncthreads();
            }
        }
    }
    if (tid < num_experts) {
        m_base -= m;
        //printf("Thread %d final value = %d, %d\n", tid, m, m_base);
        s_grouped_gemm_kernel_args[tid] = ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>({
                a_dev_ptr + m_base * k,
                b_dev_ptr + tid * n * k,
                {b_scale_dev_ptr + tid * n},
                e_dev_ptr + m_base * n,
                m,
                n,
                k,
                std::is_same<Row, A0Layout>::value ? k : m,
                std::is_same<Row, B0Layout>::value ? n : k,
                {0},
                std::is_same<Row, ELayout>::value ? n : k
        });
        grouped_gemm_kernel_args[tid] = s_grouped_gemm_kernel_args[tid];
    }
}

//void ck_quant_group_gemm(torch::Tensor& out, torch::Tensor& A, torch::Tensor& B, 
//		torch::Tensor& B_Scale, torch::Tensor& Ms, torch::Tensor& Bias,
//		int num_ms, int num_experts, bool has_bias, bool do_time,
//		bool has_b_scales, bool has_gelu_act)
void ck_quant_group_gemm(torch::Tensor& out, torch::Tensor& A, torch::Tensor& B, 
		torch::Tensor& B_Scale, torch::Tensor& Ms)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int total_m = A.size(0);
    int k = A.size(1);
    int n = B.size(2);
    //int num_ms = d.num_ms;
    //int num_experts = d.num_experts;
    //int has_b_scales = d.has_b_scales;
    //int has_bias = d.has_bias;
    //int has_gelu_act = d.has_gelu_act;
    //int do_verification = d.do_verification;
    //int do_time = d.do_time;
    //ElementTypeGemm a_type = d.a_type;
    //ElementTypeGemm b_type = d.b_type;
    //Layout a_major = d.a_major;
    //Layout b_major = d.b_major;

    int has_bias = 0;
    int has_gelu_act = 0;
    int do_time = 0;

    int has_b_scales = 1;

    int num_experts = 8;


    //printf("(m, k, n, n_idx, n_experts, a_type, b_type, has_b_scales, has_bias, has_gelu_act) = (%d, %d, %d, %d, %d, %d, %d, %d, %d, %d)\n", m, k, n, num_ms, num_experts, a_type, b_type, has_b_scales, has_bias, has_gelu_act);

    void *a_dev_ptr { static_cast<void *>(A.data_ptr()) };
    void *b_dev_ptr { static_cast<void *>(B.data_ptr()) };
    void *b_scale_dev_ptr { static_cast<void *>(B_Scale.data_ptr()) };
    void *bias_dev_ptr;
    int *ms_dev_ptr;
    void *e_dev_ptr;

    if (has_bias) {
        //bias_dev_ptr = static_cast<void *>(Bias.data_ptr());
        ms_dev_ptr = static_cast<int *>(Ms.data_ptr());
        e_dev_ptr = static_cast<void *>(out.data_ptr());
    } else {
        ms_dev_ptr = static_cast<int *>(Ms.data_ptr());
        e_dev_ptr = static_cast<void *>(out.data_ptr());
    }

    std::vector<int> unique_m;
    std::vector<int> ms_host;

    auto do_group_gemm = 
        []<typename ALayout, typename BLayout, typename DsLayout, typename ADataType, typename BDataType, typename DsDataType>(
            ADataType* a_dev_ptr, BDataType* b_dev_ptr, ADataType* e_dev_ptr,
            int num_experts, int* ms_dev_ptr, int n, int k, int kBatch, cudaStream_t stream) {

            using ELayout = ALayout;
            using EDataType = ADataType;

            int sum_of_m = 0;

            std::vector<int> Ms, Ns, Ks, StrideAs, StrideBs, StrideEs;
            std::vector<int> ms_host;
            std::vector<int> unique_m;
            int group_count;
            std::vector<int> ms_host_w_zeros(num_experts);
            hip_check_error(hipMemcpyAsync(ms_host_w_zeros.data(),
                                      ms_dev_ptr,
                                      num_experts*sizeof(int),
                                      hipMemcpyDeviceToHost, stream));

            calc_ms_and_uniq_m(num_experts, ms_host_w_zeros, ms_host, unique_m, stream);
            group_count = unique_m.size();


            for(int i = 0; i < group_count; ++i)
            {
                Ms.push_back(ms_host[i]);
                Ns.push_back(n);
                Ks.push_back(k);

                StrideAs.push_back(std::is_same<Row, ALayout>::value ? k : Ms[i]);
                StrideBs.push_back(std::is_same<Row, BLayout>::value ? n : k);
                StrideEs.push_back(std::is_same<Row, ELayout>::value ? n : k);

                sum_of_m += Ms[i];
            }

            std::vector<void*> p_e;

            p_e.reserve(group_count);

            std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;

            gemm_descs.reserve(group_count);

            std::vector<ck::tensor_operation::device::GroupedGemmKernelArgument<1>>
                grouped_gemm_kernel_args_;
            grouped_gemm_kernel_args_.reserve(group_count);

            int m_base = 0;
            for(int i = 0; i < group_count; ++i)
            {
                gemm_descs.push_back({sum_of_m, Ns[i], Ks[i], 1, StrideBs[i], 1, {0}});

                p_e.push_back(e_dev_ptr);

                grouped_gemm_kernel_args_.push_back({a_dev_ptr + m_base * k,
                        b_dev_ptr + unique_m[i] * n * k,
                        {},
                        e_dev_ptr + m_base * n,
                        Ms[i],
                        n,
                        k,
                        StrideAs[i],
                        StrideBs[i],
                        {},
                        StrideEs[i]});
                m_base += Ms[i];
            }

            using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmFixedNK<ALayout,
                  BLayout,
                  DsLayout,
                  ELayout,
                  ADataType,
                  BDataType,
                  DsDataType,
                  EDataType,
                  PassThrough,
                  PassThrough,
                  PassThrough>;

            // get device op instances
            const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                DeviceOp>::GetInstances();

            //std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

            const auto a_element_op   = PassThrough{};
            const auto b_element_op   = PassThrough{};
            const auto cde_element_op = PassThrough{};

            std::string best_op_name;
            bool found            = false;
            int best_op_id        = -1;
            float best_ave_time   = 0;
            float best_tflops     = 0;
            float best_gb_per_sec = 0;

            // profile device operation instances
            //std::cout << "Run all instances and do timing" << std::endl;

            std::vector<const void*> p_a = {}, p_b = {};
            std::vector<std::array<const void*, 0>> p_ds = {};

            for(int i = 0; i < op_ptrs.size(); ++i)
            {
                auto& op_ptr = op_ptrs[i];

                auto argument_ptr = op_ptr->MakeArgumentPointer(
                        p_a, p_b, p_ds, p_e, gemm_descs, a_element_op, b_element_op, cde_element_op);

                auto invoker_ptr = op_ptr->MakeInvokerPointer();

                SimpleDeviceMem grouped_gemm_workspace_dev(op_ptr->GetWorkSpaceSize(argument_ptr.get()));

                std::string op_name = op_ptr->GetTypeString();

                thread_local static SimpleDeviceMem gemm_kernel_args_dev(1024);
                if (gemm_kernel_args_dev.GetDeviceBufferSize() < op_ptr->GetDeviceKernelArgSize(argument_ptr.get())) {
                    SimpleDeviceMem _gemm_kernel_args_dev(op_ptr->GetDeviceKernelArgSize(argument_ptr.get()));
                    gemm_kernel_args_dev = std::move(_gemm_kernel_args_dev);
                }

                hipGetErrorString(hipMemcpy(gemm_kernel_args_dev.GetDeviceBuffer(),
                            grouped_gemm_kernel_args_.data(),
                            op_ptr->GetDeviceKernelArgSize(argument_ptr.get()),
                            hipMemcpyHostToDevice));

                op_ptr->SetWorkSpacePointer(argument_ptr.get(),
                        grouped_gemm_workspace_dev.GetDeviceBuffer());

                op_ptr->SetDeviceKernelArgs(argument_ptr.get(),
                        gemm_kernel_args_dev.GetDeviceBuffer());

                op_ptr->SetKBatch(argument_ptr.get(), kBatch);

                if(op_ptr->IsSupportedArgument(argument_ptr.get()))
                {
                    float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});

                    std::size_t flop = 0, num_btype = 0;
                    for(std::size_t j = 0; j < gemm_descs.size(); ++j)
                    {
                        flop += std::size_t(2) * Ms[j] * Ns[j] * Ks[j];

                        num_btype += sizeof(ADataType) * Ms[j] * Ks[j] + sizeof(BDataType) * Ks[j] * Ns[j] +
                            sizeof(EDataType) * Ms[j] * Ns[j];
                    }

                    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
                    float gb_per_sec = num_btype / 1.E6 / ave_time;

                    //std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                    //    << gb_per_sec << " GB/s, " << op_name << std::endl;

                    break;
                }
            }
        };
    
    auto do_group_gemm_tile_loop = 
        []<typename A0Layout, typename B0Layout, typename BsLayout, typename D0Layout, typename DsLayout,
            typename A0DataType, typename B0DataType, typename BsDataType, typename D0DataType, typename DsDataType,
            typename AElementOp, typename BElementOp, typename CDEElementOp,
            ck::index_t NumATensor, ck::index_t NumBTensor, ck::index_t NumDTensor>(
            A0DataType* a_dev_ptr, B0DataType* b_dev_ptr, A0DataType* b_scale_dev_ptr, A0DataType* bias_dev_ptr, A0DataType* e_dev_ptr,
            int num_experts, int* ms_dev_ptr, int total_m, int n, int k, cudaStream_t stream, bool do_time) {

            using AsDataType       = ck::Tuple<A0DataType>;
            using AccDataType      = CK_F32;
            using CShuffleDataType = A0DataType;
            using EDataType = A0DataType;

            using AsLayout = ck::Tuple<A0Layout>;
            using B1Layout = B0Layout;
            using ELayout = A0Layout;

            using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmTileLoop<A0Layout,
                                                                                     B0Layout,
                                                                                     DsLayout,
                                                                                     Row,
                                                                                     A0DataType,
                                                                                     B0DataType,
                                                                                     DsDataType,
                                                                                     CK_BF16,
                                                                                     AElementOp,
                                                                                     BElementOp,
                                                                                     CDEElementOp>;
            using GroupedGemmKernelArgument = ck::tensor_operation::device::GroupedGemmTileLoopKernelArguments<NumDTensor>;

            std::vector<GroupedGemmKernelArgument> grouped_gemm_kernel_args_;
            const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOp>::GetInstances();
            std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
            int group_count;

            if (do_time) {
                // Configure gemm_desc to make IsSupportedArgument() work
                std::vector<int> ms_host;
                std::vector<int> unique_m;
                std::vector<int> ms_host_w_zeros(num_experts);
                std::vector<int> Ms, Ns, Ks, StrideAs, StrideBs, StrideEs;

                hip_check_error(hipMemcpyAsync(ms_host_w_zeros.data(),
                                          ms_dev_ptr,
                                          num_experts*sizeof(int),
                                          hipMemcpyDeviceToHost, stream));


                calc_ms_and_uniq_m(num_experts, ms_host_w_zeros, ms_host, unique_m, stream);
		group_count = unique_m.size();
                for(int i = 0; i < group_count; i++) {
                    Ms.push_back(ms_host[i]);
                    Ns.push_back(n);
                    Ks.push_back(k);

                    StrideAs.push_back(std::is_same<Row, A0Layout>::value ? k : Ms[i]);
                    StrideBs.push_back(std::is_same<Row, B0Layout>::value ? n : k);
                    StrideEs.push_back(std::is_same<Row, ELayout>::value ? n : k);

                    gemm_descs.push_back({Ms[i], Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideEs[i], {0}});
                }
            } else {
            // Filled up blanks for configuring group counts only since we configure kernel args in compute_group_gemm_kerne_arg
                for(int i = 0; i < num_experts; i++) {
                    gemm_descs.push_back({0, 0, 0, 0, 0, 0, {0}});
                }
            }

            auto a_element_op   = AElementOp{};
            auto b_element_op   = BElementOp{};
            auto cde_element_op = CDEElementOp{};
            if (do_time) {
                std::cout << "found " << op_ptrs.size() << " instances" << std::endl;
            }

            std::string best_op_name;
            bool found            = false;
            int best_op_id        = -1;
            float best_ave_time   = 0;
            float best_tflops     = 0;
            float best_gb_per_sec = 0;

            int sol_idx = 0;

            if (!do_time && std::is_same<Row, A0Layout>::value && std::is_same<Row, B0Layout>::value && std::is_same<PassThrough, BElementOp>::value && std::is_same<Multiply, CDEElementOp>::value) {
                //if (total_m == 8) {
                //    sol_idx = 10;
                //} else {
                //    sol_idx = 2;
                //}
                sol_idx = 10;

		//std::cout << "sol_idx:" << sol_idx << std::endl;
            }

            for(int i = sol_idx; i < op_ptrs.size(); ++i) {
                auto& op_ptr = op_ptrs[i];
                std::vector<const void*> p_As                         = {};
                std::vector<const void*> p_Bs                         = {};
                std::vector<std::array<const void*, NumDTensor>> p_Ds = {};
                std::vector<void*> p_Cs                               = {};

                auto argument_ptr = op_ptr->MakeArgumentPointer(p_As, p_Bs, p_Ds, p_Cs, gemm_descs, a_element_op, b_element_op, cde_element_op);
                auto invoker_ptr = op_ptr->MakeInvokerPointer();

                std::string op_name = op_ptr->GetTypeString();
                bool isSupported = true;

                if (do_time) {
                    isSupported = op_ptr->IsSupportedArgument(argument_ptr.get());
                }

                if(isSupported)
                {
                    //std::cout << i << ", " << op_name << " support this problem" << std::endl;
                    thread_local static SimpleDeviceMem gemm_kernel_args_dev(1024);
                    if (gemm_kernel_args_dev.GetDeviceBufferSize() < op_ptr->GetDeviceKernelArgSize(argument_ptr.get())) {
                        SimpleDeviceMem _gemm_kernel_args_dev(op_ptr->GetDeviceKernelArgSize(argument_ptr.get()));
                        gemm_kernel_args_dev = std::move(_gemm_kernel_args_dev);
                    }

                    int threadsPerBlock = 512;
                    int gridSize = 1;//(num_experts + threadsPerBlock - 1) / threadsPerBlock;
                    size_t ldsSizePerBlock = num_experts * sizeof(ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>) + threadsPerBlock * sizeof(int);

                    compute_group_gemm_kerne_arg<NumDTensor, A0DataType, B0DataType, A0Layout, B0Layout, ELayout><<<gridSize, threadsPerBlock, ldsSizePerBlock, stream>>>(
                        gemm_kernel_args_dev.GetDeviceBuffer(),
                        group_count,
                        a_dev_ptr,
                        b_dev_ptr,
                        b_scale_dev_ptr,
                        bias_dev_ptr,
                        e_dev_ptr, num_experts, ms_dev_ptr, n, k);

                    op_ptr->SetDeviceKernelArgs(argument_ptr.get(), gemm_kernel_args_dev.GetDeviceBuffer());

                    float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, do_time});
                    std::size_t flop = std::size_t(2) * total_m * n * k;

                    std::size_t num_btype = sizeof(A0DataType) * total_m * k +
                                            sizeof(B0DataType) * n * k +
                                            sizeof(EDataType) * total_m * n;

                    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                    float gb_per_sec = num_btype / 1.E6 / ave_time;

                    if(tflops > best_tflops)
                    {
                        found           = true;
                        best_op_id      = i;
                        best_op_name    = op_name;
                        best_tflops     = tflops;
                        best_ave_time   = ave_time;
                        best_gb_per_sec = gb_per_sec;
                    }
                    if (do_time) {
                        std::cout << i << ", Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                                  << gb_per_sec << " GB/s, " << op_name << std::endl;
                    } else {
                        break;
                    }
                }
            }
        };

    //do_group_gemm_tile_loop.operator()<Row, Row, ck::Tuple<Row>, Row, ck::Tuple<Row>,
    //    CK_F16, CK_F8, ck::Tuple<CK_I8>, CK_F16, ck::Tuple<CK_F16>,
    //    PassThrough, PassThrough, Multiply, 1, 1, 1>(
    //        static_cast<CK_F16 *>(a_dev_ptr), static_cast<CK_F8 *>(b_dev_ptr),
    //        static_cast<CK_F8 *>(b_scale_dev_ptr), static_cast<CK_F16 *>(bias_dev_ptr),
    //        static_cast<CK_F16 *>(e_dev_ptr), num_experts, ms_dev_ptr, total_m, n , k, stream, do_time);
    do_group_gemm_tile_loop.operator()<Row, Row, ck::Tuple<Row>, Row, ck::Tuple<Row>,
        CK_BF16, CK_I8, ck::Tuple<CK_I8>, CK_BF16, ck::Tuple<CK_BF16>,
        PassThrough, PassThrough, Multiply, 1, 1, 1>(
            static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr),
            static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
            static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, total_m, n , k, stream, do_time);




    //if (has_b_scales && !has_bias && has_gelu_act) {
    //    if (a_major == Layout::RowMajor && b_major == Layout::RowMajor) {
    //        if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_BF16) {
    //            do_group_gemm_multi_abd.operator()<Row, Row, ck::Tuple<Row, Row>, Row, ck::Tuple<>,
    //                CK_BF16, CK_BF16, ck::Tuple<CK_BF16, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                PassThrough, Multiply, FastGelu, 1, 2, 0>(
    //                    static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_BF16 *>(b_dev_ptr),
    //                    static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                    static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //        } else if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_I8) {
    //            do_group_gemm_multi_abd.operator()<Row, Row, ck::Tuple<Row, Row>, Row, ck::Tuple<>,
    //                CK_BF16, CK_I8, ck::Tuple<CK_I8, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                PassThrough, Multiply, FastGelu, 1, 2, 0>(
    //                    static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr),
    //                    static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                    static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //        }
    //    } else if (a_major == Layout::RowMajor && b_major == Layout::ColumnMajor) {
    //        if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_BF16) {
    //            do_group_gemm_multi_abd.operator()<Row, Col, ck::Tuple<Col, Col>, Row, ck::Tuple<>,
    //                CK_BF16, CK_BF16, ck::Tuple<CK_BF16, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                PassThrough, Multiply, FastGelu, 1, 2, 0>(
    //                    static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_BF16 *>(b_dev_ptr),
    //                    static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                    static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //        } else if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_I8) {
    //            do_group_gemm_multi_abd.operator()<Row, Col, ck::Tuple<Col, Col>, Row, ck::Tuple<>,
    //                CK_BF16, CK_I8, ck::Tuple<CK_I8, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                PassThrough, Multiply, FastGelu, 1, 2, 0>(
    //                    static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr),
    //                    static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                    static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //        }
    //    } else if (a_major == Layout::ColumnMajor && b_major == Layout::RowMajor) {
    //        printf("No implementation for a_major == Layout::ColumnMajor && b_major == Layout::RowMajor\n");
    //        abort();
    //    } else if (a_major == Layout::ColumnMajor && b_major == Layout::ColumnMajor) {
    //        printf("No implementation for a_major == Layout::ColumnMajor && b_major == Layout::ColumnMajor\n");
    //        abort();
    //    }
    //} else if (has_b_scales && !has_bias && !has_gelu_act) {
    //    if (a_major == Layout::RowMajor && b_major == Layout::RowMajor) {
    //        if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_BF16) {
    //            do_group_gemm_multi_abd.operator()<Row, Row, ck::Tuple<Row, Row>, Row, ck::Tuple<>,
    //                CK_BF16, CK_BF16, ck::Tuple<CK_BF16, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                PassThrough, Multiply, PassThrough, 1, 2, 0>(
    //                    static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_BF16 *>(b_dev_ptr),
    //                    static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                    static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //        } else if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_I8) {
    //            if (1) {
    //                do_group_gemm_tile_loop.operator()<Row, Row, ck::Tuple<Row>, Row, ck::Tuple<Row>,
    //                    CK_BF16, CK_I8, ck::Tuple<CK_I8>, CK_BF16, ck::Tuple<CK_BF16>,
    //                    PassThrough, PassThrough, Multiply, 1, 1, 1>(
    //                        static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr),
    //                        static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                        static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, total_m, n , k, stream, do_time);
    //            } else {
    //                do_group_gemm_multi_abd.operator()<Row, Row, ck::Tuple<Row, Row>, Row, ck::Tuple<>,
    //                    CK_BF16, CK_I8, ck::Tuple<CK_I8, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                    PassThrough, Multiply, PassThrough, 1, 2, 0>(
    //                        static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr),
    //                        static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                        static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //            }
    //        }
    //    } else if (a_major == Layout::RowMajor && b_major == Layout::ColumnMajor) {
    //        if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_BF16) {
    //            do_group_gemm_multi_abd.operator()<Row, Col, ck::Tuple<Col, Col>, Row, ck::Tuple<>,
    //                CK_BF16, CK_BF16, ck::Tuple<CK_BF16, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                PassThrough, Multiply, PassThrough, 1, 2, 0>(
    //                    static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_BF16 *>(b_dev_ptr),
    //                    static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                    static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //        } else if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_I8) {
    //            do_group_gemm_multi_abd.operator()<Row, Col, ck::Tuple<Col, Col>, Row, ck::Tuple<>,
    //                CK_BF16, CK_I8, ck::Tuple<CK_I8, CK_BF16>, CK_BF16, ck::Tuple<>,
    //                PassThrough, Multiply, PassThrough, 1, 2, 0>(
    //                    static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr),
    //                    static_cast<CK_BF16 *>(b_scale_dev_ptr), static_cast<CK_BF16 *>(bias_dev_ptr),
    //                    static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, stream, do_time);
    //        }
    //    } else if (a_major == Layout::ColumnMajor && b_major == Layout::RowMajor) {
    //        printf("No implementation for a_major == Layout::ColumnMajor && b_major == Layout::RowMajor\n");
    //        abort();
    //    } else if (a_major == Layout::ColumnMajor && b_major == Layout::ColumnMajor) {
    //        printf("No implementation for a_major == Layout::ColumnMajor && b_major == Layout::ColumnMajor\n");
    //        abort();
    //    }
    //} else if (!has_b_scales && !has_bias && !has_gelu_act) {
    //    if (a_major == Layout::RowMajor && b_major == Layout::RowMajor) {
    //        if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_BF16) {
    //            do_group_gemm.operator()<Row, Row, ck::Tuple<>, CK_BF16, CK_BF16, ck::Tuple<>>(
    //                static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_BF16 *>(b_dev_ptr), static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, 1, stream);
    //        } else if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_I8) {
    //            do_group_gemm.operator()<Row, Row, ck::Tuple<>, CK_BF16, CK_I8, ck::Tuple<>>(
    //                static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr), static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, 1, stream);
    //        }
    //    } else if (a_major == Layout::RowMajor && b_major == Layout::ColumnMajor) {
    //        if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_BF16) {
    //            do_group_gemm.operator()<Row, Col, ck::Tuple<>, CK_BF16, CK_BF16, ck::Tuple<>>(
    //                static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_BF16 *>(b_dev_ptr), static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, 1, stream);
    //        } else if (a_type == ElementTypeGemm::QGEMM_BF16 && b_type == ElementTypeGemm::QGEMM_I8) {
    //            do_group_gemm.operator()<Row, Col, ck::Tuple<>, CK_BF16, CK_I8, ck::Tuple<>>(
    //                static_cast<CK_BF16 *>(a_dev_ptr), static_cast<CK_I8 *>(b_dev_ptr), static_cast<CK_BF16 *>(e_dev_ptr), num_experts, ms_dev_ptr, n , k, 1, stream);
    //        }
    //    } else if (a_major == Layout::ColumnMajor && b_major == Layout::RowMajor) {
    //        printf("No implementation for a_major == Layout::ColumnMajor && b_major == Layout::RowMajor\n");
    //        abort();
    //    } else if (a_major == Layout::ColumnMajor && b_major == Layout::ColumnMajor) {
    //        printf("No implementation for a_major == Layout::ColumnMajor && b_major == Layout::ColumnMajor\n");
    //        abort();
    //    }
    //} else {
    //    printf("No implementation for (has_b_scales, has_bias, has_gelu_act) = (%d, %d, %d)\n", has_b_scales, has_bias, has_gelu_act);
    //    abort();
    //}
    fflush(stdout);
    return;
}
