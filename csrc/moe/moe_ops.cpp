#include "moe_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &topk_softmax,
        "Apply topk softmax to the gating outputs.");
#ifdef USE_ROCM
  m.def("ck_quant_group_gemm", &ck_quant_group_gemm,
        "group gemm.");
#endif
}
