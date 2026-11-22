#include "common.cuh"

bool ggml_cuda_loop_ctrl_supported(const ggml_tensor * dst);

void ggml_cuda_op_loop_ctrl(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
