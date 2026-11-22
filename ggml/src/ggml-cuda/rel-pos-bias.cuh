#include "common.cuh"

bool ggml_cuda_rel_pos_bias_supported(const ggml_tensor * dst);

void ggml_cuda_op_rel_pos_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
