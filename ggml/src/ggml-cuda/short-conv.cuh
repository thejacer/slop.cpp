#include "common.cuh"

bool ggml_cuda_short_conv_supported(const ggml_tensor * dst);

void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
