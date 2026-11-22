#include "common.cuh"

// the most states (earlier ones + cur) and queries of one depth mix
#define DEPTH_MIX_MAX_STATES 8
#define DEPTH_MIX_MAX_QUERY  4

bool ggml_cuda_depth_mix_supported(const ggml_tensor * dst);

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
