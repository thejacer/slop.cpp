#include "rel-pos-bias.cuh"
#include "convert.cuh"

#define REL_POS_BIAS_BLOCK_SIZE 256
#define REL_POS_BIAS_MAX_HEADS  64

// one thread per key and query: the bias column of the distance in registers, dotted with r of each head
// the normed and scaled r and the sinks of the query are staged in shared memory once per block
template <int n_rel, typename T_mask, typename T_dst>
static __global__ void rel_pos_bias(
        const float * __restrict__ r, const float * __restrict__ bias, const int16_t * __restrict__ rel_pos, const float * __restrict__ sinks,
        const T_mask * __restrict__ mask, const float * __restrict__ r_scale, T_dst * __restrict__ dst,
        const int n_kv, const int n_q, const int n_head,
        const int64_t s_r1, const int64_t s_r2, const int64_t s_sinks1, const int64_t s_mask1, const int64_t s_mask3, const float r_eps) {
    __shared__ float s_r[REL_POS_BIAS_MAX_HEADS][n_rel];
    __shared__ float s_sink[REL_POS_BIAS_MAX_HEADS];

    const int q = blockIdx.y;
    const int s = blockIdx.z;
    const int64_t t = (int64_t) s*n_q + q;

    const float * rt = r + t*s_r2;
    for (int h = threadIdx.x; h < n_head; h += blockDim.x) {
        const float * rh = rt + h*s_r1;
        float v[n_rel];
        float sq = 0.0f;
#pragma unroll
        for (int i = 0; i < n_rel; ++i) {
            v[i] = rh[i];
            sq += v[i]*v[i];
        }
        // the l2 norm of r and its scale, as ggml_l2_norm
        const float rs = r_scale ? r_scale[t] : 1.0f;
        const float f  = r_eps > 0.0f ? rs/fmaxf(sqrtf(sq), r_eps) : rs;
#pragma unroll
        for (int i = 0; i < n_rel; ++i) {
            s_r[h][i] = f*v[i];
        }
        s_sink[h] = sinks ? sinks[h + t*s_sinks1] : 0.0f;
    }
    __syncthreads();

    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= n_kv) {
        return;
    }

    float b[n_rel];
    const float * bc = bias + (int64_t) n_rel*rel_pos[k + t*n_kv];
#pragma unroll
    for (int i = 0; i < n_rel; ++i) {
        b[i] = bc[i];
    }
    const float m = mask ? ggml_cuda_cast<float>(mask[k + q*s_mask1 + s*s_mask3]) : 0.0f;

    T_dst * out = dst + k + (int64_t) n_kv*(q + (int64_t) n_q*n_head*s);
    for (int h = 0; h < n_head; ++h) {
        float v = m - s_sink[h];
#pragma unroll
        for (int i = 0; i < n_rel; ++i) {
            v += s_r[h][i]*b[i];
        }
        out[(int64_t) h*n_kv*n_q] = ggml_cuda_cast<T_dst>(v);
    }
}

bool ggml_cuda_rel_pos_bias_supported(const ggml_tensor * dst) {
    const int64_t n_rel = dst->src[0]->ne[0];
    return (n_rel == 8 || n_rel == 16 || n_rel == 32) && dst->ne[2] <= REL_POS_BIAS_MAX_HEADS && dst->ne[1] <= 65535 && dst->ne[3] <= 65535;
}

template <int n_rel, typename T_mask, typename T_dst>
static void rel_pos_bias_launch(const ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * r       = dst->src[0];
    const ggml_tensor * bias    = dst->src[1];
    const ggml_tensor * rel_pos = dst->src[2];
    const ggml_tensor * sinks   = dst->src[3];
    const ggml_tensor * mask    = dst->src[4];
    const ggml_tensor * r_scale = dst->src[5];

    const int n_kv   = dst->ne[0];
    const int n_q    = dst->ne[1];
    const int n_head = dst->ne[2];

    const dim3 grid((n_kv + REL_POS_BIAS_BLOCK_SIZE - 1)/REL_POS_BIAS_BLOCK_SIZE, n_q, dst->ne[3]);

    rel_pos_bias<n_rel, T_mask, T_dst><<<grid, REL_POS_BIAS_BLOCK_SIZE, 0, stream>>>(
            (const float *) r->data, (const float *) bias->data, (const int16_t *) rel_pos->data,
            sinks ? (const float *) sinks->data : nullptr, mask ? (const T_mask *) mask->data : nullptr,
            r_scale ? (const float *) r_scale->data : nullptr, (T_dst *) dst->data,
            n_kv, n_q, n_head, r->nb[1]/sizeof(float), r->nb[2]/sizeof(float), sinks ? sinks->nb[1]/sizeof(float) : 0,
            mask ? mask->nb[1]/sizeof(T_mask) : 0, mask ? mask->nb[3]/sizeof(T_mask) : 0, ggml_get_op_params_f32(dst, 0));
}

template <int n_rel>
static void rel_pos_bias_switch(const ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * mask = dst->src[4];

    const bool mask_f16 = mask && mask->type == GGML_TYPE_F16;
    if (dst->type == GGML_TYPE_F16) {
        mask_f16 ? rel_pos_bias_launch<n_rel, half, half>(dst, stream) : rel_pos_bias_launch<n_rel, float, half>(dst, stream);
    } else {
        mask_f16 ? rel_pos_bias_launch<n_rel, half, float>(dst, stream) : rel_pos_bias_launch<n_rel, float, float>(dst, stream);
    }
}

void ggml_cuda_op_rel_pos_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    switch (dst->src[0]->ne[0]) {
        case 8:  rel_pos_bias_switch<8>(dst, stream);  break;
        case 16: rel_pos_bias_switch<16>(dst, stream); break;
        case 32: rel_pos_bias_switch<32>(dst, stream); break;
        default: GGML_ABORT("unsupported n_rel");
    }
}
