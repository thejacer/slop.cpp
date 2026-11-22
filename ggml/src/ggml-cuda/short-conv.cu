#include "short-conv.cuh"

#define SHORT_CONV_BLOCK_SIZE 256
#define SHORT_CONV_TOKENS     32

// one thread per channel, sequence and chunk of tokens, with a sliding window of the last K inputs in registers
template <int K, bool has_dyn>
static __global__ void short_conv_f32(
        const float * __restrict__ x, const float * __restrict__ past, const float * __restrict__ weight,
        const float * __restrict__ dyn, float * __restrict__ state, const float * __restrict__ res, float * __restrict__ dst,
        const int n_ch, const int n_cg, const int n_seq_t, const int ch_head,
        const int64_t s_x1, const int64_t s_x2, const int64_t s_past1, const int64_t s_past2,
        const int64_t s_dyn1, const int64_t s_state1, const int64_t s_state2, const int n_slot,
        const int64_t s_res1, const int64_t s_dst1, const float dyn_scale) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_ch) {
        return;
    }
    const int s  = blockIdx.z;
    const int j0 = blockIdx.y*SHORT_CONV_TOKENS;
    const int j1 = min(j0 + SHORT_CONV_TOKENS, n_seq_t);

    const float * xc = x + (c % n_cg) + (c / n_cg)*s_x1 + (int64_t) s*n_seq_t*s_x2;
    const float * pc = past + c*s_past1 + s*s_past2;

    // column i of [past, x]
    auto xx = [&](int i) -> float {
        return i < K - 1 ? pc[i] : xc[(int64_t) (i - (K - 1))*s_x2];
    };

    float w[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        w[k] = weight[c*K + k];
    }

    float win[K];
#pragma unroll
    for (int k = 0; k < K - 1; ++k) {
        win[k] = xx(j0 + k);
    }

    const float * dc = has_dyn ? dyn + (c / ch_head)*K : nullptr;

    for (int j = j0; j < j1; ++j) {
        win[K - 1] = xc[(int64_t) j*s_x2];
        const int64_t t = (int64_t) s*n_seq_t + j;

        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            sum += w[k]*win[k];
        }
        if (has_dyn) {
            const float * d = dc + t*s_dyn1;
            float sd = 0.0f;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                sd += d[k]*win[K - 1 - k];
            }
            sum += dyn_scale*sd;
        }
        if (res) {
            sum += res[c + t*s_res1];
        }
        dst[c + t*s_dst1] = sum;

#pragma unroll
        for (int k = 0; k < K - 1; ++k) {
            win[k] = win[k + 1];
        }

        // rollback slot n_seq_t - 1 - j holds the window after token j (slot 0 is the final state)
        const int slot = n_seq_t - 1 - j;
        if (state && slot < n_slot) {
            float * out = state + s*s_state1 + slot*s_state2 + (int64_t) c*(K - 1);
#pragma unroll
            for (int k = 0; k < K - 1; ++k) {
                out[k] = win[k];
            }
        }
    }
}

bool ggml_cuda_short_conv_supported(const ggml_tensor * dst) {
    const int64_t K = dst->src[2]->ne[0];
    return K >= 2 && K <= 4 && ggml_nelements(dst) < INT_MAX && ggml_nelements(dst->src[0]) < INT_MAX;
}

template <int K>
static void short_conv_launch(const ggml_tensor * dst, float * state, cudaStream_t stream) {
    const ggml_tensor * x      = dst->src[0];
    const ggml_tensor * past   = dst->src[1];
    const ggml_tensor * weight = dst->src[2];
    const ggml_tensor * dyn    = dst->src[3];
    const ggml_tensor * st     = dst->src[4];
    const ggml_tensor * res    = dst->src[5];

    const float * r_d    = res ? (const float *) res->data : nullptr;
    const int64_t s_res1 = res ? res->nb[1]/sizeof(float) : 0;

    const int n_cg    = x->ne[0];
    const int n_ch    = n_cg*x->ne[1];
    const int n_seqs  = past->ne[2];
    const int n_seq_t = x->ne[2]/n_seqs;
    const int ch_head = dyn ? n_ch/(dyn->ne[0]/K) : 1;

    const float dyn_scale = ggml_get_op_params_f32(dst, 0);

    const dim3 grid((n_ch + SHORT_CONV_BLOCK_SIZE - 1)/SHORT_CONV_BLOCK_SIZE, (n_seq_t + SHORT_CONV_TOKENS - 1)/SHORT_CONV_TOKENS, n_seqs);

    const float * x_d = (const float *) x->data;
    const float * p_d = (const float *) past->data;
    const float * w_d = (const float *) weight->data;
    float       * d_d = (float *) dst->data;

    const int64_t s_x1     = x->nb[1]/sizeof(float);
    const int64_t s_x2     = x->nb[2]/sizeof(float);
    const int64_t s_past1  = past->nb[1]/sizeof(float);
    const int64_t s_past2  = past->nb[2]/sizeof(float);
    const int64_t s_state1 = st ? st->nb[1]/sizeof(float) : 0;
    const int64_t s_state2 = st ? st->nb[2]/sizeof(float) : 0;
    const int     n_slot   = st ? st->ne[2] : 0;
    const int64_t s_dst1   = dst->nb[1]/sizeof(float);

    if (dyn) {
        short_conv_f32<K, true><<<grid, SHORT_CONV_BLOCK_SIZE, 0, stream>>>(x_d, p_d, w_d, (const float *) dyn->data, state, r_d, d_d,
                n_ch, n_cg, n_seq_t, ch_head, s_x1, s_x2, s_past1, s_past2, dyn->nb[1]/sizeof(float), s_state1, s_state2, n_slot, s_res1, s_dst1, dyn_scale);
    } else {
        short_conv_f32<K, false><<<grid, SHORT_CONV_BLOCK_SIZE, 0, stream>>>(x_d, p_d, w_d, nullptr, state, r_d, d_d,
                n_ch, n_cg, n_seq_t, ch_head, s_x1, s_x2, s_past1, s_past2, 0, s_state1, s_state2, n_slot, s_res1, s_dst1, dyn_scale);
    }
}

void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * st = dst->src[4];

    float * state = st ? (float *) st->data : nullptr;
    cudaStream_t stream = ctx.stream();

    switch (dst->src[2]->ne[0]) {
        case 2: short_conv_launch<2>(dst, state, stream); break;
        case 3: short_conv_launch<3>(dst, state, stream); break;
        case 4: short_conv_launch<4>(dst, state, stream); break;
        default: GGML_ABORT("unsupported short conv size");
    }
}
