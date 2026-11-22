#include "depth-mix.cuh"

#define DEPTH_MIX_BLOCK_SIZE 256

// one block per token: norm cur, score all states against all queries, softmax over the states, mix
template <int n_query>
static __global__ void depth_mix_f32(
        const float * __restrict__ cur, const float * __restrict__ query, const float * __restrict__ states, float * __restrict__ dst,
        const int n_embd, const int n_prev, const int64_t s_cur, const int64_t s_query, const int64_t s_states1, const int64_t s_states2,
        const int64_t s_dst1, const int64_t s_dst2, const float scale, const float eps) {
    constexpr int n_warps = DEPTH_MIX_BLOCK_SIZE/WARP_SIZE;

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    __shared__ float s_red[n_warps][DEPTH_MIX_MAX_STATES*DEPTH_MIX_MAX_QUERY];
    __shared__ float s_p[DEPTH_MIX_MAX_STATES*DEPTH_MIX_MAX_QUERY];
    __shared__ float s_inv;

    const float * c  = cur + t*s_cur;
    const float * st = states ? states + t*s_states2 : nullptr;

    // rms of cur
    float sum = 0.0f;
    for (int e = tid; e < n_embd; e += DEPTH_MIX_BLOCK_SIZE) {
        sum += c[e]*c[e];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        s_red[warp][0] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; ++w) {
            total += s_red[w][0];
        }
        s_inv = rsqrtf(total/n_embd + eps);
    }
    __syncthreads();
    const float inv = s_inv;

    // the dot products of the states with the queries
    float acc[DEPTH_MIX_MAX_STATES][n_query];
#pragma unroll
    for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
            acc[j][q] = 0.0f;
        }
    }
    for (int e = tid; e < n_embd; e += DEPTH_MIX_BLOCK_SIZE) {
        float qv[n_query];
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
            qv[q] = query[q*s_query + e];
        }
        const float cn = c[e]*inv;
#pragma unroll
        for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
            if (j <= n_prev) {
                const float s = j < n_prev ? st[j*s_states1 + e] : cn;
#pragma unroll
                for (int q = 0; q < n_query; ++q) {
                    acc[j][q] += s*qv[q];
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
            const float v = warp_reduce_sum(acc[j][q]);
            if (lane == 0) {
                s_red[warp][j*DEPTH_MIX_MAX_QUERY + q] = v;
            }
        }
    }
    __syncthreads();

    // softmax over the states, one thread per query
    if (tid < n_query) {
        const int q = tid;
        float max = -INFINITY;
        for (int j = 0; j <= n_prev; ++j) {
            float v = 0.0f;
            for (int w = 0; w < n_warps; ++w) {
                v += s_red[w][j*DEPTH_MIX_MAX_QUERY + q];
            }
            v *= scale;
            s_p[j*DEPTH_MIX_MAX_QUERY + q] = v;
            max = fmaxf(max, v);
        }
        float total = 0.0f;
        for (int j = 0; j <= n_prev; ++j) {
            const float v = expf(s_p[j*DEPTH_MIX_MAX_QUERY + q] - max);
            s_p[j*DEPTH_MIX_MAX_QUERY + q] = v;
            total += v;
        }
        for (int j = 0; j <= n_prev; ++j) {
            s_p[j*DEPTH_MIX_MAX_QUERY + q] /= total;
        }
    }
    __syncthreads();

    // the mixed states
    for (int e = tid; e < n_embd; e += DEPTH_MIX_BLOCK_SIZE) {
        float out[n_query];
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
            out[q] = 0.0f;
        }
        const float cn = c[e]*inv;
#pragma unroll
        for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
            if (j <= n_prev) {
                const float s = j < n_prev ? st[j*s_states1 + e] : cn;
#pragma unroll
                for (int q = 0; q < n_query; ++q) {
                    out[q] += s_p[j*DEPTH_MIX_MAX_QUERY + q]*s;
                }
            }
        }
#pragma unroll
        for (int q = 0; q < n_query; ++q) {
            dst[t*s_dst2 + q*s_dst1 + e] = out[q];
        }
    }
}

// one float4 of each state per thread, read once: a block of n_embd/4 threads per token
// dot(rms_norm(c), q) = inv*dot(c, q), so the sum of squares and all the dot products are reduced in one round
template <int n_query>
static __global__ void depth_mix_f32_vec(
        const float4 * __restrict__ cur, const float4 * __restrict__ query, const float4 * __restrict__ states, float4 * __restrict__ dst,
        const int n_embd4, const int n_embd, const int n_prev, const int64_t s_cur, const int64_t s_query, const int64_t s_states1, const int64_t s_states2,
        const int64_t s_dst1, const int64_t s_dst2, const float scale, const float eps) {
    constexpr int max_warps = 1024/WARP_SIZE;
    // the dots of the states and of cur with each query, then the sum of squares
    constexpr int n_vals = (DEPTH_MIX_MAX_STATES + 1)*n_query + 1;
    constexpr int i_sq   = n_vals - 1;

    const int t       = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp    = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    const bool active = tid < n_embd4;

    __shared__ float s_red[max_warps][n_vals];
    __shared__ float s_tot[n_vals];
    __shared__ float s_p[(DEPTH_MIX_MAX_STATES + 1)*n_query];

    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 c = active ? cur[t*s_cur + tid] : zero;

    float4 st[DEPTH_MIX_MAX_STATES];
#pragma unroll
    for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
        st[j] = active && j < n_prev ? states[t*s_states2 + j*s_states1 + tid] : zero;
    }
    float4 qv[n_query];
#pragma unroll
    for (int q = 0; q < n_query; ++q) {
        qv[q] = active ? query[q*s_query + tid] : zero;
    }

    auto dot4 = [](const float4 a, const float4 b) {
        return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    };

    // state j = n_prev is cur
#pragma unroll
    for (int j = 0; j <= DEPTH_MIX_MAX_STATES; ++j) {
        if (j <= n_prev) {
            const float4 sj = j < n_prev ? st[j < DEPTH_MIX_MAX_STATES ? j : 0] : c;
#pragma unroll
            for (int q = 0; q < n_query; ++q) {
                const float v = warp_reduce_sum(dot4(sj, qv[q]));
                if (lane == 0) {
                    s_red[warp][j*n_query + q] = v;
                }
            }
        }
    }
    {
        const float v = warp_reduce_sum(dot4(c, c));
        if (lane == 0) {
            s_red[warp][i_sq] = v;
        }
    }
    __syncthreads();

    // the sums over the warps, one thread per value
    for (int v = tid; v < n_vals; v += blockDim.x) {
        const bool used = v == i_sq || v < (n_prev + 1)*n_query;
        float total = 0.0f;
        if (used) {
            for (int w = 0; w < n_warps; ++w) {
                total += s_red[w][v];
            }
        }
        s_tot[v] = total;
    }
    __syncthreads();

    const float inv = rsqrtf(s_tot[i_sq]/n_embd + eps);

    // softmax over the states, one thread per query
    if (tid < n_query) {
        const int q = tid;
        float max = -INFINITY;
        for (int j = 0; j <= n_prev; ++j) {
            const float v = scale*s_tot[j*n_query + q]*(j == n_prev ? inv : 1.0f);
            s_p[j*n_query + q] = v;
            max = fmaxf(max, v);
        }
        float total = 0.0f;
        for (int j = 0; j <= n_prev; ++j) {
            const float v = expf(s_p[j*n_query + q] - max);
            s_p[j*n_query + q] = v;
            total += v;
        }
        const float r = 1.0f/total;
        for (int j = 0; j <= n_prev; ++j) {
            s_p[j*n_query + q] *= r;
        }
    }
    __syncthreads();

    if (!active) {
        return;
    }
#pragma unroll
    for (int q = 0; q < n_query; ++q) {
        const float pc = s_p[n_prev*n_query + q]*inv;
        float4 out = make_float4(pc*c.x, pc*c.y, pc*c.z, pc*c.w);
#pragma unroll
        for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
            if (j < n_prev) {
                const float p = s_p[j*n_query + q];
                out.x += p*st[j].x;
                out.y += p*st[j].y;
                out.z += p*st[j].z;
                out.w += p*st[j].w;
            }
        }
        dst[t*s_dst2 + q*s_dst1 + tid] = out;
    }
}

bool ggml_cuda_depth_mix_supported(const ggml_tensor * dst) {
    const ggml_tensor * query  = dst->src[1];
    const ggml_tensor * states = dst->src[2];
    const int64_t n_prev = states ? states->ne[1] : 0;
    return n_prev + 1 <= DEPTH_MIX_MAX_STATES && query->ne[1] >= 1 && query->ne[1] <= DEPTH_MIX_MAX_QUERY && dst->src[0]->ne[0] <= INT32_MAX;
}

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur    = dst->src[0];
    const ggml_tensor * query  = dst->src[1];
    const ggml_tensor * states = dst->src[2];

    GGML_ASSERT(ggml_cuda_depth_mix_supported(dst));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float scale = ggml_get_op_params_f32(dst, 0);
    const float eps   = ggml_get_op_params_f32(dst, 1);

    const int     n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int     n_query  = query->ne[1];
    const int     n_prev   = states ? states->ne[1] : 0;

    const float * cur_d    = (const float *) cur->data;
    const float * query_d  = (const float *) query->data;
    const float * states_d = states ? (const float *) states->data : nullptr;
    float       * dst_d    = (float *) dst->data;

    const int64_t s_cur     = cur->nb[1]/sizeof(float);
    const int64_t s_query   = query->nb[1]/sizeof(float);
    const int64_t s_states1 = states ? states->nb[1]/sizeof(float) : 0;
    const int64_t s_states2 = states ? states->nb[2]/sizeof(float) : 0;
    // the kernels take the query stride as s_dst1 and the token stride as s_dst2
    const int64_t s_dst1    = dst->nb[2]/sizeof(float);
    const int64_t s_dst2    = dst->nb[1]/sizeof(float);

    cudaStream_t stream = ctx.stream();

    // the register kernel takes all rows as float4 in one block
    const int n_embd4 = n_embd/4;
    const bool vec = n_embd % 4 == 0 && n_embd4 <= 1024 && s_cur % 4 == 0 && s_query % 4 == 0 && s_states1 % 4 == 0 &&
        s_states2 % 4 == 0 && s_dst1 % 4 == 0 && s_dst2 % 4 == 0 &&
        (uintptr_t) cur_d % 16 == 0 && (uintptr_t) query_d % 16 == 0 && (uintptr_t) states_d % 16 == 0 && (uintptr_t) dst_d % 16 == 0;
    if (vec) {
        const int block_size = (n_embd4 + WARP_SIZE - 1)/WARP_SIZE*WARP_SIZE;

#define DEPTH_MIX_VEC_LAUNCH(NQ) depth_mix_f32_vec<NQ><<<n_tokens, block_size, 0, stream>>>( \
        (const float4 *) cur_d, (const float4 *) query_d, (const float4 *) states_d, (float4 *) dst_d, \
        n_embd4, n_embd, n_prev, s_cur/4, s_query/4, s_states1/4, s_states2/4, s_dst1/4, s_dst2/4, scale, eps)

        switch (n_query) {
            case 1: DEPTH_MIX_VEC_LAUNCH(1); break;
            case 2: DEPTH_MIX_VEC_LAUNCH(2); break;
            case 3: DEPTH_MIX_VEC_LAUNCH(3); break;
            case 4: DEPTH_MIX_VEC_LAUNCH(4); break;
            default: GGML_ABORT("fatal error");
        }

#undef DEPTH_MIX_VEC_LAUNCH
        return;
    }

    const dim3 grid(n_tokens);
    const dim3 block(DEPTH_MIX_BLOCK_SIZE);

#define DEPTH_MIX_LAUNCH(NQ) depth_mix_f32<NQ><<<grid, block, 0, stream>>>(cur_d, query_d, states_d, dst_d, \
        n_embd, n_prev, s_cur, s_query, s_states1, s_states2, s_dst1, s_dst2, scale, eps)

    switch (n_query) {
        case 1: DEPTH_MIX_LAUNCH(1); break;
        case 2: DEPTH_MIX_LAUNCH(2); break;
        case 3: DEPTH_MIX_LAUNCH(3); break;
        case 4: DEPTH_MIX_LAUNCH(4); break;
        default: GGML_ABORT("fatal error");
    }

#undef DEPTH_MIX_LAUNCH
}
