#include "loop-ctrl.cuh"

#define LOOP_CTRL_CHANNELS 64
#define LOOP_CTRL_SLICES   8
#define LOOP_CTRL_TOKENS   8

// a block of 64 channels and a chunk of tokens, the rank split over 8 slices and summed in shared memory
// the transposed weights are read coalesced once per chunk, the gate values are uniform over a slice
template <int n_tok>
static __global__ void __launch_bounds__(LOOP_CTRL_CHANNELS*LOOP_CTRL_SLICES) loop_ctrl_f32(
        const float * __restrict__ gate, const float * __restrict__ wt, const float * __restrict__ x, const float * __restrict__ m,
        float * __restrict__ dst, const int n, const int n_t, const int rank,
        const int64_t s_gate1, const int64_t s_gate2, const int64_t s_x1, const int64_t s_x2, const int64_t s_m1, const int64_t s_m2,
        const int64_t s_dst1, const int64_t s_dst2) {
    __shared__ float s_acc[LOOP_CTRL_SLICES - 1][n_tok][2][LOOP_CTRL_CHANNELS];

    const int c     = threadIdx.x;
    const int slice = threadIdx.y;
    const int i     = blockIdx.x*LOOP_CTRL_CHANNELS + c;
    const int t0    = blockIdx.y*n_tok;
    const int g     = blockIdx.z;
    const bool active = i < n;

    const float * w  = wt + (int64_t) g*2*n*rank;
    const float * gv = gate + t0*s_gate1 + g*s_gate2;

    float sg[n_tok] = {0.0f};
    float sf[n_tok] = {0.0f};
    if (active) {
        for (int k = slice; k < rank; k += LOOP_CTRL_SLICES) {
            const float wg = w[(int64_t) k*2*n + i];
            const float wf = w[(int64_t) k*2*n + n + i];
#pragma unroll
            for (int j = 0; j < n_tok; ++j) {
                if (t0 + j < n_t) {
                    const float gk = gv[j*s_gate1 + k];
                    sg[j] += wg*gk;
                    sf[j] += wf*gk;
                }
            }
        }
    }

    if (slice > 0) {
#pragma unroll
        for (int j = 0; j < n_tok; ++j) {
            s_acc[slice - 1][j][0][c] = sg[j];
            s_acc[slice - 1][j][1][c] = sf[j];
        }
    }
    __syncthreads();
    if (slice > 0 || !active) {
        return;
    }

#pragma unroll
    for (int j = 0; j < n_tok; ++j) {
        const int t = t0 + j;
        if (t >= n_t) {
            continue;
        }
        float a = sg[j];
        float b = sf[j];
#pragma unroll
        for (int s = 0; s < LOOP_CTRL_SLICES - 1; ++s) {
            a += s_acc[s][j][0][c];
            b += s_acc[s][j][1][c];
        }
        const float xv = x[i + t*s_x1 + g*s_x2];
        const float mv = m[i + t*s_m1 + g*s_m2];
        dst[i + t*s_dst1 + g*s_dst2] = xv - tanhf(b)*xv + tanhf(a)*mv;
    }
}

// many tokens: a block of 256 channels and 32 tokens, the gate of the tokens staged in shared memory
#define LOOP_CTRL_TILE_CHANNELS 256
#define LOOP_CTRL_TILE_TOKENS   32
#define LOOP_CTRL_MAX_RANK      128

static __global__ void __launch_bounds__(LOOP_CTRL_TILE_CHANNELS) loop_ctrl_f32_tile(
        const float * __restrict__ gate, const float * __restrict__ wt, const float * __restrict__ x, const float * __restrict__ m,
        float * __restrict__ dst, const int n, const int n_t, const int rank,
        const int64_t s_gate1, const int64_t s_gate2, const int64_t s_x1, const int64_t s_x2, const int64_t s_m1, const int64_t s_m2,
        const int64_t s_dst1, const int64_t s_dst2) {
    __shared__ float s_gate[LOOP_CTRL_MAX_RANK][LOOP_CTRL_TILE_TOKENS];

    const int i  = blockIdx.x*LOOP_CTRL_TILE_CHANNELS + threadIdx.x;
    const int t0 = blockIdx.y*LOOP_CTRL_TILE_TOKENS;
    const int g  = blockIdx.z;
    const int nt = min(LOOP_CTRL_TILE_TOKENS, n_t - t0);

    const float * gv = gate + t0*s_gate1 + g*s_gate2;
    for (int e = threadIdx.x; e < rank*LOOP_CTRL_TILE_TOKENS; e += blockDim.x) {
        const int k = e % rank;
        const int j = e / rank;
        s_gate[k][j] = j < nt ? gv[j*s_gate1 + k] : 0.0f;
    }
    __syncthreads();

    if (i >= n) {
        return;
    }

    const float * w = wt + (int64_t) g*2*n*rank;

    float sg[LOOP_CTRL_TILE_TOKENS] = {0.0f};
    float sf[LOOP_CTRL_TILE_TOKENS] = {0.0f};
    for (int k = 0; k < rank; ++k) {
        const float wg = w[(int64_t) k*2*n + i];
        const float wf = w[(int64_t) k*2*n + n + i];
#pragma unroll
        for (int j = 0; j < LOOP_CTRL_TILE_TOKENS; ++j) {
            sg[j] += wg*s_gate[k][j];
            sf[j] += wf*s_gate[k][j];
        }
    }

#pragma unroll
    for (int j = 0; j < LOOP_CTRL_TILE_TOKENS; ++j) {
        if (j < nt) {
            const int t = t0 + j;
            const float xv = x[i + t*s_x1 + g*s_x2];
            const float mv = m[i + t*s_m1 + g*s_m2];
            dst[i + t*s_dst1 + g*s_dst2] = xv - tanhf(sf[j])*xv + tanhf(sg[j])*mv;
        }
    }
}

bool ggml_cuda_loop_ctrl_supported(const ggml_tensor * dst) {
    return ggml_nelements(dst->src[1]) < INT_MAX && dst->src[0]->ne[0] <= LOOP_CTRL_MAX_RANK && dst->ne[1] <= 65535*LOOP_CTRL_TOKENS && dst->ne[2] <= 65535;
}

void ggml_cuda_op_loop_ctrl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * wt   = dst->src[1];
    const ggml_tensor * x    = dst->src[2];
    const ggml_tensor * m    = dst->src[3];

    const int n    = x->ne[0];
    const int n_t  = x->ne[1];
    const int rank = gate->ne[0];

    cudaStream_t stream = ctx.stream();

    const size_t fs = sizeof(float);
    // a single token needs no chunk
    const dim3 block(LOOP_CTRL_CHANNELS, LOOP_CTRL_SLICES, 1);
    if (n_t == 1) {
        const dim3 grid((n + LOOP_CTRL_CHANNELS - 1)/LOOP_CTRL_CHANNELS, 1, x->ne[2]);
        loop_ctrl_f32<1><<<grid, block, 0, stream>>>(
                (const float *) gate->data, (const float *) wt->data, (const float *) x->data, (const float *) m->data, (float *) dst->data,
                n, n_t, rank, gate->nb[1]/fs, gate->nb[2]/fs, x->nb[1]/fs, x->nb[2]/fs, m->nb[1]/fs, m->nb[2]/fs, dst->nb[1]/fs, dst->nb[2]/fs);
        return;
    }
    if (n_t > 2*LOOP_CTRL_TOKENS) {
        const dim3 grid((n + LOOP_CTRL_TILE_CHANNELS - 1)/LOOP_CTRL_TILE_CHANNELS, (n_t + LOOP_CTRL_TILE_TOKENS - 1)/LOOP_CTRL_TILE_TOKENS, x->ne[2]);
        loop_ctrl_f32_tile<<<grid, LOOP_CTRL_TILE_CHANNELS, 0, stream>>>(
                (const float *) gate->data, (const float *) wt->data, (const float *) x->data, (const float *) m->data, (float *) dst->data,
                n, n_t, rank, gate->nb[1]/fs, gate->nb[2]/fs, x->nb[1]/fs, x->nb[2]/fs, m->nb[1]/fs, m->nb[2]/fs, dst->nb[1]/fs, dst->nb[2]/fs);
        return;
    }
    const dim3 grid((n + LOOP_CTRL_CHANNELS - 1)/LOOP_CTRL_CHANNELS, (n_t + LOOP_CTRL_TOKENS - 1)/LOOP_CTRL_TOKENS, x->ne[2]);
    loop_ctrl_f32<LOOP_CTRL_TOKENS><<<grid, block, 0, stream>>>(
            (const float *) gate->data, (const float *) wt->data, (const float *) x->data, (const float *) m->data, (float *) dst->data,
            n, n_t, rank, gate->nb[1]/fs, gate->nb[2]/fs, x->nb[1]/fs, x->nb[2]/fs, m->nb[1]/fs, m->nb[2]/fs, dst->nb[1]/fs, dst->nb[2]/fs);
}
