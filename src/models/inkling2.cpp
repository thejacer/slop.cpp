#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    // the sum of the selected sigmoid weights is never near 0, the F16 clamp of the default path matches the fused top-k kernels
    hparams.expert_weights_norm_eps = 0.0f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

llama_model_inkling2::~llama_model_inkling2() {
    for (ggml_backend_buffer_t buf : derived_bufs) {
        ggml_backend_buffer_free(buf);
    }
    for (ggml_context * ctx : derived_ctxs) {
        ggml_free(ctx);
    }
}

ggml_tensor * llama_model_inkling2::new_derived(const ggml_tensor * src, ggml_type type, int64_t ne0, int64_t ne1,
        const std::string & name, std::vector<uint8_t> && data, int64_t ne2) {
    if (src->buffer == nullptr || ggml_backend_buffer_is_host(src->buffer)) {
        return nullptr;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src->buffer);
    ggml_context *& ctx = derived_pending_ctx[buft];
    if (ctx == nullptr) {
        // at most a few derived tensors per layer
        ggml_init_params params = {
            /*.mem_size   =*/ 8*LLAMA_MAX_LAYERS*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("inkling2: failed to create the derived weight context");
        }
        derived_ctxs.push_back(ctx);
    }

    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_set_name(t, name.c_str());
    GGML_ASSERT(data.size() == ggml_nbytes(t));

    derived_pending.push_back({t, std::move(data)});
    return t;
}

void llama_model_inkling2::alloc_derived() {
    for (auto & [buft, ctx] : derived_pending_ctx) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            throw std::runtime_error(format("inkling2: unable to allocate the derived weights in %s", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        derived_bufs.push_back(buf);

        LLAMA_LOG_INFO("%s: %s derived weights = %.2f MiB\n", __func__, ggml_backend_buft_name(buft), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
    }
    for (auto & fill : derived_pending) {
        ggml_backend_tensor_set(fill.tensor, fill.data.data(), 0, fill.data.size());
    }
    derived_pending_ctx.clear();
    derived_pending.clear();
}

static std::vector<uint8_t> tensor_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());
    return data;
}

// scale row r of a weight by s[r]: the block scales for Q8_0, false for the other quantized types
static bool scale_rows(std::vector<uint8_t> & data, ggml_type type, int64_t ne0, const std::vector<float> & s) {
    const size_t row_size = ggml_row_size(type, ne0);
    GGML_ASSERT(data.size() == row_size*s.size());
    for (size_t r = 0; r < s.size(); ++r) {
        uint8_t * row = data.data() + r*row_size;
        switch (type) {
            case GGML_TYPE_F32:
                for (int64_t i = 0; i < ne0; ++i) {
                    ((float *) row)[i] *= s[r];
                }
                break;
            case GGML_TYPE_F16:
                for (int64_t i = 0; i < ne0; ++i) {
                    ((ggml_fp16_t *) row)[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(((ggml_fp16_t *) row)[i])*s[r]);
                }
                break;
            case GGML_TYPE_Q8_0:
                // block_q8_0: an F16 scale and 32 int8 values
                for (int64_t b = 0; b < ne0/32; ++b) {
                    ggml_fp16_t * d = (ggml_fp16_t *) (row + b*(sizeof(ggml_fp16_t) + 32));
                    *d = ggml_fp32_to_fp16(ggml_fp16_to_fp32(*d)*s[r]);
                }
                break;
            default:
                return false;
        }
    }
    return true;
}

void llama_model_inkling2::post_load_tensors() {
    make_derived();
    alloc_derived();
}

void llama_model_inkling2::make_derived() {
    const int     n_phys = n_layer_phys();
    const int64_t n_conv = hparams.n_shortconv_l_cache;
    const int64_t n_embd = hparams.n_embd;

    derived.assign(n_phys, {});

    // fold the residual and the output scale into the conv weights: (x + conv(w, x))*s = conv(s*(w + e_last), x)
    // only when all of them are F32 in device memory, the conv weights of the loaded file may be read-only host memory
    conv_folded = true;
    for (int il = 0; il < n_phys; ++il) {
        const auto & layer = layers[il];
        for (const ggml_tensor * w : {layer.attn_k_conv, layer.attn_v_conv, layer.attn_post_conv, layer.ffn_post_conv}) {
            conv_folded = conv_folded && w->type == GGML_TYPE_F32 && w->ne[0] == n_conv && w->buffer && !ggml_backend_buffer_is_host(w->buffer);
        }
    }
    if (conv_folded) {
        for (int il = 0; il < n_phys; ++il) {
            const auto & layer = layers[il];
            const bool dynamic = !hparams.is_swa(il);
            for (ggml_tensor * w : {layer.attn_k_conv, layer.attn_v_conv, layer.attn_post_conv, layer.ffn_post_conv}) {
                const bool  dyn = dynamic && (w == layer.attn_k_conv || w == layer.attn_v_conv);
                const float s   = 1.0f/sqrtf((dyn ? 2 : 1)*n_conv + 1);

                std::vector<float> data(ggml_nelements(w));
                ggml_backend_tensor_get(w, data.data(), 0, ggml_nbytes(w));
                for (int64_t c = 0; c < w->ne[1]; ++c) {
                    float * row = data.data() + c*n_conv;
                    row[n_conv - 1] += 1.0f;
                    for (int64_t k = 0; k < n_conv; ++k) {
                        row[k] *= s;
                    }
                }
                ggml_backend_tensor_set(w, data.data(), 0, ggml_nbytes(w));
            }
        }
    }

    for (int il = 0; il < n_phys; ++il) {
        const auto & layer = layers[il];
        auto & d = derived[il];

        // the bias table times relative_clip, with the zero row of the far distances
        if (layer.attn_rel_bias->type == GGML_TYPE_F32) {
            const int64_t n_rel    = layer.attn_rel_bias->ne[0];
            const int64_t n_extent = layer.attn_rel_bias->ne[1];

            std::vector<float> src(n_rel*n_extent);
            ggml_backend_tensor_get(layer.attn_rel_bias, src.data(), 0, ggml_nbytes(layer.attn_rel_bias));

            std::vector<uint8_t> data((n_rel*(n_extent + 1))*sizeof(float), 0);
            float * dst = (float *) data.data();
            for (int64_t i = 0; i < n_rel*n_extent; ++i) {
                dst[i] = src[i]*hparams.f_relative_clip;
            }
            d.rel_bias = new_derived(layer.attn_rel_bias, GGML_TYPE_F32, n_rel, n_extent + 1, format("inkling2.rel_bias.%d", il), std::move(data));
        }

        // SWA layers convolve K and V alike and keep them next to each other in the conv state: one conv for both
        const bool kv_stack = hparams.is_swa(il) && conv_folded && layer.wk->type == layer.wv->type && ggml_are_same_shape(layer.wk, layer.wv) &&
            ggml_is_contiguous(layer.wk) && ggml_is_contiguous(layer.wv) && layer.attn_k_conv->type == GGML_TYPE_F32 && layer.attn_v_conv->type == GGML_TYPE_F32;
        if (kv_stack) {
            std::vector<uint8_t> conv = tensor_bytes(layer.attn_k_conv);
            std::vector<uint8_t> bytes = tensor_bytes(layer.attn_v_conv);
            conv.insert(conv.end(), bytes.begin(), bytes.end());
            d.kv_conv = new_derived(layer.attn_k_conv, GGML_TYPE_F32, n_conv, 2*layer.attn_k_conv->ne[1], format("inkling2.kv_conv.%d", il), std::move(conv));
        }
        const bool looped = layer_looped(il);

        // one projection for Q, the output gate, the sink logits and R, which all read the Q input
        // SWA layers without messages read K and V from the same input: wk and wv join it
        const ggml_tensor * parts[] = {layer.wq, layer.wqkv_gate, layer.attn_sink_proj, layer.attn_rel};
        bool can_stack = true;
        int64_t ne1 = 0;
        for (const ggml_tensor * p : parts) {
            can_stack = can_stack && p->type == layer.wq->type && p->ne[0] == layer.wq->ne[0] && ggml_is_contiguous(p);
            ne1 += p->ne[1];
        }
        const bool with_kv = can_stack && d.kv_conv && layer.wk->type == layer.wq->type && layer.wk->ne[0] == layer.wq->ne[0];
        if (can_stack) {
            // the per-channel clip scales of Q and R are folded into their rows, as they come before the l2 norm
            const ggml_type t = layer.wq->type;
            const bool fold = (t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0) &&
                layer.attn_q_clip->type == GGML_TYPE_F32 && layer.attn_rel_clip->type == GGML_TYPE_F32 &&
                ggml_nelements(layer.attn_q_clip) == layer.wq->ne[1] && ggml_nelements(layer.attn_rel_clip) == layer.attn_rel->ne[1];
            std::vector<uint8_t> data;
            for (const ggml_tensor * p : parts) {
                std::vector<uint8_t> bytes = tensor_bytes(p);
                const ggml_tensor * clip = p == layer.wq ? layer.attn_q_clip : p == layer.attn_rel ? layer.attn_rel_clip : nullptr;
                if (fold && clip) {
                    std::vector<float> s(ggml_nelements(clip));
                    ggml_backend_tensor_get(clip, s.data(), 0, ggml_nbytes(clip));
                    GGML_ASSERT(scale_rows(bytes, p->type, p->ne[0], s));
                }
                data.insert(data.end(), bytes.begin(), bytes.end());
            }
            if (with_kv) {
                std::vector<uint8_t> data_kv = data;
                for (const ggml_tensor * p : {layer.wk, layer.wv}) {
                    std::vector<uint8_t> bytes = tensor_bytes(p);
                    data_kv.insert(data_kv.end(), bytes.begin(), bytes.end());
                }
                d.wqgsr_kv = new_derived(layer.wq, t, layer.wq->ne[0], ne1 + 2*layer.wk->ne[1], format("inkling2.wqgsr_kv.%d", il), std::move(data_kv));
            }
            // the layers with messages, and the ones without the stacked K and V
            if (!d.wqgsr_kv || looped || !hparams.is_swa(il)) {
                d.wqgsr = new_derived(layer.wq, t, layer.wq->ne[0], ne1, format("inkling2.wqgsr.%d", il), std::move(data));
            }
            d.clip_folded = fold && (d.wqgsr || d.wqgsr_kv);
        }

        // full attention: the three input norms and the K|dyn and V|dyn projections batched
        if (!hparams.is_swa(il)) {
            const ggml_tensor * norms[] = {layer.attn_q_inp_norm, layer.attn_k_inp_norm, layer.attn_v_inp_norm};
            bool can_norm = true;
            for (const ggml_tensor * w : norms) {
                can_norm = can_norm && w->type == GGML_TYPE_F32 && ggml_nelements(w) == n_embd;
            }
            if (can_norm) {
                std::vector<uint8_t> data;
                for (const ggml_tensor * w : norms) {
                    std::vector<uint8_t> bytes = tensor_bytes(w);
                    data.insert(data.end(), bytes.begin(), bytes.end());
                }
                d.qkv_norm = new_derived(layer.attn_q_inp_norm, GGML_TYPE_F32, n_embd, 1, format("inkling2.qkv_norm.%d", il), std::move(data), 3);
            }

            const ggml_tensor * parts[] = {layer.wk, layer.attn_k_conv_proj, layer.wv, layer.attn_v_conv_proj};
            bool can_batch = conv_folded && ggml_are_same_shape(layer.wk, layer.wv) && ggml_are_same_shape(layer.attn_k_conv_proj, layer.attn_v_conv_proj);
            for (const ggml_tensor * p : parts) {
                can_batch = can_batch && p->type == layer.wk->type && p->ne[0] == layer.wk->ne[0] && ggml_is_contiguous(p);
            }
            if (can_batch) {
                std::vector<uint8_t> data;
                for (const ggml_tensor * p : parts) {
                    std::vector<uint8_t> bytes = tensor_bytes(p);
                    data.insert(data.end(), bytes.begin(), bytes.end());
                }
                d.wkvc = new_derived(layer.wk, layer.wk->type, layer.wk->ne[0], layer.wk->ne[1] + layer.attn_k_conv_proj->ne[1],
                        format("inkling2.wkvc.%d", il), std::move(data), 2);
            }
        }

        // SWA layers with messages: K and V from one batched matmul
        if (looped && d.kv_conv) {
            std::vector<uint8_t> data = tensor_bytes(layer.wk);
            std::vector<uint8_t> bytes = tensor_bytes(layer.wv);
            data.insert(data.end(), bytes.begin(), bytes.end());
            d.wkv = new_derived(layer.wk, layer.wk->type, layer.wk->ne[0], 2*layer.wk->ne[1], format("inkling2.wkv.%d", il), std::move(data));
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// the row scaled down to an L2 norm of at most threshold, after the optional per-channel scale, divided by threshold
// l2_norm(x, eps) = x/max(|x|, eps), so the caller multiplies by threshold or folds it into a later scale
ggml_tensor * llama_model_inkling2::graph::clip_dir(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    return ggml_l2_norm(ctx0, cur, threshold);
}

llama_model_inkling2::graph::mix_states llama_model_inkling2::graph::get_mix_states(const std::vector<ggml_tensor *> & blocks) {
    if (blocks.empty()) {
        return {};
    }

    auto it = mix_cache.find(blocks);
    if (it != mix_cache.end()) {
        return it->second;
    }

    const mix_states prev = get_mix_states(std::vector<ggml_tensor *>(blocks.begin(), blocks.end() - 1));

    ggml_tensor * block = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, blocks.back(), n_embd, 1, n_tokens), 1e-6f);

    mix_states res;
    res.s = prev.s ? ggml_concat(ctx0, prev.s, block, 1) : block;

    return mix_cache[blocks] = res;
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// the normed blocks are built once per block list, the op norms cur and does the scores, softmax and mix in one kernel
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    const mix_states prev = get_mix_states(blocks);

    return ggml_depth_mix(ctx0, ggml_reshape_2d(ctx0, cur, n_embd, n_tokens), query, prev.s, 1.0f/sqrtf(n_embd), 1e-6f);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset,
        int64_t n_groups,
        ggml_tensor * res) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0]*n_groups;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    const float scale  = 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1);
    const bool  folded = static_cast<const llama_model_inkling2 &>(model).conv_folded;

    // one op reads the history and the inputs, writes the new state and does the conv with the dynamic taps
    // cur is [n_channels, n_tokens], or [n_channels/n_groups, n_groups, n_tokens] for matmul outputs side by side in the state
    if (folded) {
        ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);
        ggml_tensor * x    = n_groups > 1 ? cur : ggml_view_3d(ctx0, cur, n_channels, 1, n_tokens, cur->nb[1], cur->nb[1], 0);
        // with rollback (speculative decoding) slot k of each cell is the state after all but the last k tokens,
        // the slots are whole copies of the cells, see build_conv_state_store
        const int64_t n_slot = (int64_t) cparams.n_rs_seq + 1;
        ggml_tensor * cell = ggml_view_3d(ctx0, state, (n_conv - 1)*n_channels, n_seqs, n_slot, state->nb[1],
                (size_t) inp->mctx->get_size()*state->nb[1],
                inp->mctx->get_head()*state->nb[1] + offset*(n_conv - 1)*ggml_element_size(state));

        return ggml_short_conv(ctx0, x, past, weight, dynamic, cell, res, scale);
    }
    if (!ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx0, cur);
    }
    // grouped inputs are channel-major, so they flatten to [n_channels, n_tokens]
    cur = ggml_reshape_2d(ctx0, cur, n_channels, n_tokens);

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        if (folded) {
            dynamic = ggml_scale(ctx0, dynamic, scale);
        }
        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    // the folded weights hold the residual and the scale
    out = folded ? out : ggml_scale(ctx0, ggml_add(ctx0, cur, out), scale);
    return res ? ggml_add(ctx0, res, out) : out;
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
// cur_norm and message_norm are rms_norm(cur) and rms_norm(message), shared by the controllers of a layer
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * cur_norm, ggml_tensor * message, ggml_tensor * message_norm,
        const loop_layer & loop, int ctrl) {
    ggml_tensor * gate = ggml_concat(ctx0, cur_norm, message_norm, 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    // the transposed gain|forget projection, the tanh and the mix in one op
    if (loop.ctrl_gain_forget[ctrl]) {
        return ggml_reshape_2d(ctx0, ggml_loop_ctrl(ctx0, gate, loop.ctrl_gain_forget[ctrl], cur, message), n_embd, n_tokens);
    }

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    sinks_zero = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_head(0)), 0.0f);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the messages are normed once for all the controllers
    ggml_tensor * messages_norm = messages ? ggml_rms_norm(ctx0, messages, 1e-6f) : nullptr;

    // the message of controller ctrl, [n_embd, n_tokens]
    auto message = [&](ggml_tensor * src, int ctrl) {
        return ggml_view_2d(ctx0, src, n_embd, n_tokens, src->nb[1], ctrl*src->nb[2]);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    // the Q, K and V inputs as one [n_embd, n_tokens, 3], inputs[j] are its views: the full-attention mix or the batched controllers
    ggml_tensor * qkv_out = nullptr;
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query);
        if (is_swa) {
            inputs[0] = inputs[1] = inputs[2] = ggml_reshape_2d(ctx0, mixed, n_embd, n_tokens);
        } else {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], j*mixed->nb[2]);
            }
            qkv_out = mixed;
        }
    }

    if (messages && loop->ctrl_down_qkv && loop->ctrl_gain_forget_qkv) {
        // the Q, K and V controllers batched over a third dim: x, m [n_embd, n_tokens, 3]
        // the mixes are [n_embd, n_tokens, n_query], so the first 3 queries are a contiguous view
        auto qkv = [&](ggml_tensor * src) {
            return ggml_view_3d(ctx0, src, n_embd, n_tokens, 3, src->nb[1], src->nb[2], 0);
        };

        // the inputs are one tensor, or the 3 query views of the mixed [n_embd, n_tokens, 3]
        ggml_tensor * x;
        if (inputs[0] == inputs[1]) {
            x = ggml_repeat(ctx0, inputs[0], ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, n_tokens, 3));
        } else {
            GGML_ASSERT(inputs[0]->view_src && inputs[0]->view_src->ne[2] == 3);
            x = qkv(inputs[0]->view_src);
        }
        ggml_tensor * m   = qkv(messages);
        ggml_tensor * m_n = qkv(messages_norm);

        ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, x, 1e-6f), m_n, 0);
        gate = ggml_mul_mat(ctx0, loop->ctrl_down_qkv, gate);

        ggml_tensor * out = ggml_loop_ctrl(ctx0, gate, loop->ctrl_gain_forget_qkv, x, m);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(ctx0, out, n_embd, n_tokens, out->nb[1], j*out->nb[2]);
        }
        qkv_out = out;
    } else if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        ggml_tensor * shared_norm = is_swa ? ggml_rms_norm(ctx0, inputs[0], 1e-6f) : nullptr;
        ggml_tensor * outputs[3];
        for (int j = 0; j < 3; ++j) {
            ggml_tensor * cur_norm = shared_norm ? shared_norm : ggml_rms_norm(ctx0, inputs[j], 1e-6f);
            outputs[j] = control(inputs[j], cur_norm, message(messages, LOOP_CTRL_Q + j), message(messages_norm, LOOP_CTRL_Q + j),
                    *loop, LOOP_CTRL_Q + j);
        }
        for (int j = 0; j < 3; ++j) {
            inputs[j] = outputs[j];
        }
        qkv_out = nullptr;
    }

    // no derived weights in a model loaded without tensor data (the memory fit), the loaded ones are used
    static const derived_layer no_derived;
    const auto & derived = static_cast<const llama_model_inkling2 &>(model).derived;
    const auto & d = ip < (int) derived.size() ? derived[ip] : no_derived;

    // the K and V inputs side by side, [n_embd, n_tokens, 2], for the batched K|V matmuls
    ggml_tensor * kv_in = nullptr;
    if (!is_swa && qkv_out && d.qkv_norm) {
        // the three input norms as one, with their weights stacked [n_embd, 1, 3]
        qkv_out = ggml_mul(ctx0, ggml_rms_norm(ctx0, qkv_out, hparams.f_norm_rms_eps), d.qkv_norm);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(ctx0, qkv_out, n_embd, n_tokens, qkv_out->nb[1], j*qkv_out->nb[2]);
        }
        kv_in = ggml_view_3d(ctx0, qkv_out, n_embd, n_tokens, 2, qkv_out->nb[1], qkv_out->nb[2], qkv_out->nb[2]);
    } else if (is_swa && qkv_out) {
        qkv_out = build_norm(qkv_out, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(ctx0, qkv_out, n_embd, n_tokens, qkv_out->nb[1], j*qkv_out->nb[2]);
        }
        kv_in = ggml_view_3d(ctx0, qkv_out, n_embd, n_tokens, 2, qkv_out->nb[1], qkv_out->nb[2], qkv_out->nb[2]);
    } else if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        const int64_t n_rel = hparams.n_relative;

        // Q [n_embd_head, n_head, n_tokens], the gate [n_embd_head*n_head, n_tokens], the sinks [n_head, n_tokens] and R [n_rel, n_head, n_tokens]
        ggml_tensor * Qcur;
        ggml_tensor * gate;
        ggml_tensor * sinks;
        ggml_tensor * Rcur;
        // K and V [2*n_embd_k_gqa, n_tokens] from the stacked projection, when they read the Q input
        ggml_tensor * kv_proj = nullptr;
        const bool shared_input = inputs[0] == inputs[1] && inputs[0] == inputs[2];
        ggml_tensor * wproj = shared_input && d.wqgsr_kv ? d.wqgsr_kv : d.wqgsr;
        if (wproj) {
            ggml_tensor * qgsr = build_lora_mm(wproj, inputs[0]);
            const size_t es = ggml_element_size(qgsr);
            const int64_t n_q = n_embd_head*n_head;
            if (wproj == d.wqgsr_kv) {
                kv_proj = ggml_view_2d(ctx0, qgsr, 2*n_embd_k_gqa, n_tokens, qgsr->nb[1], (2*n_q + n_head + n_rel*n_head)*es);
            }

            Qcur  = ggml_view_3d(ctx0, qgsr, n_embd_head, n_head, n_tokens, n_embd_head*es, qgsr->nb[1], 0);
            gate  = ggml_view_2d(ctx0, qgsr, n_q, n_tokens, qgsr->nb[1], n_q*es);
            sinks = ggml_view_2d(ctx0, qgsr, n_head, n_tokens, qgsr->nb[1], 2*n_q*es);
            Rcur  = ggml_view_3d(ctx0, qgsr, n_rel, n_head, n_tokens, n_rel*es, qgsr->nb[1], (2*n_q + n_head)*es);
        } else {
            Qcur  = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, inputs[0]), n_embd_head, n_head, n_tokens);
            gate  = build_lora_mm(layer.wqkv_gate, inputs[0]);
            sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);
            Rcur  = ggml_reshape_3d(ctx0, build_lora_mm(layer.attn_rel, inputs[0]), n_rel, n_head, n_tokens);
        }

        // K and V [n_embd_head, n_head_kv, n_tokens]
        ggml_tensor * Kcur;
        ggml_tensor * Vcur;
        if (!is_swa && d.wkvc && kv_in) {
            // K|k_conv_proj and V|v_conv_proj in one batched matmul [n_embd_k_gqa + n_dyn, n_tokens, 2], then a conv each
            ggml_tensor * kvc = ggml_mul_mat(ctx0, d.wkvc, kv_in);
            const size_t  es    = ggml_element_size(kvc);
            const int64_t n_dyn = kvc->ne[0] - n_embd_k_gqa;
            ggml_tensor * out[2];
            ggml_tensor * conv[2] = {layer.attn_k_conv, layer.attn_v_conv};
            for (int g = 0; g < 2; ++g) {
                ggml_tensor * x   = ggml_view_2d(ctx0, kvc, n_embd_k_gqa, n_tokens, kvc->nb[1], g*kvc->nb[2]);
                ggml_tensor * dyn = ggml_view_2d(ctx0, kvc, n_dyn, n_tokens, kvc->nb[1], g*kvc->nb[2] + n_embd_k_gqa*es);
                out[g] = ggml_reshape_3d(ctx0, convolve(x, conv[g], dyn, state, history, recr, g*n_embd_k_gqa), n_embd_head, n_head_kv, n_tokens);
            }
            Kcur = out[0];
            Vcur = out[1];
        } else if (kv_proj || (d.wkv && kv_in)) {
            // SWA: K and V from the stacked projection or one batched matmul, and one conv, [2*n_embd_k_gqa, n_tokens]
            ggml_tensor * kv;
            int64_t n_groups = 1;
            if (kv_proj) {
                kv = kv_proj;
            } else {
                kv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, d.wkv, n_embd, n_embd_k_gqa, 2), kv_in);
                kv = ggml_permute(ctx0, kv, 0, 2, 1, 3);
                n_groups = 2;
            }
            kv = convolve(kv, d.kv_conv, nullptr, state, history, recr, 0, n_groups);

            const size_t es = ggml_element_size(kv);
            Kcur = ggml_view_3d(ctx0, kv, n_embd_head, n_head_kv, n_tokens, n_embd_head*es, kv->nb[1], 0);
            Vcur = ggml_view_3d(ctx0, kv, n_embd_head, n_head_kv, n_tokens, n_embd_head*es, kv->nb[1], n_embd_k_gqa*es);
        } else {
            Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                    is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
            Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                    is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        }

        // the clip thresholds of Q and K are folded into the KQ scale, the one of R into the derived bias table
        Qcur = clip_dir(Qcur, d.clip_folded ? nullptr : layer.attn_q_clip, hparams.f_qk_clip);
        Kcur = clip_dir(Kcur, nullptr, hparams.f_qk_clip);
        Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        // R is normed in ggml_rel_pos_bias, with the tau of the full-attention layers
        if (!d.clip_folded) {
            Rcur = ggml_mul(ctx0, Rcur, layer.attn_rel_clip);
        }

        const float kq_scale = hparams.f_qk_clip*hparams.f_qk_clip/sqrtf(n_embd_head);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        ggml_tensor * kq_mask  = (is_swa ? attn->self_attn_mask_swa : attn->self_attn_mask).raw;

        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        // one op dots r with the bias row of each distance (the last row, zero, for the far ones), subtracts the sinks
        // and adds the attention mask: the per-head mask [n_kv, n_tokens/n_stream, n_head, n_stream] that flash attention takes in F16
        ggml_tensor * rel_bias = d.rel_bias ? d.rel_bias : ggml_scale(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), hparams.f_relative_clip);
        ggml_tensor * r_scale  = is_swa ? nullptr : tau;

        // the mask of the heads [h0, h0 + n_h): a long context builds it per chunk of heads
        auto kq_b_fn = [&](int64_t h0, int64_t n_h) {
            ggml_tensor * r = ggml_view_3d(ctx0, Rcur, Rcur->ne[0], n_h, Rcur->ne[2], Rcur->nb[1], Rcur->nb[2], h0*Rcur->nb[1]);
            ggml_tensor * s = ggml_view_2d(ctx0, sinks, n_h, sinks->ne[1], sinks->nb[1], h0*sinks->nb[0]);
            ggml_tensor * kq_b = ggml_rel_pos_bias(ctx0, r, rel_bias, rel_pos, s, kq_mask, r_scale, hparams.f_relative_clip, kq_mask->ne[3],
                    cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);
            cb(kq_b, "kq_b", il);
            return kq_b;
        };

        GGML_ASSERT(sinks_zero->ne[0] == n_head);

        cur = build_attn(inp, Qcur, Kcur, Vcur, kq_b_fn, sinks_zero, kq_scale, il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
        const bool block_start = ip % hparams.attn_res_block_size == 0;
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa, 1,
                block_start ? nullptr : inpSA);
        cb(cur, "attn_out", il);

        if (block_start) {
            blocks.push_back(inpSA);
        }
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, ggml_rms_norm(ctx0, cur, 1e-6f), message(messages, LOOP_CTRL_FFN), message(messages_norm, LOOP_CTRL_FFN),
                *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);

    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd, 1, ffn_inp);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
