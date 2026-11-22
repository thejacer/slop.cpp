#include "models.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_LOOP_START,     core_start);
    ml.get_key(LLM_KV_LOOP_END,       core_end);
    ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }
}

void llama_model_inkling2_loop::post_arch_tensors(llama_model_loader &) {
    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

// the gain and forget rows of a controller stacked and transposed, [2*n_embd, rank]: ggml_loop_ctrl reads them coalesced
static std::vector<uint8_t> ctrl_gain_forget_t(const ggml_tensor * gain, const ggml_tensor * forget) {
    const int64_t rank = gain->ne[0];
    const int64_t n    = gain->ne[1];

    std::vector<float> src(2*n*rank);
    ggml_backend_tensor_get(gain,   src.data(),          0, ggml_nbytes(gain));
    ggml_backend_tensor_get(forget, src.data() + n*rank, 0, ggml_nbytes(forget));

    std::vector<uint8_t> data(src.size()*sizeof(float));
    float * dst = (float *) data.data();
    for (int64_t r = 0; r < 2*n; ++r) {
        for (int64_t k = 0; k < rank; ++k) {
            dst[k*2*n + r] = src[r*rank + k];
        }
    }
    return data;
}

void llama_model_inkling2_loop::post_load_tensors() {
    make_derived();

    // the gain and forget projections read the same gate, one op does both with the tanh and the mix
    for (size_t i = 0; i < loop_layers.size(); ++i) {
        auto & l = loop_layers[i];
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            const ggml_tensor * gain   = l.ctrl_gain[c];
            const ggml_tensor * forget = l.ctrl_forget[c];
            if (gain->type != GGML_TYPE_F32 || forget->type != GGML_TYPE_F32 || !ggml_are_same_shape(gain, forget) ||
                    !ggml_is_contiguous(gain) || !ggml_is_contiguous(forget)) {
                continue;
            }
            l.ctrl_gain_forget[c] = new_derived(gain, GGML_TYPE_F32, 2*gain->ne[1], gain->ne[0],
                    format("inkling2.ctrl_gain_forget.%zu.%d", core_start + i, c), ctrl_gain_forget_t(gain, forget));
        }

        // the q, k and v controllers run at the same time, one batched matmul each for the down and the gain/forget projections
        bool can_batch = true;
        for (int c = LOOP_CTRL_Q; c <= LOOP_CTRL_V; ++c) {
            can_batch = can_batch && l.ctrl_gain_forget[c] && ggml_is_contiguous(l.ctrl_down[c]) &&
                l.ctrl_down[c]->type == l.ctrl_down[LOOP_CTRL_Q]->type && ggml_are_same_shape(l.ctrl_down[c], l.ctrl_down[LOOP_CTRL_Q]) &&
                l.ctrl_gain[c]->type == l.ctrl_gain[LOOP_CTRL_Q]->type && ggml_are_same_shape(l.ctrl_gain[c], l.ctrl_gain[LOOP_CTRL_Q]);
        }
        if (can_batch) {
            const ggml_tensor * down = l.ctrl_down[LOOP_CTRL_Q];
            const ggml_tensor * gain = l.ctrl_gain[LOOP_CTRL_Q];

            std::vector<uint8_t> down_data;
            std::vector<uint8_t> gf_data;
            for (int c = LOOP_CTRL_Q; c <= LOOP_CTRL_V; ++c) {
                std::vector<uint8_t> d(ggml_nbytes(l.ctrl_down[c]));
                ggml_backend_tensor_get(l.ctrl_down[c], d.data(), 0, d.size());
                down_data.insert(down_data.end(), d.begin(), d.end());

                // the gain/forget weights are rebuilt from the loaded tensors, the derived ones are not filled yet
                std::vector<uint8_t> g = ctrl_gain_forget_t(l.ctrl_gain[c], l.ctrl_forget[c]);
                gf_data.insert(gf_data.end(), g.begin(), g.end());
            }
            l.ctrl_down_qkv = new_derived(down, down->type, down->ne[0], down->ne[1],
                    format("inkling2.ctrl_down_qkv.%zu", core_start + i), std::move(down_data), 3);
            l.ctrl_gain_forget_qkv = new_derived(gain, GGML_TYPE_F32, 2*gain->ne[1], gain->ne[0],
                    format("inkling2.ctrl_gain_forget_qkv.%zu", core_start + i), std::move(gf_data), 3);
        }
    }

    alloc_derived();
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llama_model_inkling2::graph(params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;
    int visit = 0;
    for (int il = 0; il < start; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            res->t_layer_inp[visit++] = cur;
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop.msg_query), &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks);
}
