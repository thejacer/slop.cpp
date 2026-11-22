# slop.cpp

A personal fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) tuned to run
**Inkling 2 Mini** (`inkling2_loop`, Q8_0) fast on **2x AMD MI50 (gfx906)** with the HIP backend.
It is not affiliated with or endorsed by the llama.cpp project. The upstream README is in
[README.llama.cpp.md](README.llama.cpp.md) and its license is kept in [LICENSE](LICENSE) (MIT).

## What is different

- Tensor parallelism (`-sm tensor`) for Inkling 2: the relative-position bias works with flash attention
  and a split model, and long contexts (256k) fit through head-chunked attention.
- Fused GPU ops for the Inkling 2 blocks: `SHORT_CONV`, `REL_POS_BIAS`, `LOOP_CTRL` and a vectorized `DEPTH_MIX`.
- Load-time weight folding and stacked projections.
- gfx906 tuning: Q8_0 matvec configs for 1-8 tokens and an async upload ring for the graph inputs.
- Experimental Inkling 2 DSpark speculative draft (exact; see below).

## Results (2x MI50, Q8_0, `-sm tensor`)

| | `-sm layer` | `-sm tensor` |
|---|---|---|
| prompt, 2048 tokens | ~860 t/s | ~990 t/s |
| decode, single stream | ~51 t/s | ~64 t/s |
| decode, 4 parallel streams | - | ~108 t/s aggregate |

The DSpark draft gives identical output but on this hardware only wins on code (up to ~79 t/s);
prose is slower than plain decoding, so it is off by default.

## Build (ROCm, gfx906)

```sh
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_RCCL=ON -DGGML_HIP_FORCE_MMQ=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench
```

## Run

```sh
GGML_CUDA_ALLREDUCE=internal ./build/bin/llama-server -m Inkling-2-Mini-Q8_0.gguf --mmproj mmproj-Inkling-2-Mini-f16.gguf \
  -ngl all -fa on -dev ROCm0,ROCm1 -sm tensor -b 2048 -ub 1024 -c 262144 -np 4 -kvu
```

Optional speculative draft (exact, best for code):

```sh
  -md dspark-Inkling-2-Mini-Q8_0.gguf --spec-type draft-dspark -devd ROCm1 -ngld all --spec-draft-n-max 2
```
