# Prana

**A feasibility study + working prototype for a Rust reimplementation of the
[Cactus](https://github.com/cactus-compute/cactus) on-device AI inference
engine.**

Prana is not (yet) a production engine. It is the answer to a specific
question — *"is it worth rewriting Cactus's inference engine in Rust?"* —
backed by a compiling, tested Rust workspace that demonstrates the proposed
architecture rather than just asserting it — up to and including **running a
real trained LLM end-to-end**: it loads Karpathy's 15M-parameter TinyStories
Llama from either **llama2.c or GGUF** containers and generates coherent
stories at up to **~1270 tok/s** (Q8 + AVX2/FMA SIMD + persistent worker
pool; ~520 tok/s f32) — 2.1× the throughput of llama2.c's best OpenMP build
on the same machine — with greedy output token-identical to llama2.c's
reference implementation.

```text
$ prana run
Once upon a time, in a small garden, there lived a little car named Red.
Red loved to play with his friends, but he had one problem. He had hurt
his wheel and could not move anymore. ...
  5 prompt + 162 generated tokens in 1.09s  ->  152.8 tok/s
```

## TL;DR

- A **full big-bang rewrite is not worth it**: Cactus's speed ceiling is set by
  hand-tuned ARM NEON + Apple Metal, which Rust can match but not beat, and a
  from-scratch rewrite would shed years of accumulated correctness.
- A **targeted, incremental Rust reimplementation of the orchestration
  layers** (engine, computation graph, tooling) over the existing C ABI **is
  worth it**: ~52k lines of lifetime-, ownership-, and concurrency-heavy code
  move to a domain where the compiler removes whole bug classes at ~zero
  runtime cost.

The full analysis — layer by layer, with a phased strangler-fig migration plan —
is in **[EVALUATION.md](./EVALUATION.md)**.

## The prototype

A Cargo workspace mirroring Cactus's layering, with everything above the kernel
boundary marked `#![forbid(unsafe_code)]`:

| Crate | Mirrors | What it shows |
|-------|---------|---------------|
| [`prana-tensor`](crates/prana-tensor) | `BufferDesc` / `BufferPool` | dtype-erased tensors + size-bucketed buffer pool, zero `unsafe` |
| [`prana-kernels`](crates/prana-kernels) | `matmul` / `quants` / `norms_rope` / `attention` / `threading` | Q8 quantized matmul, dense matmul, RMSNorm, softmax, RoPE, causal grouped-query attention, scoped-thread `parallel_for` |
| [`prana-graph`](crates/prana-graph) | `CactusGraph` | checked-handle, shape-validating, define-then-run graph |
| [`prana-cactus`](crates/prana-cactus) | `bindings/rust/cactus.rs` + `cactus_engine.h` | **Phase 0**: safe RAII wrapper over the real Cactus C ABI — `Result` errors, streaming-callback trampoline with panic containment, grow-and-retry buffers; testable everywhere via an in-process mock of the ABI |
| [`prana-model`](crates/prana-model) | `cactus-engine` model loading / tokenizer / sampling | **real-model inference**: llama2.c *and GGUF* loaders for **llama, qwen2, and gemma** architectures (F32/F16/Q8_0/Q4_K/Q6_K, all quantized types running natively in packed form), SentencePiece *and* GPT-2 byte-level BPE tokenizers (file or GGUF-embedded), KV-cached forward pass with QKV biases / NeoX RoPE / GELU / decoupled head_dim, greedy/temperature sampling — zero `unsafe`, zero deps |
| [`prana-cli`](crates/prana-cli) | `cactus run` / `benchmark` / `chat` | real-model text generation (`run`), transformer-block demo, microbenchmark, Phase 0 chat driver |

### Run it

```bash
cargo test                                   # 53 tests
./scripts/fetch-model.sh                     # fetch stories15M + tokenizer into models/
cargo run --release -p prana-cli -- run      # generate a story with the real model
cargo run --release -p prana-cli -- run --q8 # same, with Q8-quantized weights (fastest)
cargo run --release -p prana-cli -- demo     # run one full transformer block (GQA, seq=8)
cargo run --release -p prana-cli -- bench    # microbenchmark the Q8 matmul
cargo run --release -p prana-cli -- chat     # drive the Phase 0 safe wrapper (mock engine)

# GGUF: convert the checkpoint, then run it (tokenizer is embedded in the file)
python3 scripts/convert-to-gguf.py models/stories15M.bin models/tokenizer.bin models/stories15M.gguf
cargo run --release -p prana-cli -- run models/stories15M.gguf "Once upon a time" --q8

# K-quants: build a Q4_K/Q6_K mixture file (Q4_K_M-style) and run it
python3 scripts/convert-to-gguf.py models/stories15M.bin models/tokenizer.bin models/stories15M-q4km.gguf q4km
cargo run --release -p prana-cli -- run models/stories15M-q4km.gguf "Once upon a time"
```

`prana run` options: `[model.bin tokenizer.bin | model.gguf] [prompt]
[--steps N] [--temp T] [--seed S] [--q8]`. With `--temp 0` (greedy) the output
is token-identical to llama2.c's reference output for the same checkpoint —
via both container formats — which is the correctness check for the whole
pipeline (tokenizer, RoPE convention, attention, SwiGLU, SIMD kernels). The
AVX2+FMA SIMD tier engages automatically when the CPU supports it.

To bind `prana-cactus` against the real engine instead of the mock, build
`libcactus_engine.a` on an ARM/Apple host (`cactus-engine/build.sh` in the
Cactus repo) and use:

```bash
CACTUS_LIB_DIR=/path/to/lib cargo build -p prana-cactus --features link-cactus
```

Example:

```
$ cargo run --release -p prana-cli -- bench
  problem           : [1 x 2048] * [2048 x 2048]  (block=32)
  dense  f32 matmul :   0.664 ms/iter     12.6 GFLOP/s
  quant  q8  matmul :   0.949 ms/iter      8.8 GFLOP/s
  weight memory     : 16384 KiB f32  ->  4608 KiB q8  (3.56x smaller)
  max abs error     : 0.8181  (0.08% of output RMS)
```

The benchmark is deliberately honest: with the scalar kernel tier, Q8's win at
this cache-resident size is the **3.5× smaller weight footprint**, not speed —
the raw-speed win from quantization appears at cache-spilling model sizes and
once the target-gated NEON/AVX kernel tier lands. See EVALUATION.md §5.

## Status & scope

This is a prototype to support an architectural decision — but it now runs a
**real trained model end-to-end**: `prana-model` loads llama2.c checkpoints
and **GGUF v2/v3** files (F32/F16/Q8_0 tensors; GGUF's embedded SentencePiece
tokenizer), and generates via a KV-cached forward pass built entirely from
`prana-kernels` — with an **AVX2/FMA SIMD tier** selected at runtime and
parity-tested against the safe scalar tier. Greedy decoding reproduces
llama2.c's reference output token-for-token through both container formats.
It also implements **Phase 0 of the migration plan**: `prana-cactus`, a safe
idiomatic wrapper over the engine's real C ABI, exercised against an
in-process mock of that ABI (the native static lib is ARM/Metal-only).
**Architecture support:** llama-family plus **Qwen2** (QKV biases, NeoX RoPE,
GPT-2 BPE tokenizer from GGUF merges) and **Gemma** ((1+w) RMSNorm folded at
load, √dim embedding scale, tanh-GELU MLP, decoupled head_dim). Every model
host reachable from this environment is egress-blocked, so Qwen2/Gemma are
verified with synthetic GGUF fixtures (loader knobs, bias effects, norm
folding, finite deterministic forward passes) rather than real weights — the
llama path is the one verified token-identical against a reference
implementation. K-quant tensors (Q4_K/Q6_K) run **natively in packed form**
(4.5/6.6 bits per weight in RAM) with the AVX2 tier; native dots are
parity-tested against dequantized dense matmuls.

Still out of scope: ARM NEON/Metal targets, safetensors, further
architectures (Phi, Gemma-2 softcapping), exact GPT-2 pre-tokenizer regex,
and the CQ rotation-codebook quantization; later phases, not built here.

## License

Apache-2.0. Cactus is a separate project under its own license; Prana is an
independent evaluation and does not vendor Cactus source.
