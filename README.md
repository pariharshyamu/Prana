# Prana

**A feasibility study + working prototype for a Rust reimplementation of the
[Cactus](https://github.com/cactus-compute/cactus) on-device AI inference
engine.**

Prana is not (yet) a production engine. It is the answer to a specific
question — *"is it worth rewriting Cactus's inference engine in Rust?"* —
backed by a compiling, tested Rust workspace that demonstrates the proposed
architecture rather than just asserting it — up to and including **running a
real trained LLM end-to-end**: it loads Karpathy's 15M-parameter TinyStories
Llama (`stories15M`, llama2.c format) and generates coherent stories at
~150 tok/s on safe-Rust scalar kernels, in f32 or Prana's Q8 quantization.

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
| [`prana-model`](crates/prana-model) | `cactus-engine` model loading / tokenizer / sampling | **real-model inference**: llama2.c checkpoint loader (f32 or Q8-at-load), Llama SentencePiece BPE tokenizer, KV-cached forward pass, greedy/temperature sampling — zero `unsafe`, zero deps |
| [`prana-cli`](crates/prana-cli) | `cactus run` / `benchmark` / `chat` | real-model text generation (`run`), transformer-block demo, microbenchmark, Phase 0 chat driver |

### Run it

```bash
cargo test                                   # 48 tests
./scripts/fetch-model.sh                     # fetch stories15M + tokenizer into models/
cargo run --release -p prana-cli -- run      # generate a story with the real model
cargo run --release -p prana-cli -- run --q8 # same, with Q8-quantized weights
cargo run --release -p prana-cli -- demo     # run one full transformer block (GQA, seq=8)
cargo run --release -p prana-cli -- bench    # microbenchmark the Q8 matmul
cargo run --release -p prana-cli -- chat     # drive the Phase 0 safe wrapper (mock engine)
```

`prana run` options: `[model.bin] [tokenizer.bin] [prompt] [--steps N]
[--temp T] [--seed S] [--q8]`. With `--temp 0` (greedy) the output matches
llama2.c's reference output for the same checkpoint, which is the correctness
check for the whole pipeline (tokenizer, RoPE convention, attention, SwiGLU).

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
**real trained model end-to-end**: `prana-model` loads llama2.c-format
checkpoints (Karpathy's TinyStories 15M/42M/110M models), tokenizes with the
Llama SentencePiece vocab, and generates via a KV-cached forward pass built
entirely from `prana-kernels` — in f32 or with weights re-quantized to Prana's
Q8 blocks at load. Greedy decoding reproduces llama2.c's reference output.
It also implements **Phase 0 of the migration plan**: `prana-cactus`, a safe
idiomatic wrapper over the engine's real C ABI, exercised against an
in-process mock of that ABI (the native static lib is ARM/Metal-only).
Still out of scope: ARM NEON/Metal targets, GGUF/safetensors loaders, and the
CQ rotation-codebook quantization; those are the migration plan's later
phases, not built here.

## License

Apache-2.0. Cactus is a separate project under its own license; Prana is an
independent evaluation and does not vendor Cactus source.
