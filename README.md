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
| [`prana-model`](crates/prana-model) | `cactus-engine` model loading / tokenizer / sampling | **real-model inference**: llama2.c *and GGUF* loaders for **llama, qwen2, and gemma** architectures (F32/F16/Q4_0/Q8_0/Q4_K/Q6_K, all quantized types running natively in packed form, embedding rows dequantized per lookup), SentencePiece *and* GPT-2 byte-level BPE tokenizers with control/special tokens, ChatML chat mode, KV-cached forward pass with QKV biases / NeoX RoPE / GELU / decoupled head_dim, greedy/temperature sampling — zero `unsafe`, zero deps |
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

# An off-the-shelf instruct model (any qwen2-arch q4_0/q8_0/q4_k_m GGUF):
cargo run --release -p prana-cli -- run qwen2.5-0.5b-instruct-q4_0.gguf "Who are you?" --chat
```

`prana run` options: `[model.bin tokenizer.bin | model.gguf] [prompt]
[--steps N] [--temp T] [--seed S] [--q8] [--chat]`. With `--temp 0` (greedy)
the output is token-identical to llama2.c's reference output for the same
checkpoint — via both container formats — which is the correctness check for
the whole pipeline (tokenizer, RoPE convention, attention, SwiGLU, SIMD
kernels). The AVX2+FMA SIMD tier engages automatically when the CPU supports
it. `--chat` wraps the prompt in the ChatML template instruct models are
trained on (the `<|im_start|>` markers encode via the tokenizer's control
tokens and generation stops at `<|im_end|>`).

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
  dense  f32 matmul :   1.141 ms/iter      7.4 GFLOP/s
  quant  q8  matmul :   0.085 ms/iter     99.0 GFLOP/s
  weight memory     : 16384 KiB f32  ->  4608 KiB q8  (3.56x smaller)
  max abs error     : 1.2964  (0.13% of output RMS)
```

The quantized matmul runs **integer dot products**: activations are quantized
to i8 per 32-group once per call (llama.cpp's approach), so the inner loop is
int8×int8 — 32 MACs per AVX2 `maddubs`+`madd` pair versus 8 f32 FMA lanes —
at ~0.1%-of-RMS accuracy cost. The honest caveat now points the other way:
this cache-resident kernel ratio (13×) overstates the end-to-end gain, because
real-model decode is DRAM-bandwidth-bound; the integer tier mainly buys
compute headroom (prefill, many-core scaling) rather than single-stream
decode tok/s. `PRANA_THREADS` overrides the worker-pool size — worth sweeping
on hybrid P+E-core parts. See EVALUATION.md §5.

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
GPT-2 BPE tokenizer with control/special tokens from GGUF metadata) and
**Gemma** ((1+w) RMSNorm folded at load, √dim embedding scale, tanh-GELU MLP,
decoupled head_dim). The Qwen2 path is **verified on real weights**: an
off-the-shelf `Qwen2.5-0.5B-Instruct` q4_0 GGUF loads, answers ChatML prompts
coherently as an assistant, and stops at `<|im_end|>`; Gemma remains verified
via synthetic GGUF fixtures (loader knobs, norm folding, finite deterministic
forward passes), and the llama path is verified token-identical against
llama2.c's reference output. Quantized tensors (Q4_0/Q4_K/Q6_K) run
**natively in packed form** (4.5/6.6 bits per weight in RAM) with the AVX2
tier — including the token-embedding table, whose rows dequantize per lookup
instead of expanding to f32 at load — and native dots are parity-tested
against dequantized dense matmuls, with all quantized matmuls running
int8×int8 integer inner loops against per-32-group-quantized activations.
On an Intel Core Ultra 5 laptop the 0.5B instruct model decodes at ~30 tok/s
(vs ~18 before packed-native Q4_0) with ~410 MiB of weight memory (vs
~1.1 GiB); throughput there is DRAM-bandwidth- and thermal-bound, not
kernel-bound.

Still out of scope: ARM NEON/Metal targets, safetensors, further
architectures (Phi, Gemma-2 softcapping), exact GPT-2 pre-tokenizer regex,
and the CQ rotation-codebook quantization; later phases, not built here.

## License

Apache-2.0. Cactus is a separate project under its own license; Prana is an
independent evaluation and does not vendor Cactus source.
