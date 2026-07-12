# Prana

**A feasibility study + working prototype for a Rust reimplementation of the
[Cactus](https://github.com/cactus-compute/cactus) on-device AI inference
engine.**

Prana is not (yet) a production engine. It is the answer to a specific
question — *"is it worth rewriting Cactus's inference engine in Rust?"* —
backed by a small, compiling, tested Rust workspace that demonstrates the
proposed architecture rather than just asserting it.

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
| [`prana-kernels`](crates/prana-kernels) | `matmul` / `quants` / `norms_rope` / `threading` | Q8 quantized matmul, dense matmul, RMSNorm, softmax, scoped-thread `parallel_for` |
| [`prana-graph`](crates/prana-graph) | `CactusGraph` | checked-handle, shape-validating, define-then-run graph |
| [`prana-cli`](crates/prana-cli) | `cactus run` / `benchmark` | end-to-end demo + decode-path microbenchmark |

### Run it

```bash
cargo test                                   # 17 tests
cargo run --release -p prana-cli -- demo     # run one transformer-style block
cargo run --release -p prana-cli -- bench    # microbenchmark the Q8 matmul
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

This is a prototype to support an architectural decision. It implements a small
op set (matmul, quantized matmul, RMSNorm, add, softmax) — enough to run one
transformer block and prove the tensor → kernels → graph layering composes and
computes correctly. It does **not** load real model weights, target ARM/Metal,
or implement the CQ rotation-codebook quantization; those are scoped in the
migration plan, not built here.

## License

Apache-2.0. Cactus is a separate project under its own license; Prana is an
independent evaluation and does not vendor Cactus source.
