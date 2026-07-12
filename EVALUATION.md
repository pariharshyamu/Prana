# Prana: Evaluating a Rust Rewrite of the Cactus Inference Engine

**Question posed:** clone [`cactus-compute/cactus`](https://github.com/cactus-compute/cactus)
and evaluate whether re-implementing this on-device inference engine in Rust
(as "Prana") is worth doing.

**Short answer:** A *full* big-bang rewrite is **not** worth it, and would most
likely be slower to ship and no faster at runtime. But a **targeted,
incremental Rust reimplementation of the orchestration layers — engine, graph,
tooling — on top of the existing (or gradually ported) SIMD kernels is worth
it**, and is where Rust's advantages are real rather than cosmetic. This
repository contains a small **working prototype** (`crates/`) that demonstrates
the proposed layering compiles, runs, and computes correctly.

---

## 1. What Cactus actually is

Cactus v2.0.1 is a hybrid edge-cloud AI engine for phones and wearables. It is
**~81,000 lines of C++/Objective-C++/Metal** across four native layers plus a
Python transpiler:

| Layer | Path | ~LOC | Language | Role |
|-------|------|-----:|----------|------|
| Engine | `cactus-engine/` | 32,900 | C/C++ | OpenAI-compatible APIs: chat, streaming, tool-calling, transcription, embeddings, RAG, vision, vector index, cloud handoff |
| Kernels | `cactus-kernels/` | 34,700 | C++/NEON/Metal | matmul, attention, conv, quant, DSP, norms/RoPE; `cactus_kernels.metal` alone is 185 KB |
| Graph | `cactus-graph/` | 19,600 | C++ | zero-copy computation graph, buffer pool, op set, Metal planner |
| Quants | (in kernels) | — | C++ | rotation-and-codebook quantization, CQ1–CQ4 |
| Transpiler | `python/` | — | Python | PyTorch → Cactus runtime graph |

Key facts that shape the evaluation, all verified in the source:

- **The hot path is hand-written ARM NEON.** 14 kernel files use intrinsics
  (`vld1q_*`, `vmlaq_*`, `float16x8_t`). `matmul.cpp` is 126 KB;
  `metal_backend.mm` is 135 KB.
- **The compute target is mobile ARM + Apple Metal**, not the x86 server this
  was analyzed on. Performance claims (2964 tok/s prefill on M5 Max) come from
  Accelerate/Metal + NEON tuned for specific SoCs (Apple, Samsung, Pixel).
- **Memory management is manual and intricate.** `cactus-graph/src/core.cpp`'s
  `BufferDesc` move-constructor manually nulls ~15 raw pointer fields; the
  `BufferPool` recycles `new char[]` allocations by power-of-two bucket.
- **The public boundary is already a C ABI.** `cactus-engine`'s FFI
  (`cactus_init`, `cactus_complete`, …) is consumed by bindings for Swift,
  Kotlin, Flutter, React Native, Python — **and Rust already exists**
  (`bindings/rust/cactus.rs`), but only as raw `extern "C"` declarations, not a
  Rust implementation.

---

## 2. Where Rust genuinely helps (and where it doesn't)

The instinct "rewrite the inference engine in Rust for speed" misdiagnoses
where the risk and cost actually are. Broken down by layer:

### 2a. Kernels (`cactus-kernels`) — **Rust helps least**

This is the performance-critical 35k lines, and it is exactly where a rewrite
buys the *least*:

- Peak throughput comes from NEON intrinsics and Apple's Accelerate/Metal.
  Rust can call the **same intrinsics** (`core::arch::aarch64`) and the **same
  frameworks**, so the ceiling is identical — but you'd be rewriting
  battle-tested, benchmarked SIMD for a *lateral* move.
- `unsafe` does not go away here. SIMD intrinsics, raw pointers, and FFI to
  Metal are `unsafe` in Rust too. The safety story for this layer is roughly a
  wash.
- Metal shaders (`.metal`, 185 KB) and the Objective-C++ bridge
  (`metal_backend.mm`) don't get rewritten in Rust at all — they stay as-is and
  are called through FFI.

**Verdict:** keep these kernels initially. Port them **opportunistically**,
kernel by kernel, only where a measured win or a real safety bug justifies it.
The prototype here shows the shape of a *portable* kernel tier (safe,
autovectorized scalar code) with a documented seam for a target-gated
`unsafe` NEON/AVX path behind the same function signatures.

### 2b. Graph + Engine (`cactus-graph`, `cactus-engine`) — **Rust helps most**

This is ~52k lines of buffer lifetime management, op wiring, tokenization,
sampling, constraint/grammar handling, tool-call parsing, streaming state, RAG,
and a vector index. It is:

- **Full of exactly the bugs Rust prevents.** Manual pointer nulling in move
  constructors, hand-tracked buffer ownership in the pool, `char response[4096]`
  fixed C buffers passed across FFI, `void*`-typed graph outputs the caller must
  interpret with the correct precision. None of this is compute-bound — it is
  *correctness- and lifetime-bound*, which is Rust's home turf.
- **Not on the tight inner loop.** Graph construction, scheduling, and the
  engine's request handling are dominated by the kernel calls they dispatch, so
  expressing them in safe Rust costs ~nothing at runtime.
- **Where concurrency bugs live.** The custom `ThreadPool` with manual core
  pinning (`threading.h`) is precisely the kind of shared-state code where
  Rust's `Send`/`Sync` and scoped threads turn "correct by careful review" into
  "correct by compiler." The prototype's `parallel_for` uses `thread::scope` so
  a worker provably cannot outlive the slices it borrows.

**Verdict:** this is the rewrite worth doing. You get memory safety, no data
races, `Result`-based error handling instead of log-and-return-null, and a
dramatically smaller `unsafe` surface — while keeping performance.

### 2c. Tooling / bindings / server — **Rust helps and is easy**

The CLI, the OpenAI-compatible HTTP server (`cactus serve`), model download/
convert plumbing, and the coding agent are a natural fit for Rust's ecosystem
(`tokio`, `axum`/`hyper`, `clap`, `serde`). These are low-risk, high-ergonomics
wins and a sensible place to start because they can wrap the existing C ABI on
day one.

---

## 3. The honest risks and costs

1. **Rewrites lose accumulated correctness.** 81k lines encode years of
   edge-case handling (per-SoC quirks, tokenizer corner cases, quantization
   numerics). A from-scratch rewrite silently drops these until each is
   rediscovered as a bug. This is the single biggest reason not to big-bang it.
2. **You don't escape `unsafe`.** SIMD, Metal FFI, and mmap'd weight loading are
   `unsafe` in Rust. The win is *concentrating* `unsafe` into a small audited
   core, not eliminating it. Overselling "safe rewrite" is a mistake.
3. **Metal/Objective-C++ interop stays C/ObjC.** `metal_backend.mm` and the
   `.metal` shaders are not becoming Rust. Rust orchestrates them via FFI.
4. **Quantization numerics are subtle.** The CQ rotation+codebook scheme is the
   product's quality moat (the accuracy tables in the README). Reimplementing it
   must be bit-accuracy-validated against the C++ reference, not just
   "looks close."
5. **Mobile toolchain friction.** Rust → iOS/Android static libs, `xcframework`
   packaging, and 6+ language bindings is real integration work that C++ already
   has wired up in this repo.
6. **Effort is large.** A credible incremental port of engine+graph to Rust,
   keeping kernels via FFI, is a multi-engineer, multi-quarter effort — not a
   weekend.

---

## 4. Recommended strategy: strangler-fig, not big-bang

The right path exploits the fact that **Cactus already exposes a C ABI**:

```
Phase 0  Rust CLI + server wrapping the existing libcactus_engine.a via FFI.
         (The raw bindings already exist; make them a safe, idiomatic crate.)
Phase 1  Reimplement the engine's orchestration in Rust — request handling,
         streaming, sampling, tokenizer, tool-call/grammar parsing, RAG,
         vector index — still calling C++ graph+kernels underneath.
Phase 2  Port the computation graph + buffer pool to safe Rust (this is the
         highest safety payoff, near-zero perf cost). Kernels stay FFI.
Phase 3  Opportunistically port individual kernels to Rust with target-gated
         intrinsics, one at a time, each gated behind a benchmark that proves
         parity with the C++/NEON original. Keep Metal shaders as-is.
```

Each phase ships a working product and is independently reversible. At no point
is there a months-long "nothing works yet" valley.

---

## 5. What the prototype in this repo demonstrates

`crates/` is a small but **real, compiling, tested** Rust workspace that
concretely shows Phases 1–3 are feasible and what the layering looks like:

| Crate | Mirrors in Cactus | Shows |
|-------|-------------------|-------|
| `prana-tensor` | `BufferDesc`, `BufferPool` (`core.cpp`) | dtype-erased tensors + a size-bucketed buffer pool with **zero `unsafe`** (`#![forbid(unsafe_code)]`) — the manual pointer-nulling move-ctor becomes compiler-tracked ownership |
| `prana-kernels` | `matmul.cpp`, `quants.cpp`, `norms_rope.cpp`, `threading.h` | Q8 block-quantized matmul, dense matmul, RMSNorm, softmax, and a `thread::scope` `parallel_for`; scalar autovectorized *default* tier with a documented seam for a NEON/AVX `unsafe` tier |
| `prana-graph` | `CactusGraph` (`builder.cpp`, `execute.cpp`) | define-then-run graph where node handles are checked indices, ops validate shapes, and an unbound input yields a typed `Result` error instead of UB |
| `prana-cli` | `cactus run` / `cactus benchmark` | runs a transformer-style block end-to-end and microbenchmarks the decode-path matmul |

Everything above the kernel boundary is `#![forbid(unsafe_code)]`. Run it:

```bash
cargo test              # 17 tests across the workspace
cargo run --release -p prana-cli -- demo
cargo run --release -p prana-cli -- bench
```

### What the benchmark honestly shows

On the x86 dev box, the Q8 kernel's **measured win is a 3.5× smaller weight
footprint at ~0.08% relative error**, *not* raw speed: with the scalar tier the
on-the-fly `i8→f32` dequant is compute-bound at cache-resident sizes and is
slightly slower than dense FMA. The speed win from quantization shows up (a) at
model sizes where weights spill cache and bandwidth dominates — the actual
on-device regime — and (b) once the target-gated SIMD tier replaces the scalar
dot product. The prototype is deliberately explicit about this rather than
quoting a flattering number. It is a **layering and correctness demonstration**,
not a competitor to the tuned NEON/Metal kernels.

---

## 6. Bottom line

| | |
|---|---|
| **Rewrite everything in Rust for speed?** | No. The speed ceiling is set by NEON/Metal, which Rust reaches but does not exceed. A big-bang rewrite risks regressions and delays with little runtime upside. |
| **Reimplement engine + graph + tooling in Rust for safety and maintainability, incrementally, over the C ABI?** | **Yes — this is the real opportunity.** ~52k lines of lifetime/ownership/concurrency-heavy code move to a domain where the compiler eliminates whole bug classes at ~zero perf cost. |
| **Port kernels?** | Only opportunistically, each behind a parity benchmark. Keep Metal shaders. |
| **Recommended first step** | A safe idiomatic Rust crate over the existing `libcactus_engine` C ABI (Phase 0), then strangle inward. |

Prana is worth building — as a Rust *reimagining of the orchestration layers*
that treats Cactus's kernels as an asset to preserve and gradually absorb, not
as 81k lines to retype.
