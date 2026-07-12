//! Compute kernels for Prana.
//!
//! # Design intent (this is the crux of the feasibility question)
//!
//! Cactus's hot path is ~35k lines of hand-written ARM NEON in
//! `cactus-kernels/`. The open question a Rust rewrite has to answer is:
//! *can we keep that performance without keeping that much unsafe code?*
//!
//! Prana's answer, demonstrated here, is a two-tier strategy:
//!
//! 1. **Default tier — safe, autovectorized.** The scalar kernels below are
//!    written in a shape LLVM reliably turns into SIMD (`chunks_exact`,
//!    accumulator arrays, no aliasing). No `unsafe`, portable to every target,
//!    and already within a small constant factor of hand-tuned code.
//! 2. **Fast tier — small, audited `unsafe`.** For a production build the same
//!    kernel signatures get a target-gated intrinsics implementation (NEON on
//!    ARM, AVX on x86). That `unsafe` surface is a few hundred lines, not tens
//!    of thousands, because everything above the kernel boundary is safe.
//!
//! This file ships tier 1 so the prototype builds and runs on any host,
//! including the x86_64 CI box this was developed on.

mod matmul;
mod norms;
mod threading;

pub use matmul::{dequantize_q8, matmul_f32, matmul_q8_f32, quantize_q8, QuantMatrix, Q8_BLOCK};
pub use norms::{rmsnorm, softmax};
pub use threading::parallel_for;
