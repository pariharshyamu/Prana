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
//! 2. **Fast tier — small, audited `unsafe`.** The same kernel signatures get
//!    a target-gated intrinsics implementation, selected at runtime. This
//!    ships today for x86-64 (`simd_x86.rs`: AVX2+FMA dot products, ~100 lines
//!    of commented `unsafe`); an aarch64 NEON twin slots in beside it the same
//!    way. That `unsafe` surface is a few hundred lines, not tens of
//!    thousands, because everything above the kernel boundary is safe.
//!
//! Tier 1 keeps the prototype correct and portable everywhere; tier 2 is used
//! automatically wherever the CPU supports it, and every SIMD kernel is tested
//! for parity against its scalar twin.

mod attention;
mod matmul;
mod norms;
mod pool;
mod simd_x86;
mod threading;

pub use attention::{attention, attention_decode, rope, rope_interleaved};
pub use matmul::{dequantize_q8, matmul_f32, matmul_q8_f32, quantize_q8, QuantMatrix, Q8_BLOCK};
pub use norms::{rmsnorm, silu, softmax};
pub use pool::{global as pool, Pool};
pub use threading::parallel_for;
