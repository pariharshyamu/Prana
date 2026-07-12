//! Core tensor primitives for Prana.
//!
//! This crate is `#![forbid(unsafe_code)]` on purpose: the whole point of the
//! Prana experiment is to show that the *orchestration* layers of an inference
//! engine (shapes, dtypes, buffer lifetime, graph wiring) can be expressed with
//! zero `unsafe`, pushing all remaining `unsafe` down into a small, auditable
//! set of SIMD kernels (see `prana-kernels`).
//!
//! In Cactus the equivalent code lives in `cactus-graph/src/core.cpp`
//! (`BufferPool`, `BufferDesc`) where buffer ownership is tracked by hand with
//! raw `new[]`/`delete[]` and a hand-written move constructor that nulls out ~15
//! pointer fields. Here the compiler tracks all of that for us.

mod dtype;
mod pool;
mod tensor;

pub use dtype::DType;
pub use pool::BufferPool;
pub use tensor::{Shape, Tensor};
