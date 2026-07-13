//! The "fast tier" promised in the crate docs: hand-written SIMD dot products
//! behind the same signatures as the scalar tier, selected at runtime.
//!
//! This is the *only* module in the workspace (besides FFI) that contains
//! `unsafe`, and every unsafe operation is a CPU intrinsic whose precondition
//! is a CPU feature checked once at startup ([`available`]) plus in-bounds
//! pointer arithmetic derived directly from slice lengths. On non-x86 targets
//! the module compiles to a stub that always reports unavailable, and the
//! scalar tier runs — an aarch64 NEON twin would slot in beside this file the
//! same way (Phase 3 of the migration plan).

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    /// One-time CPU feature check: AVX2 + FMA.
    pub fn available() -> bool {
        static AVAIL: OnceLock<bool> = OnceLock::new();
        *AVAIL.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }

    /// Horizontal sum of an 8-lane f32 vector.
    ///
    /// # Safety
    /// Requires AVX (guaranteed by the callers' `avx2` target feature).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        // Pure register ops — safe inside a target_feature fn, no memory access.
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
        _mm_cvtss_f32(s)
    }

    /// AVX2+FMA dense dot product.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA (see [`available`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 16;
        // SAFETY: all loads are within a[..chunks*16] / b[..chunks*16], which
        // is in-bounds by construction; loadu allows unaligned addresses.
        unsafe {
            let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            for i in 0..chunks {
                let o = i * 16;
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(o)), _mm256_loadu_ps(pb.add(o)), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(o + 8)), _mm256_loadu_ps(pb.add(o + 8)), acc1);
            }
            let mut sum = hsum(_mm256_add_ps(acc0, acc1));
            for i in chunks * 16..n {
                sum += a[i] * b[i];
            }
            sum
        }
    }

    /// AVX2+FMA Q8-block dot product: `sum_g scale[g] * dot(a[g], q[g])`,
    /// with 32-wide blocks (crate::Q8_BLOCK).
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA (see [`available`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8(a: &[f32], q: &[i8], scales: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), q.len());
        debug_assert_eq!(a.len(), scales.len() * 32);
        // SAFETY: per block g we touch a[g*32..g*32+32] and q likewise, both
        // in-bounds because a.len() == q.len() == scales.len()*32.
        unsafe {
            let mut total = _mm256_setzero_ps();
            let (pa, pq) = (a.as_ptr(), q.as_ptr());
            for (g, &scale) in scales.iter().enumerate() {
                let base = g * 32;
                let mut blk = _mm256_setzero_ps();
                for j in 0..4 {
                    let o = base + j * 8;
                    // 8 int8 -> 8 int32 -> 8 f32, then FMA with activations.
                    let qi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(pq.add(o) as *const __m128i));
                    let qf = _mm256_cvtepi32_ps(qi);
                    blk = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(o)), qf, blk);
                }
                total = _mm256_fmadd_ps(blk, _mm256_set1_ps(scale), total);
            }
            hsum(total)
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    pub fn available() -> bool {
        false
    }
    /// # Safety
    /// Never callable: `available()` is always false on this target.
    pub unsafe fn dot_f32(_a: &[f32], _b: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
    /// # Safety
    /// Never callable: `available()` is always false on this target.
    pub unsafe fn dot_q8(_a: &[f32], _q: &[i8], _scales: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
}

pub use imp::{available, dot_f32, dot_q8};

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use crate::matmul::{dot_f32_scalar, dot_q8_scalar};

    #[test]
    fn simd_dot_f32_matches_scalar() {
        if !super::available() {
            eprintln!("skipping: no AVX2+FMA on this CPU");
            return;
        }
        for n in [0usize, 1, 7, 16, 33, 288, 2048, 2051] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).cos()).collect();
            let scalar = dot_f32_scalar(&a, &b);
            // SAFETY: available() checked above.
            let simd = unsafe { super::dot_f32(&a, &b) };
            let tol = 1e-4 * (n.max(1) as f32).sqrt();
            assert!((scalar - simd).abs() < tol, "n={n}: {scalar} vs {simd}");
        }
    }

    #[test]
    fn simd_dot_q8_matches_scalar() {
        if !super::available() {
            eprintln!("skipping: no AVX2+FMA on this CPU");
            return;
        }
        for blocks in [1usize, 2, 9, 64] {
            let n = blocks * 32;
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).sin()).collect();
            let q: Vec<i8> = (0..n).map(|i| ((i * 37) % 255) as i8).collect();
            let scales: Vec<f32> = (0..blocks).map(|g| 0.01 + g as f32 * 0.003).collect();
            let scalar = dot_q8_scalar(&a, &q, &scales);
            // SAFETY: available() checked above.
            let simd = unsafe { super::dot_q8(&a, &q, &scales) };
            let tol = 1e-3 * (n as f32).sqrt();
            assert!((scalar - simd).abs() < tol, "blocks={blocks}: {scalar} vs {simd}");
        }
    }
}
