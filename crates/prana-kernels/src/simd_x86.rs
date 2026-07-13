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

    /// AVX2+FMA native Q4_K row dot over packed 144-byte super-blocks.
    /// `asums[g]` must hold the activation sum of 32-group `g` (the affine
    /// `−dmin·m` term needs only Σa, not the per-element products).
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA (see [`available`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4k(a: &[f32], row: &[u8], asums: &[f32]) -> f32 {
        use crate::kquant::{q4k_scale_min, Q4_K_BLOCK_BYTES, QK_K};
        debug_assert_eq!(row.len() % Q4_K_BLOCK_BYTES, 0);
        debug_assert_eq!(a.len(), row.len() / Q4_K_BLOCK_BYTES * QK_K);

        let mut total = 0f32;
        // SAFETY: every load below stays inside one 144-byte block (16-byte
        // loads at qs offsets 0/16 of a 128-byte field) and inside a's
        // matching 256-float window, both guaranteed by the debug-checked
        // length relations; loadu tolerates unaligned addresses.
        unsafe {
            let nib_mask = _mm_set1_epi8(0x0F);
            for (b_idx, b) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
                let d = crate::f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                let dmin = crate::f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
                let scales = &b[4..16];
                let qs = b.as_ptr().add(16);
                let a_blk = a.as_ptr().add(b_idx * QK_K);
                let s_blk = &asums[b_idx * 8..(b_idx + 1) * 8];

                for pair in 0..4 {
                    let q = qs.add(pair * 32);
                    let a_lo = a_blk.add(pair * 64);
                    let a_hi = a_blk.add(pair * 64 + 32);
                    let mut acc_lo = _mm256_setzero_ps();
                    let mut acc_hi = _mm256_setzero_ps();
                    // 32 bytes = two 16-byte halves; each half yields 16 low
                    // and 16 high nibbles, converted 8 lanes at a time.
                    for h in 0..2 {
                        let bytes = _mm_loadu_si128(q.add(h * 16) as *const __m128i);
                        let lo = _mm_and_si128(bytes, nib_mask);
                        let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), nib_mask);
                        for j in 0..2 {
                            let off = h * 16 + j * 8;
                            let lo8 = if j == 0 { lo } else { _mm_srli_si128(lo, 8) };
                            let hi8 = if j == 0 { hi } else { _mm_srli_si128(hi, 8) };
                            let lo_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo8));
                            let hi_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi8));
                            acc_lo = _mm256_fmadd_ps(_mm256_loadu_ps(a_lo.add(off)), lo_f, acc_lo);
                            acc_hi = _mm256_fmadd_ps(_mm256_loadu_ps(a_hi.add(off)), hi_f, acc_hi);
                        }
                    }
                    let (sc1, m1) = q4k_scale_min(pair * 2, scales);
                    let (sc2, m2) = q4k_scale_min(pair * 2 + 1, scales);
                    total += d * sc1 * hsum(acc_lo) - dmin * m1 * s_blk[pair * 2];
                    total += d * sc2 * hsum(acc_hi) - dmin * m2 * s_blk[pair * 2 + 1];
                }
            }
        }
        total
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
    /// # Safety
    /// Never callable: `available()` is always false on this target.
    pub unsafe fn dot_q4k(_a: &[f32], _row: &[u8], _asums: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
}

pub use imp::{available, dot_f32, dot_q4k, dot_q8};

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
