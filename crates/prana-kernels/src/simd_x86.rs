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

    /// One-time check for AVX-VNNI (`vpdpbusd` on 256-bit registers without
    /// AVX-512 — Alder Lake onward). Fuses the multiply-accumulate that
    /// takes `maddubs`+`madd` on plain AVX2 into one instruction.
    pub fn available_vnni() -> bool {
        static AVAIL: OnceLock<bool> = OnceLock::new();
        *AVAIL.get_or_init(|| available() && is_x86_feature_detected!("avxvnni"))
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

    /// Sum of the 32 int8×int8 products in `x`·`y` as 8 i32 lanes.
    /// Both inputs signed; `maddubs` needs one unsigned operand, so move
    /// x's sign onto y (llama.cpp's `mul_sum_i8_pairs`): |x|·sign(y,x) ≡ x·y.
    ///
    /// # Safety
    /// Requires AVX2 (guaranteed by the callers' target feature).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn i8_dot_i32(x: __m256i, y: __m256i) -> __m256i {
        // Pure register ops — no memory access.
        let ax = _mm256_sign_epi8(x, x);
        let sy = _mm256_sign_epi8(y, x);
        // u8×i8 pairs → i16 (max 2·127·127, fits), then i16 pairs → i32.
        _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1))
    }

    /// AVX-VNNI Q8×Q8 dot: as [`dot_q8_q8`], with `vpdpbusd` replacing the
    /// `maddubs`+`madd` pair (32 MACs → i32 lanes in one instruction).
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA+AVX-VNNI
    /// (see [`available_vnni`]).
    #[target_feature(enable = "avx2,fma,avxvnni")]
    pub unsafe fn dot_q8_q8_vnni(acts: &crate::matmul::QuantActs, q: &[i8], scales: &[f32]) -> f32 {
        debug_assert_eq!(acts.q.len(), q.len());
        debug_assert_eq!(q.len(), scales.len() * 32);
        // SAFETY: same bounds as dot_q8_q8; sign-trick keeps operands in
        // dpbusd's unsigned×signed domain.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let (pa, pq) = (acts.q.as_ptr(), q.as_ptr());
            for (g, &sw) in scales.iter().enumerate() {
                let qa = _mm256_loadu_si256(pa.add(g * 32) as *const __m256i);
                let qw = _mm256_loadu_si256(pq.add(g * 32) as *const __m256i);
                let ax = _mm256_sign_epi8(qw, qw);
                let sy = _mm256_sign_epi8(qa, qw);
                let p = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sw * acts.scales[g]), acc);
            }
            hsum(acc)
        }
    }

    /// AVX-VNNI Q4_0 row dot: as [`dot_q40_q8`] with fused int MACs.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA+AVX-VNNI
    /// (see [`available_vnni`]).
    #[target_feature(enable = "avx2,fma,avxvnni")]
    pub unsafe fn dot_q40_q8_vnni(acts: &crate::matmul::QuantActs, row: &[u8], ds: &[f32]) -> f32 {
        use crate::kquant::{Q4_0_BLOCK, Q4_0_BLOCK_BYTES};
        debug_assert_eq!(row.len() % Q4_0_BLOCK_BYTES, 0);
        debug_assert_eq!(acts.q.len(), row.len() / Q4_0_BLOCK_BYTES * Q4_0_BLOCK);
        debug_assert_eq!(ds.len(), row.len() / Q4_0_BLOCK_BYTES);

        let n_blocks = row.len() / Q4_0_BLOCK_BYTES;
        // SAFETY: same bounds as dot_q40_q8.
        unsafe {
            let (pr, pa) = (row.as_ptr(), acts.q.as_ptr());
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let zero = _mm256_setzero_si256();
            let mut i = 0;
            while i + 2 <= n_blocks {
                let qw0 = q40_block_i8(pr.add(i * Q4_0_BLOCK_BYTES + 2));
                let qw1 = q40_block_i8(pr.add((i + 1) * Q4_0_BLOCK_BYTES + 2));
                let qa0 = _mm256_loadu_si256(pa.add(i * 32) as *const __m256i);
                let qa1 = _mm256_loadu_si256(pa.add(i * 32 + 32) as *const __m256i);
                let p0 = _mm256_dpbusd_avx_epi32(zero, _mm256_sign_epi8(qw0, qw0), _mm256_sign_epi8(qa0, qw0));
                let p1 = _mm256_dpbusd_avx_epi32(zero, _mm256_sign_epi8(qw1, qw1), _mm256_sign_epi8(qa1, qw1));
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(ds[i] * acts.scales[i]), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(ds[i + 1] * acts.scales[i + 1]), acc1);
                i += 2;
            }
            if i < n_blocks {
                let qw = q40_block_i8(pr.add(i * Q4_0_BLOCK_BYTES + 2));
                let qa = _mm256_loadu_si256(pa.add(i * 32) as *const __m256i);
                let p = _mm256_dpbusd_avx_epi32(zero, _mm256_sign_epi8(qw, qw), _mm256_sign_epi8(qa, qw));
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(ds[i] * acts.scales[i]), acc0);
            }
            hsum(_mm256_add_ps(acc0, acc1))
        }
    }

    /// AVX2 integer Q8×Q8 dot: `Σ_g sw[g]·sa[g]·Σ_32 qw·qa`.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA (see [`available`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8_q8(acts: &crate::matmul::QuantActs, q: &[i8], scales: &[f32]) -> f32 {
        debug_assert_eq!(acts.q.len(), q.len());
        debug_assert_eq!(q.len(), scales.len() * 32);
        // SAFETY: per group g both loads cover [g*32, g*32+32), in-bounds by
        // the debug-checked length relations; loadu tolerates unalignment.
        // Lane sums stay < 2^17, exact in f32 after the convert.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let (pa, pq) = (acts.q.as_ptr(), q.as_ptr());
            for (g, &sw) in scales.iter().enumerate() {
                let qa = _mm256_loadu_si256(pa.add(g * 32) as *const __m256i);
                let qw = _mm256_loadu_si256(pq.add(g * 32) as *const __m256i);
                let prod = _mm256_cvtepi32_ps(i8_dot_i32(qw, qa));
                acc = _mm256_fmadd_ps(prod, _mm256_set1_ps(sw * acts.scales[g]), acc);
            }
            hsum(acc)
        }
    }

    /// Unpack one Q4_0 block's 16 quant bytes into 32 *centered* i8 lanes
    /// (`q − 8` ∈ −8..=7), llama.cpp's `bytes_from_nibbles_32` + offset.
    ///
    /// # Safety
    /// Requires AVX2; `qs` must point at 16 readable bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn q40_block_i8(qs: *const u8) -> __m256i {
        // SAFETY: one 16-byte load from `qs` (caller guarantees); the rest
        // are register ops.
        unsafe {
            let bytes = _mm_loadu_si128(qs as *const __m128i);
            let nib = _mm_set1_epi8(0x0F);
            let lo = _mm_and_si128(bytes, nib); // elements 0..16
            let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), nib); // 16..32
            _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8))
        }
    }

    /// AVX2 integer Q4_0 row dot over packed 18-byte blocks:
    /// `Σ a·d(q−8) ≈ d·sa·Σ qa·(q−8)`, with the −8 folded into the integer
    /// lanes (no scalar correction chain — that serial dependency capped
    /// this kernel's throughput). Two blocks per iteration on independent
    /// accumulators for ILP.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA (see [`available`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q40_q8(acts: &crate::matmul::QuantActs, row: &[u8], ds: &[f32]) -> f32 {
        use crate::kquant::{Q4_0_BLOCK, Q4_0_BLOCK_BYTES};
        debug_assert_eq!(row.len() % Q4_0_BLOCK_BYTES, 0);
        debug_assert_eq!(acts.q.len(), row.len() / Q4_0_BLOCK_BYTES * Q4_0_BLOCK);
        debug_assert_eq!(ds.len(), row.len() / Q4_0_BLOCK_BYTES);

        let n_blocks = row.len() / Q4_0_BLOCK_BYTES;
        // SAFETY: block i's quant bytes live at [i*18+2, i*18+18) and its
        // activations at [i*32, i*32+32), in-bounds by the debug-checked
        // length relations; loadu tolerates unalignment.
        unsafe {
            let (pr, pa) = (row.as_ptr(), acts.q.as_ptr());
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 2 <= n_blocks {
                let qw0 = q40_block_i8(pr.add(i * Q4_0_BLOCK_BYTES + 2));
                let qw1 = q40_block_i8(pr.add((i + 1) * Q4_0_BLOCK_BYTES + 2));
                let qa0 = _mm256_loadu_si256(pa.add(i * 32) as *const __m256i);
                let qa1 = _mm256_loadu_si256(pa.add(i * 32 + 32) as *const __m256i);
                let p0 = _mm256_cvtepi32_ps(i8_dot_i32(qw0, qa0));
                let p1 = _mm256_cvtepi32_ps(i8_dot_i32(qw1, qa1));
                acc0 = _mm256_fmadd_ps(p0, _mm256_set1_ps(ds[i] * acts.scales[i]), acc0);
                acc1 = _mm256_fmadd_ps(p1, _mm256_set1_ps(ds[i + 1] * acts.scales[i + 1]), acc1);
                i += 2;
            }
            if i < n_blocks {
                let qw = q40_block_i8(pr.add(i * Q4_0_BLOCK_BYTES + 2));
                let qa = _mm256_loadu_si256(pa.add(i * 32) as *const __m256i);
                let p = _mm256_cvtepi32_ps(i8_dot_i32(qw, qa));
                acc0 = _mm256_fmadd_ps(p, _mm256_set1_ps(ds[i] * acts.scales[i]), acc0);
            }
            hsum(_mm256_add_ps(acc0, acc1))
        }
    }

    /// AVX2 integer Q4_K row dot over packed 144-byte super-blocks: per
    /// 32-group, `Σ a·(d·sc·q − dmin·m) ≈ sa·(d·sc·Σ qa·q − dmin·m·Σ qa)`.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2+FMA (see [`available`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4k_q8(acts: &crate::matmul::QuantActs, row: &[u8], ds: &[f32]) -> f32 {
        use crate::kquant::{q4k_scale_min, Q4_K_BLOCK_BYTES, QK_K};
        debug_assert_eq!(row.len() % Q4_K_BLOCK_BYTES, 0);
        debug_assert_eq!(acts.q.len(), row.len() / Q4_K_BLOCK_BYTES * QK_K);
        debug_assert_eq!(ds.len(), row.len() / Q4_K_BLOCK_BYTES * 2);

        let mut total = 0f32;
        // SAFETY: every load stays inside one 144-byte block (32 qs bytes at
        // offsets 16/48/80/112) or the matching 256-i8 activation window,
        // in-bounds by the debug-checked length relations. Group sums stay
        // < 2^17 per lane pair, exact in f32 after the convert.
        unsafe {
            let nib_mask = _mm256_set1_epi8(0x0F);
            let ones = _mm256_set1_epi16(1);
            for (b_idx, b) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
                let (d, dmin) = (ds[b_idx * 2], ds[b_idx * 2 + 1]);
                let scales = &b[4..16];
                let qs = b.as_ptr().add(16);
                let pa = acts.q.as_ptr().add(b_idx * QK_K);

                for pair in 0..4 {
                    // 32 qs bytes: low nibbles are group 2p, high are 2p+1.
                    let bytes = _mm256_loadu_si256(qs.add(pair * 32) as *const __m256i);
                    let lo = _mm256_and_si256(bytes, nib_mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nib_mask);
                    let qa_lo = _mm256_loadu_si256(pa.add(pair * 64) as *const __m256i);
                    let qa_hi = _mm256_loadu_si256(pa.add(pair * 64 + 32) as *const __m256i);
                    let s1 = hsum(_mm256_cvtepi32_ps(_mm256_madd_epi16(
                        _mm256_maddubs_epi16(lo, qa_lo),
                        ones,
                    )));
                    let s2 = hsum(_mm256_cvtepi32_ps(_mm256_madd_epi16(
                        _mm256_maddubs_epi16(hi, qa_hi),
                        ones,
                    )));
                    let g_lo = b_idx * 8 + pair * 2;
                    let g_hi = g_lo + 1;
                    let (sc1, m1) = q4k_scale_min(pair * 2, scales);
                    let (sc2, m2) = q4k_scale_min(pair * 2 + 1, scales);
                    total += acts.scales[g_lo]
                        * (d * sc1 * s1 - dmin * m1 * acts.sums[g_lo] as f32);
                    total += acts.scales[g_hi]
                        * (d * sc2 * s2 - dmin * m2 * acts.sums[g_hi] as f32);
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
    pub fn available_vnni() -> bool {
        false
    }
    /// # Safety
    /// Never callable: `available_vnni()` is always false on this target.
    pub unsafe fn dot_q8_q8_vnni(_acts: &crate::matmul::QuantActs, _q: &[i8], _scales: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
    /// # Safety
    /// Never callable: `available_vnni()` is always false on this target.
    pub unsafe fn dot_q40_q8_vnni(_acts: &crate::matmul::QuantActs, _row: &[u8], _ds: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
    /// # Safety
    /// Never callable: `available()` is always false on this target.
    pub unsafe fn dot_f32(_a: &[f32], _b: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
    /// # Safety
    /// Never callable: `available()` is always false on this target.
    pub unsafe fn dot_q8_q8(_acts: &crate::matmul::QuantActs, _q: &[i8], _scales: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
    /// # Safety
    /// Never callable: `available()` is always false on this target.
    pub unsafe fn dot_q4k_q8(_acts: &crate::matmul::QuantActs, _row: &[u8], _ds: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
    /// # Safety
    /// Never callable: `available()` is always false on this target.
    pub unsafe fn dot_q40_q8(_acts: &crate::matmul::QuantActs, _row: &[u8], _ds: &[f32]) -> f32 {
        unreachable!("simd tier unavailable on this target")
    }
}

pub use imp::{
    available, available_vnni, dot_f32, dot_q40_q8, dot_q40_q8_vnni, dot_q4k_q8, dot_q8_q8,
    dot_q8_q8_vnni,
};

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use crate::matmul::dot_f32_scalar;

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
    fn simd_dot_q8_q8_matches_scalar() {
        if !super::available() {
            eprintln!("skipping: no AVX2+FMA on this CPU");
            return;
        }
        for blocks in [1usize, 2, 9, 64] {
            let n = blocks * 32;
            // Extreme activations included: ±amax quantizes to ±127, the
            // worst case for the maddubs i16 bound.
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin() * 3.0).collect();
            let acts = crate::matmul::quantize_acts(&a);
            let q: Vec<i8> = (0..n).map(|i| ((i * 37) % 255) as i8).collect();
            let scales: Vec<f32> = (0..blocks).map(|g| 0.01 + g as f32 * 0.003).collect();
            let scalar = crate::matmul::dot_q8_q8_scalar(&acts, &q, &scales);
            // SAFETY: available() checked above.
            let simd = unsafe { super::dot_q8_q8(&acts, &q, &scales) };
            let tol = 1e-3 * (n as f32).sqrt();
            assert!((scalar - simd).abs() < tol, "blocks={blocks}: {scalar} vs {simd}");
        }
    }
}
