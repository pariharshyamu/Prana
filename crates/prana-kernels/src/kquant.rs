//! Packed quantized block formats (Q4_0 / Q4_K / Q6_K): dequantization *and*
//! native matmul.
//!
//! Q4_K/Q6_K are llama.cpp's super-block formats — 256 weights per block with
//! hierarchical scales — the dominant types inside `Q4_K_M` model files.
//! Q4_0 is the simple legacy format (32 weights, one f16 scale, `y=d·(q−8)`)
//! that plain `q4_0` GGUFs use; it is not a K-quant but shares all the
//! machinery here. Storing the packed blocks and dotting against them
//! directly keeps weights at ~4.5 / ~6.6 bits each in RAM (vs 32 after
//! dequantize-at-load) and reads 7x less weight memory per token, which is
//! what decode is bound by.
//!
//! The native dots run in *integer* arithmetic: activations are quantized to
//! i8 per 32-group once per matmul ([`crate::quantize_acts`]), so the inner
//! loop is int8×int8 multiplies, and the affine terms (Q4_0's `−8`, Q4_K's
//! `−dmin·m`) collapse to one multiply per group via the quantized groups'
//! integer sums: `Σ a·(d·sc·q − dmin·m) ≈ sa·(d·sc·Σ qa·qw − dmin·m·Σ qa)`.
//!
//! Everything here is safe scalar Rust except the AVX2 dots, which live in
//! `simd_x86` with the other intrinsics code.

use crate::f16_to_f32;
use crate::matmul::{quantize_acts, run_rows, QuantActs};

/// K-quant super-block width.
pub const QK_K: usize = 256;
/// Q4_K super-block: f16 d + f16 dmin + 12 bytes packed 6-bit scales/mins +
/// 128 bytes of 4-bit quants.
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;
/// Q6_K super-block: 128 B low nibbles + 64 B high 2-bits + 16 i8 scales + f16 d.
pub const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
/// Q4_0 block width (32 weights per block, unlike the 256-wide K-quants).
pub const Q4_0_BLOCK: usize = 32;
/// Q4_0 block: f16 d + 16 bytes of 4-bit quants (low nibbles are elements
/// 0..16, high nibbles 16..32); `y = d · (q − 8)`.
pub const Q4_0_BLOCK_BYTES: usize = 2 + Q4_0_BLOCK / 2;

/// Dequantize one Q4_0 block (18 bytes -> 32 f32).
pub fn dequant_q40_block(b: &[u8], out: &mut Vec<f32>) {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    for j in 0..16 {
        out.push(d * ((b[2 + j] & 0xF) as f32 - 8.0));
    }
    for j in 0..16 {
        out.push(d * ((b[2 + j] >> 4) as f32 - 8.0));
    }
}

/// Unpack the 6-bit (scale, min) pair `j` (0..8) from a Q4_K super-block's
/// 12-byte packed `scales` field — llama.cpp's `get_scale_min_k4`.
#[inline]
pub fn q4k_scale_min(j: usize, s: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((s[j] & 63) as f32, (s[j + 4] & 63) as f32)
    } else {
        (
            ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4)) as f32,
            ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as f32,
        )
    }
}

/// Dequantize one Q4_K super-block (144 bytes -> 256 f32), llama.cpp
/// `dequantize_row_q4_K`: eight 32-wide groups, each `d·sc·q − dmin·m`;
/// groups (2p, 2p+1) share 32 bytes as (low nibbles, high nibbles).
pub fn dequant_q4k_block(b: &[u8], out: &mut Vec<f32>) {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
    let scales = &b[4..16];
    let qs = &b[16..16 + QK_K / 2];
    for pair in 0..4 {
        let q = &qs[pair * 32..(pair + 1) * 32];
        let (sc1, m1) = q4k_scale_min(pair * 2, scales);
        let (sc2, m2) = q4k_scale_min(pair * 2 + 1, scales);
        let (d1, min1) = (d * sc1, dmin * m1);
        let (d2, min2) = (d * sc2, dmin * m2);
        for &byte in q {
            out.push(d1 * (byte & 0xF) as f32 - min1);
        }
        for &byte in q {
            out.push(d2 * (byte >> 4) as f32 - min2);
        }
    }
}

/// Dequantize one Q6_K super-block (210 bytes -> 256 f32), llama.cpp
/// `dequantize_row_q6_K`: 6-bit values (low nibble + 2 high bits) minus 32,
/// times `d * scales[sub-block]`, in interleaved 32-wide output groups.
pub fn dequant_q6k_block(b: &[u8], out: &mut Vec<f32>) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
    let scale = |i: usize| d * sc[i] as i8 as f32;
    let start = out.len();
    out.resize(start + QK_K, 0.0);
    let y = &mut out[start..start + QK_K];
    for half in 0..2 {
        let (ql, qh, s0, yo) = (&ql[half * 64..], &qh[half * 32..], half * 8, half * 128);
        for l in 0..32 {
            let is = s0 + l / 16;
            let q1 = (((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32) as f32;
            let q2 = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32) as f32;
            let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32) as f32;
            let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32) as f32;
            y[yo + l] = scale(is) * q1;
            y[yo + l + 32] = scale(is + 2) * q2;
            y[yo + l + 64] = scale(is + 4) * q3;
            y[yo + l + 96] = scale(is + 6) * q4;
        }
    }
}

/// A row-major matrix stored as packed quantized blocks.
#[derive(Clone)]
pub struct KQuantMatrix {
    pub rows: usize,
    pub cols: usize,
    kind: KQuantKind,
    /// `rows * cols/block_values * block_bytes`, rows contiguous.
    blocks: Vec<u8>,
    /// Block scales hoisted to f32 at load (`scales_per_block` per block,
    /// same order as `blocks`). The packed blocks store them as f16, and a
    /// matmul touches one per 32-256 weights — decoding f16 in the hot dot
    /// loop measurably dominated decode (millions of conversions per
    /// token), so it happens exactly once, here.
    dscales: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantKind {
    Q40,
    Q4K,
    Q6K,
}

impl KQuantKind {
    pub fn block_bytes(self) -> usize {
        match self {
            KQuantKind::Q40 => Q4_0_BLOCK_BYTES,
            KQuantKind::Q4K => Q4_K_BLOCK_BYTES,
            KQuantKind::Q6K => Q6_K_BLOCK_BYTES,
        }
    }

    /// Weights per block: 32 for Q4_0, 256 for the K-quant super-blocks.
    pub fn block_values(self) -> usize {
        match self {
            KQuantKind::Q40 => Q4_0_BLOCK,
            KQuantKind::Q4K | KQuantKind::Q6K => QK_K,
        }
    }

    /// f16 scale fields per block (hoisted to `dscales`): Q4_K carries
    /// `(d, dmin)`, the others a single `d`.
    fn scales_per_block(self) -> usize {
        match self {
            KQuantKind::Q4K => 2,
            KQuantKind::Q40 | KQuantKind::Q6K => 1,
        }
    }
}

impl KQuantMatrix {
    /// Wrap raw GGUF tensor bytes. Panics on size mismatch (loader bug).
    pub fn from_raw(rows: usize, cols: usize, kind: KQuantKind, blocks: Vec<u8>) -> Self {
        let bv = kind.block_values();
        assert_eq!(cols % bv, 0, "cols must be a multiple of {bv}");
        assert_eq!(blocks.len(), rows * cols / bv * kind.block_bytes());
        let f16 = |b: &[u8], off: usize| f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]));
        let mut dscales = Vec::with_capacity(blocks.len() / kind.block_bytes() * kind.scales_per_block());
        for b in blocks.chunks_exact(kind.block_bytes()) {
            match kind {
                KQuantKind::Q40 => dscales.push(f16(b, 0)),
                KQuantKind::Q4K => {
                    dscales.push(f16(b, 0));
                    dscales.push(f16(b, 2));
                }
                KQuantKind::Q6K => dscales.push(f16(b, 208)),
            }
        }
        Self { rows, cols, kind, blocks, dscales }
    }

    pub fn kind(&self) -> KQuantKind {
        self.kind
    }

    pub fn stored_bytes(&self) -> usize {
        self.blocks.len()
    }

    fn row(&self, r: usize) -> &[u8] {
        let per_row = self.cols / self.kind.block_values() * self.kind.block_bytes();
        &self.blocks[r * per_row..(r + 1) * per_row]
    }

    /// The hoisted f32 scales of row `r`'s blocks.
    fn row_dscales(&self, r: usize) -> &[f32] {
        let per_row = self.cols / self.kind.block_values() * self.kind.scales_per_block();
        &self.dscales[r * per_row..(r + 1) * per_row]
    }

    /// Integer dot of packed weight row `r` against quantized activations.
    #[inline]
    pub fn row_dot(&self, acts: &QuantActs, r: usize) -> f32 {
        kquant_row_dot(self, acts, r)
    }

    /// Dequantize a single row (token-embedding lookups).
    pub fn dequantize_row(&self, r: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.cols);
        self.dequant_blocks(self.row(r), &mut out);
        out
    }

    /// Full dequantization (tests / small tensors).
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        self.dequant_blocks(&self.blocks, &mut out);
        out
    }

    fn dequant_blocks(&self, blocks: &[u8], out: &mut Vec<f32>) {
        for b in blocks.chunks_exact(self.kind.block_bytes()) {
            match self.kind {
                KQuantKind::Q40 => dequant_q40_block(b, out),
                KQuantKind::Q4K => dequant_q4k_block(b, out),
                KQuantKind::Q6K => dequant_q6k_block(b, out),
            }
        }
    }
}

/// Scalar integer Q4_0 row dot against quantized activations:
/// `Σ a·d(q−8) ≈ d·sa·(Σ qa·q − 8·Σ qa)`. `ds` holds the row's hoisted
/// f32 block scales — decoding the packed f16 here cost millions of scalar
/// conversions per token before they were hoisted to load time.
#[inline]
fn dot_q40_q8_scalar(acts: &QuantActs, row: &[u8], ds: &[f32]) -> f32 {
    let mut total = 0f32;
    for (i, b) in row.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let qa = &acts.q[i * 32..(i + 1) * 32];
        let mut s = 0i32;
        for l in 0..16 {
            s += qa[l] as i32 * (b[2 + l] & 0xF) as i32
                + qa[16 + l] as i32 * (b[2 + l] >> 4) as i32;
        }
        total += ds[i] * acts.scales[i] * (s - 8 * acts.sums[i]) as f32;
    }
    total
}

/// Scalar integer Q4_K row dot: per 32-group,
/// `Σ a·(d·sc·q − dmin·m) ≈ sa·(d·sc·Σ qa·q − dmin·m·Σ qa)`.
#[inline]
fn dot_q4k_q8_scalar(acts: &QuantActs, row: &[u8], ds: &[f32]) -> f32 {
    let mut total = 0f32;
    for (b_idx, b) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let (d, dmin) = (ds[b_idx * 2], ds[b_idx * 2 + 1]);
        let scales = &b[4..16];
        let qs = &b[16..144];
        for pair in 0..4 {
            let q = &qs[pair * 32..(pair + 1) * 32];
            let g_lo = b_idx * 8 + pair * 2;
            let g_hi = g_lo + 1;
            let qa_lo = &acts.q[g_lo * 32..(g_lo + 1) * 32];
            let qa_hi = &acts.q[g_hi * 32..(g_hi + 1) * 32];
            let (sc1, m1) = q4k_scale_min(pair * 2, scales);
            let (sc2, m2) = q4k_scale_min(pair * 2 + 1, scales);
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for l in 0..32 {
                s1 += qa_lo[l] as i32 * (q[l] & 0xF) as i32;
                s2 += qa_hi[l] as i32 * (q[l] >> 4) as i32;
            }
            total += acts.scales[g_lo] * (d * sc1 * s1 as f32 - dmin * m1 * acts.sums[g_lo] as f32);
            total += acts.scales[g_hi] * (d * sc2 * s2 as f32 - dmin * m2 * acts.sums[g_hi] as f32);
        }
    }
    total
}

/// Scalar integer Q6_K row dot (no affine trick needed — Q6_K has no mins;
/// `q−32` fits i8 so the products accumulate in integers directly).
#[inline]
fn dot_q6k_q8_scalar(acts: &QuantActs, row: &[u8], ds: &[f32]) -> f32 {
    let mut total = 0f32;
    for (b_idx, b) in row.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        let d = ds[b_idx];
        let qa_blk = &acts.q[b_idx * QK_K..(b_idx + 1) * QK_K];
        for half in 0..2 {
            let (ql, qh, s0, ao) = (&ql[half * 64..], &qh[half * 32..], half * 8, half * 128);
            let mut sums = [0i32; 8]; // per (group, sub) partial dots
            for l in 0..32 {
                let sub = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
                sums[sub] += qa_blk[ao + l] as i32 * q1;
                sums[2 + sub] += qa_blk[ao + l + 32] as i32 * q2;
                sums[4 + sub] += qa_blk[ao + l + 64] as i32 * q3;
                sums[6 + sub] += qa_blk[ao + l + 96] as i32 * q4;
            }
            for (g, chunk) in sums.chunks_exact(2).enumerate() {
                // Activation 32-group covering elements ao + g*32 of the block.
                let sa = acts.scales[b_idx * 8 + half * 4 + g];
                total += d * sa * sc[s0 + g * 2] as i8 as f32 * chunk[0] as f32;
                total += d * sa * sc[s0 + g * 2 + 1] as i8 as f32 * chunk[1] as f32;
            }
        }
    }
    total
}

/// Packed-quantized weight × f32 activation matmul. Activations are
/// quantized to i8 once per row (integer inner loops); output
/// `[n_tokens, n_rows]`.
pub fn matmul_kquant_f32(a: &[f32], m: &KQuantMatrix, n_tokens: usize) -> Vec<f32> {
    let (k, n_rows) = (m.cols, m.rows);
    assert_eq!(a.len(), n_tokens * k);
    let mut out = vec![0f32; n_tokens * n_rows];

    for t in 0..n_tokens {
        let acts = quantize_acts(&a[t * k..(t + 1) * k]);
        run_rows(&mut out[t * n_rows..(t + 1) * n_rows], k, |r| kquant_row_dot(m, &acts, r));
    }
    out
}

/// One row of a packed-quantized matmul: integer dot against `acts`,
/// dispatched by block format and SIMD availability.
#[inline]
pub(crate) fn kquant_row_dot(m: &KQuantMatrix, acts: &QuantActs, r: usize) -> f32 {
    let row = m.row(r);
    let ds = m.row_dscales(r);
    match m.kind {
        KQuantKind::Q40 => {
            if crate::simd_x86::available_vnni() {
                // SAFETY: available_vnni() verified AVX2+FMA+AVX-VNNI.
                unsafe { crate::simd_x86::dot_q40_q8_vnni(acts, row, ds) }
            } else if crate::simd_x86::available() {
                // SAFETY: available() verified AVX2+FMA on this CPU.
                unsafe { crate::simd_x86::dot_q40_q8(acts, row, ds) }
            } else {
                dot_q40_q8_scalar(acts, row, ds)
            }
        }
        KQuantKind::Q4K => {
            if crate::simd_x86::available() {
                // SAFETY: available() verified AVX2+FMA on this CPU.
                unsafe { crate::simd_x86::dot_q4k_q8(acts, row, ds) }
            } else {
                dot_q4k_q8_scalar(acts, row, ds)
            }
        }
        KQuantKind::Q6K => dot_q6k_q8_scalar(acts, row, ds),
    }
}

/// Prefill matmul over a batch of quantized activation rows, weight-row
/// outer (each packed row streams once for all tokens). See
/// [`crate::matmul_q8_prefill`]. Output `[n_tokens, rows]`.
pub fn matmul_kquant_prefill(acts: &[QuantActs], m: &KQuantMatrix) -> Vec<f32> {
    crate::matmul::run_rows_multi(m.rows, acts.len(), m.cols, |r, t| kquant_row_dot(m, &acts[t], r))
}

/// Team version of [`matmul_kquant_f32`] for one activation row: fills
/// `out[..rows]` across the team, no dispatch — barriers only.
pub fn matmul_kquant_team(team: &crate::team::Team, acts: &QuantActs, m: &KQuantMatrix, out: &crate::team::TeamCell<Vec<f32>>) {
    crate::team::team_fill_rows_weighted(team, out, m.rows, m.cols, |r| kquant_row_dot(m, acts, r));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul_f32;

    /// Deterministic pseudo-random bytes: any byte string is a valid K-quant
    /// block, so this exercises every bit path.
    fn pseudo_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                (s.wrapping_mul(0x2545F4914F6CDD1D) >> 56) as u8
            })
            .collect()
    }

    fn check_matches_dequant_reference(kind: KQuantKind, seed: u64) {
        let rows = 5;
        let cols = QK_K * 2;
        let bb = kind.block_bytes();
        let mut bytes = pseudo_bytes(rows * cols / kind.block_values() * bb, seed);
        // Random bytes are a valid block EXCEPT the f16 scale fields, which
        // can decode to NaN/Inf; overwrite those with finite values.
        for b in bytes.chunks_exact_mut(bb) {
            match kind {
                KQuantKind::Q40 => {
                    b[0..2].copy_from_slice(&0x2e66u16.to_le_bytes()); // d ~ 0.1
                }
                KQuantKind::Q4K => {
                    b[0..2].copy_from_slice(&0x2e66u16.to_le_bytes()); // d ~ 0.1
                    b[2..4].copy_from_slice(&0x2a66u16.to_le_bytes()); // dmin ~ 0.05
                }
                KQuantKind::Q6K => {
                    b[208..210].copy_from_slice(&0x2e66u16.to_le_bytes());
                }
            }
        }
        let m = KQuantMatrix::from_raw(rows, cols, kind, bytes);
        let n_tokens = 2;
        let a: Vec<f32> = (0..n_tokens * cols).map(|i| (i as f32 * 0.013).sin()).collect();

        let native = matmul_kquant_f32(&a, &m, n_tokens);
        // Reference: dequantized weights × *dequantized quantized* activations
        // — the integer kernels see i8 activations, so the oracle must too;
        // any remaining difference is a kernel bug, not quantization error.
        let mut a_dq = Vec::with_capacity(a.len());
        for row in a.chunks_exact(cols) {
            let acts = quantize_acts(row);
            a_dq.extend(
                acts.q.iter().enumerate().map(|(i, &q)| q as f32 * acts.scales[i / 32]),
            );
        }
        let dense = matmul_f32(&a_dq, &m.dequantize(), n_tokens, cols, rows);

        for (i, (n, d)) in native.iter().zip(&dense).enumerate() {
            // Random blocks can have large f16 scales; compare relatively.
            // (Integer path sums exactly; slack is f32 accumulation order.)
            let tol = 1e-3 * d.abs().max(1.0);
            assert!((n - d).abs() < tol, "{kind:?}[{i}]: native {n} vs dequant {d}");
        }
    }

    #[test]
    fn q40_native_matmul_matches_dequant_reference() {
        for seed in [3u64, 77, 2024] {
            check_matches_dequant_reference(KQuantKind::Q40, seed);
        }
    }

    #[test]
    fn q40_row_dequant_matches_full_dequant() {
        let rows = 3;
        let cols = 64;
        let mut bytes = pseudo_bytes(rows * cols / 32 * Q4_0_BLOCK_BYTES, 9);
        for b in bytes.chunks_exact_mut(Q4_0_BLOCK_BYTES) {
            b[0..2].copy_from_slice(&0x2e66u16.to_le_bytes());
        }
        let m = KQuantMatrix::from_raw(rows, cols, KQuantKind::Q40, bytes);
        let full = m.dequantize();
        for r in 0..rows {
            assert_eq!(m.dequantize_row(r), full[r * cols..(r + 1) * cols]);
        }
    }

    #[test]
    fn q4k_native_matmul_matches_dequant_reference() {
        for seed in [1u64, 42, 12345] {
            check_matches_dequant_reference(KQuantKind::Q4K, seed);
        }
    }

    #[test]
    fn q6k_native_matmul_matches_dequant_reference() {
        for seed in [7u64, 99, 54321] {
            check_matches_dequant_reference(KQuantKind::Q6K, seed);
        }
    }

    #[test]
    fn integer_simd_dots_match_scalar() {
        if !crate::simd_x86::available() {
            eprintln!("skipping: no AVX2+FMA on this CPU");
            return;
        }
        for kind in [KQuantKind::Q40, KQuantKind::Q4K] {
            let cols = QK_K * 2;
            let bb = kind.block_bytes();
            let mut row = pseudo_bytes(cols / kind.block_values() * bb, 31 + bb as u64);
            for b in row.chunks_exact_mut(bb) {
                b[0..2].copy_from_slice(&0x2e66u16.to_le_bytes());
                if kind == KQuantKind::Q4K {
                    b[2..4].copy_from_slice(&0x2a66u16.to_le_bytes());
                }
            }
            let a: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.021).cos()).collect();
            let acts = quantize_acts(&a);
            // Hoisted f32 scales, exactly as from_raw builds them.
            let m = KQuantMatrix::from_raw(1, cols, kind, row.clone());
            let ds = m.row_dscales(0);
            let (scalar, simd) = match kind {
                // SAFETY: available() checked above.
                KQuantKind::Q40 => {
                    (dot_q40_q8_scalar(&acts, &row, ds), unsafe {
                        crate::simd_x86::dot_q40_q8(&acts, &row, ds)
                    })
                }
                KQuantKind::Q4K => {
                    (dot_q4k_q8_scalar(&acts, &row, ds), unsafe {
                        crate::simd_x86::dot_q4k_q8(&acts, &row, ds)
                    })
                }
                KQuantKind::Q6K => unreachable!(),
            };
            let tol = 1e-3 * scalar.abs().max(1.0);
            assert!((scalar - simd).abs() < tol, "{kind:?}: {scalar} vs {simd}");

            if kind == KQuantKind::Q40 && crate::simd_x86::available_vnni() {
                // SAFETY: available_vnni() checked.
                let vnni = unsafe { crate::simd_x86::dot_q40_q8_vnni(&acts, &row, ds) };
                assert!((scalar - vnni).abs() < tol, "vnni: {scalar} vs {vnni}");
            }
        }
    }

    #[test]
    fn stored_bytes_reflect_the_compression() {
        let rows = 4;
        let cols = QK_K;
        let m = KQuantMatrix::from_raw(rows, cols, KQuantKind::Q4K, vec![0; rows * Q4_K_BLOCK_BYTES]);
        // 144 bytes per 256 weights = 4.5 bits/weight vs 32 for f32.
        assert_eq!(m.stored_bytes(), rows * 144);
        assert!(m.stored_bytes() * 7 < rows * cols * 4);
    }
}
