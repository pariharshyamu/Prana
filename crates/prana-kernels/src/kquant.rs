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
//! The native dot exploits the affine structure of Q4_K: within a 32-group,
//! `y = d·sc·q − dmin·m`, so `Σ a·y = d·sc·(Σ a·q) − dmin·m·(Σ a)` — the
//! activation group sums `Σ a` are computed once per row-batch and the `−m`
//! term costs one multiply per group instead of 32.
//!
//! Everything here is safe scalar Rust except the AVX2 Q4_K dot, which lives
//! in `simd_x86` with the other intrinsics code.

use crate::f16_to_f32;
use crate::pool;
use std::sync::Mutex;

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
}

impl KQuantMatrix {
    /// Wrap raw GGUF tensor bytes. Panics on size mismatch (loader bug).
    pub fn from_raw(rows: usize, cols: usize, kind: KQuantKind, blocks: Vec<u8>) -> Self {
        let bv = kind.block_values();
        assert_eq!(cols % bv, 0, "cols must be a multiple of {bv}");
        assert_eq!(blocks.len(), rows * cols / bv * kind.block_bytes());
        Self { rows, cols, kind, blocks }
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

/// Scalar Q4_0 row dot. `asums[g]` must hold `Σ a[g*32..g*32+32]`:
/// `Σ a·d(q−8) = d·(Σ a·q − 8·Σ a)`, one multiply per block for the −8 term.
#[inline]
fn dot_q40_scalar(a: &[f32], row: &[u8], asums: &[f32]) -> f32 {
    let mut total = 0f32;
    for (i, b) in row.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let a_lo = &a[i * 32..i * 32 + 16];
        let a_hi = &a[i * 32 + 16..i * 32 + 32];
        let mut s = 0f32;
        for l in 0..16 {
            s += a_lo[l] * (b[2 + l] & 0xF) as f32 + a_hi[l] * (b[2 + l] >> 4) as f32;
        }
        total += d * (s - 8.0 * asums[i]);
    }
    total
}

/// Scalar Q4_K row dot. `asums[g]` must hold `Σ a[g*32..g*32+32]`.
#[inline]
fn dot_q4k_scalar(a: &[f32], row: &[u8], asums: &[f32]) -> f32 {
    let mut total = 0f32;
    for (b_idx, b) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let scales = &b[4..16];
        let qs = &b[16..144];
        let a_blk = &a[b_idx * QK_K..(b_idx + 1) * QK_K];
        let s_blk = &asums[b_idx * 8..(b_idx + 1) * 8];
        for pair in 0..4 {
            let q = &qs[pair * 32..(pair + 1) * 32];
            let a_lo = &a_blk[pair * 64..pair * 64 + 32];
            let a_hi = &a_blk[pair * 64 + 32..pair * 64 + 64];
            let (sc1, m1) = q4k_scale_min(pair * 2, scales);
            let (sc2, m2) = q4k_scale_min(pair * 2 + 1, scales);
            let mut s1 = 0f32;
            let mut s2 = 0f32;
            for l in 0..32 {
                s1 += a_lo[l] * (q[l] & 0xF) as f32;
                s2 += a_hi[l] * (q[l] >> 4) as f32;
            }
            total += d * sc1 * s1 - dmin * m1 * s_blk[pair * 2];
            total += d * sc2 * s2 - dmin * m2 * s_blk[pair * 2 + 1];
        }
    }
    total
}

/// Scalar Q6_K row dot (no affine trick needed — Q6_K has no mins).
#[inline]
fn dot_q6k_scalar(a: &[f32], row: &[u8]) -> f32 {
    let mut total = 0f32;
    for (b_idx, b) in row.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
        let a_blk = &a[b_idx * QK_K..(b_idx + 1) * QK_K];
        for half in 0..2 {
            let (ql, qh, s0, ao) = (&ql[half * 64..], &qh[half * 32..], half * 8, half * 128);
            let mut sums = [0f32; 8]; // per (group, sub) partial dots
            for l in 0..32 {
                let sub = l / 16;
                let q1 = (((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32) as f32;
                let q2 = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32) as f32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32) as f32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32) as f32;
                sums[sub] += a_blk[ao + l] * q1;
                sums[2 + sub] += a_blk[ao + l + 32] * q2;
                sums[4 + sub] += a_blk[ao + l + 64] * q3;
                sums[6 + sub] += a_blk[ao + l + 96] * q4;
            }
            for (g, chunk) in sums.chunks_exact(2).enumerate() {
                total += d * sc[s0 + g * 2] as i8 as f32 * chunk[0];
                total += d * sc[s0 + g * 2 + 1] as i8 as f32 * chunk[1];
            }
        }
    }
    total
}

/// Per-32-group activation sums, shared by every row of a Q4_K matmul.
fn group_sums(a: &[f32]) -> Vec<f32> {
    a.chunks_exact(32).map(|c| c.iter().sum()).collect()
}

/// K-quant weight × f32 activation matmul. Output `[n_tokens, n_rows]`.
pub fn matmul_kquant_f32(a: &[f32], m: &KQuantMatrix, n_tokens: usize) -> Vec<f32> {
    let (k, n_rows) = (m.cols, m.rows);
    assert_eq!(a.len(), n_tokens * k);
    let mut out = vec![0f32; n_tokens * n_rows];

    for t in 0..n_tokens {
        let a_row = &a[t * k..(t + 1) * k];
        let asums = match m.kind {
            KQuantKind::Q40 | KQuantKind::Q4K => group_sums(a_row),
            KQuantKind::Q6K => Vec::new(),
        };
        let out_t = &mut out[t * n_rows..(t + 1) * n_rows];

        let dot = |r: usize| -> f32 {
            let row = m.row(r);
            match m.kind {
                KQuantKind::Q40 => {
                    if crate::simd_x86::available() {
                        // SAFETY: available() verified AVX2+FMA on this CPU.
                        unsafe { crate::simd_x86::dot_q40(a_row, row, &asums) }
                    } else {
                        dot_q40_scalar(a_row, row, &asums)
                    }
                }
                KQuantKind::Q4K => {
                    if crate::simd_x86::available() {
                        // SAFETY: available() verified AVX2+FMA on this CPU.
                        unsafe { crate::simd_x86::dot_q4k(a_row, row, &asums) }
                    } else {
                        dot_q4k_scalar(a_row, row, &asums)
                    }
                }
                KQuantKind::Q6K => dot_q6k_scalar(a_row, row),
            }
        };

        // ~4.5 bits/weight still means real work per row; parallelize rows on
        // the pool for all but tiny matrices (same policy as run_matmul).
        let pool = pool::global();
        if k * n_rows < 32 * 1024 || pool.threads == 1 {
            for (r, o) in out_t.iter_mut().enumerate() {
                *o = dot(r);
            }
        } else {
            let row_chunk = n_rows.div_ceil(pool.threads).max(1);
            let chunks: Vec<Mutex<(usize, &mut [f32])>> = out_t
                .chunks_mut(row_chunk)
                .enumerate()
                .map(|(i, slot)| Mutex::new((i * row_chunk, slot)))
                .collect();
            pool.run(chunks.len(), &|ci| {
                let mut guard = chunks[ci].lock().unwrap();
                let (r0, slot) = &mut *guard;
                for (i, o) in slot.iter_mut().enumerate() {
                    *o = dot(*r0 + i);
                }
            });
        }
    }
    out
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
        let dense = matmul_f32(&a, &m.dequantize(), n_tokens, cols, rows);

        for (i, (n, d)) in native.iter().zip(&dense).enumerate() {
            // Random blocks can have large f16 scales; compare relatively.
            let tol = 1e-4 * d.abs().max(1.0);
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
    fn stored_bytes_reflect_the_compression() {
        let rows = 4;
        let cols = QK_K;
        let m = KQuantMatrix::from_raw(rows, cols, KQuantKind::Q4K, vec![0; rows * Q4_K_BLOCK_BYTES]);
        // 144 bytes per 256 weights = 4.5 bits/weight vs 32 for f32.
        assert_eq!(m.stored_bytes(), rows * 144);
        assert!(m.stored_bytes() * 7 < rows * cols * 4);
    }
}
