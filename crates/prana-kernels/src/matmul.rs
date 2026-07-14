//! Matrix multiply and Q8 block quantization — the two kernels that dominate
//! LLM inference cost (weight-times-activation in every projection).
//!
//! `QuantMatrix` stores weights as int8 blocks with a per-block fp32 scale,
//! the same idea as a Cactus CQ tensor's group scales (`group_size`,
//! per-group scale). Decode-time matmul then multiplies a quantized weight
//! matrix by fp32 activations, dequantizing on the fly.

use std::sync::Mutex;

/// Block size for Q8 quantization: 32 weights share one fp32 scale.
/// Matches `KV_QUANT_GROUP_SIZE = 32` in `cactus-kernels/cactus_kernels.h`.
pub const Q8_BLOCK: usize = 32;

/// A row-major `rows x cols` weight matrix stored as Q8 blocks.
///
/// Each row is quantized independently in `Q8_BLOCK`-wide groups; `scales`
/// holds one f32 per group. `cols` must be a multiple of `Q8_BLOCK` for the
/// prototype (production would pad the tail).
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Length `rows * cols` int8 weights.
    q: Vec<i8>,
    /// Length `rows * (cols / Q8_BLOCK)` block scales.
    scales: Vec<f32>,
}

impl QuantMatrix {
    /// Assemble from already-quantized parts (e.g. a GGUF Q8_0 tensor whose
    /// blocks map 1:1 onto this layout). Panics on inconsistent lengths —
    /// a loader bug, not a runtime input.
    pub fn from_parts(rows: usize, cols: usize, q: Vec<i8>, scales: Vec<f32>) -> Self {
        assert_eq!(cols % Q8_BLOCK, 0, "cols must be a multiple of {Q8_BLOCK}");
        assert_eq!(q.len(), rows * cols);
        assert_eq!(scales.len(), rows * (cols / Q8_BLOCK));
        Self { rows, cols, q, scales }
    }

    fn groups_per_row(&self) -> usize {
        self.cols / Q8_BLOCK
    }

    /// Integer dot of weight row `r` against quantized activations.
    #[inline]
    pub fn row_dot(&self, acts: &QuantActs, r: usize) -> f32 {
        let k = self.cols;
        let gpr = self.groups_per_row();
        dot_q8_q8(acts, &self.q[r * k..(r + 1) * k], &self.scales[r * gpr..(r + 1) * gpr])
    }

    /// Bytes of weight storage — the whole point of quantizing.
    pub fn stored_bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }
}

/// Quantize a dense f32 weight matrix to Q8. Panics if `cols` is not a multiple
/// of `Q8_BLOCK` (a build-time layout invariant, not runtime input).
pub fn quantize_q8(rows: usize, cols: usize, w: &[f32]) -> QuantMatrix {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(cols % Q8_BLOCK, 0, "cols must be a multiple of {Q8_BLOCK}");

    let groups_per_row = cols / Q8_BLOCK;
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows * groups_per_row];

    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * cols + g * Q8_BLOCK;
            let block = &w[base..base + Q8_BLOCK];
            let amax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = 1.0 / scale;
            for (i, &x) in block.iter().enumerate() {
                let v = (x * inv).round().clamp(-127.0, 127.0);
                q[base + i] = v as i8;
            }
            scales[r * groups_per_row + g] = scale;
        }
    }
    QuantMatrix { rows, cols, q, scales }
}

/// One activation row quantized to i8 in `Q8_BLOCK`-wide groups.
///
/// This is the llama.cpp trick that makes quantized-weight matmuls fast on
/// CPU: with *both* sides in int8, the inner loop becomes integer multiplies
/// (32 MACs per AVX2 `maddubs`+`madd` pair vs 8 f32 FMA lanes), and the
/// per-group `sums` let affine corrections (Q4_0's `−8`, Q4_K's `−dmin·m`)
/// cost one multiply per group. Quantizing the activations once per matmul
/// adds ~0.1% RMS error — the same order as the weight quantization already
/// present — in exchange for ~4x the dot-product throughput per core.
pub struct QuantActs {
    pub q: Vec<i8>,
    /// One f32 scale per 32-group.
    pub scales: Vec<f32>,
    /// Integer sum of each group's quants (for affine correction terms).
    pub sums: Vec<i32>,
}

/// Quantize one activation row (length must be a multiple of `Q8_BLOCK`,
/// which every model dimension we load already is).
pub fn quantize_acts(a: &[f32]) -> QuantActs {
    assert!(a.len().is_multiple_of(Q8_BLOCK));
    let groups = a.len() / Q8_BLOCK;
    let mut q = vec![0i8; a.len()];
    let mut scales = vec![0f32; groups];
    let mut sums = vec![0i32; groups];
    for g in 0..groups {
        let base = g * Q8_BLOCK;
        let block = &a[base..base + Q8_BLOCK];
        let amax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv = 1.0 / scale;
        let mut sum = 0i32;
        for (i, &x) in block.iter().enumerate() {
            let v = (x * inv).round().clamp(-127.0, 127.0) as i32;
            q[base + i] = v as i8;
            sum += v;
        }
        scales[g] = scale;
        sums[g] = sum;
    }
    QuantActs { q, scales, sums }
}

/// Reconstruct a dense f32 matrix from Q8 (for error measurement / tests).
pub fn dequantize_q8(m: &QuantMatrix) -> Vec<f32> {
    let groups_per_row = m.groups_per_row();
    let mut out = vec![0f32; m.rows * m.cols];
    for r in 0..m.rows {
        for g in 0..groups_per_row {
            let scale = m.scales[r * groups_per_row + g];
            let base = r * m.cols + g * Q8_BLOCK;
            for i in 0..Q8_BLOCK {
                out[base + i] = m.q[base + i] as f32 * scale;
            }
        }
    }
    out
}

/// Dense f32 matmul: `out[t, r] = sum_k a[t, k] * w[r, k]`.
///
/// `w` is `n_rows x k` (already transposed for the projection), so each output
/// is a dot product of an activation row with a weight row — the cache-friendly
/// layout inference engines actually use. Output is `[n_tokens, n_rows]`.
pub fn matmul_f32(a: &[f32], w: &[f32], n_tokens: usize, k: usize, n_rows: usize) -> Vec<f32> {
    assert_eq!(a.len(), n_tokens * k);
    assert_eq!(w.len(), n_rows * k);
    let mut out = vec![0f32; n_tokens * n_rows];
    let dot = |a_row: &[f32], r: usize| dot_f32(a_row, &w[r * k..(r + 1) * k]);
    run_matmul(&mut out, a, n_tokens, k, n_rows, dot);
    out
}

/// Q8 weight × f32 activation matmul. Activations are quantized to i8 once
/// per row, then every weight row is dotted in integer arithmetic.
/// Output is `[n_tokens, n_rows]`.
pub fn matmul_q8_f32(a: &[f32], w: &QuantMatrix, n_tokens: usize) -> Vec<f32> {
    let k = w.cols;
    let n_rows = w.rows;
    assert_eq!(a.len(), n_tokens * k);
    let groups_per_row = w.groups_per_row();
    let mut out = vec![0f32; n_tokens * n_rows];
    for t in 0..n_tokens {
        let acts = quantize_acts(&a[t * k..(t + 1) * k]);
        let dot = |r: usize| {
            let q_row = &w.q[r * k..(r + 1) * k];
            let s_row = &w.scales[r * groups_per_row..(r + 1) * groups_per_row];
            dot_q8_q8(&acts, q_row, s_row)
        };
        run_rows(&mut out[t * n_rows..(t + 1) * n_rows], k, dot);
    }
    out
}

/// Team version of [`matmul_q8_f32`] for one already-quantized activation
/// row (barriers only, no pool dispatch).
pub fn matmul_q8_team(team: &crate::team::Team, acts: &QuantActs, w: &QuantMatrix, out: &crate::team::TeamCell<Vec<f32>>) {
    let k = w.cols;
    let groups_per_row = w.groups_per_row();
    crate::team::team_fill_rows_weighted(team, out, w.rows, k, |r| {
        let q_row = &w.q[r * k..(r + 1) * k];
        let s_row = &w.scales[r * groups_per_row..(r + 1) * groups_per_row];
        dot_q8_q8(acts, q_row, s_row)
    });
}

/// Team version of the dense matmul for one activation row.
pub fn matmul_f32_team(team: &crate::team::Team, a: &[f32], w: &[f32], n_rows: usize, out: &crate::team::TeamCell<Vec<f32>>) {
    let k = a.len();
    assert_eq!(w.len(), n_rows * k);
    crate::team::team_fill_rows_weighted(team, out, n_rows, k, |r| dot_f32(a, &w[r * k..(r + 1) * k]));
}

/// Shared parallel driver for both matmul kernels.
///
/// - **Decode** (`n_tokens == 1`): split the `n_rows` outputs across the pool.
/// - **Prefill** (`n_tokens > 1`): give each token's row of outputs to a chunk.
///
/// `dot(a_row, r)` computes one output element. Work is fanned out on the
/// persistent [`crate::pool`], so per-call cost is a wakeup, not a thread
/// spawn — small per-layer projections can afford to parallelize. Disjoint
/// output chunks are handed to workers through per-chunk `Mutex`es (locked
/// exactly once each), keeping this function entirely safe.
fn run_matmul<D>(out: &mut [f32], a: &[f32], n_tokens: usize, k: usize, n_rows: usize, dot: D)
where
    D: Fn(&[f32], usize) -> f32 + Sync,
{
    let pool = crate::pool::global();

    // Below this many MACs even a pool wakeup costs more than the work.
    const MIN_MACS_FOR_THREADS: usize = 32 * 1024;
    if n_tokens * k * n_rows < MIN_MACS_FOR_THREADS || pool.threads == 1 {
        for t in 0..n_tokens {
            let a_row = &a[t * k..(t + 1) * k];
            for r in 0..n_rows {
                out[t * n_rows + r] = dot(a_row, r);
            }
        }
        return;
    }

    if n_tokens == 1 {
        let a_row = &a[0..k];
        let row_chunk = n_rows.div_ceil(pool.threads).max(1);
        let chunks: Vec<Mutex<(usize, &mut [f32])>> = out
            .chunks_mut(row_chunk)
            .enumerate()
            .map(|(i, slot)| Mutex::new((i * row_chunk, slot)))
            .collect();
        pool.run(chunks.len(), &|ci| {
            let mut guard = chunks[ci].lock().unwrap();
            let (r0, slot) = &mut *guard;
            for (i, o) in slot.iter_mut().enumerate() {
                *o = dot(a_row, *r0 + i);
            }
        });
        return;
    }

    // Prefill: one chunk per token, each fills that token's `n_rows` outputs.
    let chunks: Vec<Mutex<(usize, &mut [f32])>> = out
        .chunks_mut(n_rows)
        .enumerate()
        .map(|(t, slot)| Mutex::new((t, slot)))
        .collect();
    pool.run(chunks.len(), &|ci| {
        let mut guard = chunks[ci].lock().unwrap();
        let (t, slot) = &mut *guard;
        let a_row = &a[*t * k..(*t + 1) * k];
        for (r, o) in slot.iter_mut().enumerate() {
            *o = dot(a_row, r);
        }
    });
}

/// Row-parallel driver for one token's outputs: split `out` (one element per
/// weight row, `k` MACs each) across the persistent pool. Shared by every
/// quantized matmul, where per-token activation prep happens before fan-out.
pub(crate) fn run_rows<D>(out: &mut [f32], k: usize, dot: D)
where
    D: Fn(usize) -> f32 + Sync,
{
    let pool = crate::pool::global();
    // Measured on the integer kernels: dispatch+join costs ~20-60µs while a
    // 1M-MAC job runs in ~40µs on one thread — fanning out anything smaller
    // *loses* time. Per token this keeps the small Q/K/V/O projections serial
    // on the hot thread and parallelizes only the FFN and LM-head matmuls.
    const MIN_MACS_FOR_THREADS: usize = 1 << 20;
    if k * out.len() < MIN_MACS_FOR_THREADS || pool.threads == 1 {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot(r);
        }
        return;
    }
    // Several chunks per thread, not one: with work stealing, finer chunks
    // let fast cores absorb the slack when P- and E-cores run at different
    // speeds — equal-size one-per-thread chunks would make every matmul
    // wait for the slowest core.
    let row_chunk = out.len().div_ceil(pool.threads * 4).max(16);
    let chunks: Vec<Mutex<(usize, &mut [f32])>> = out
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

/// Dense dot product: SIMD tier when the CPU supports it, scalar otherwise.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if crate::simd_x86::available() {
        // SAFETY: available() verified AVX2+FMA support on this CPU.
        return unsafe { crate::simd_x86::dot_f32(a, b) };
    }
    dot_f32_scalar(a, b)
}

/// Integer Q8×Q8 dot product: SIMD tier when available, scalar otherwise.
#[inline]
fn dot_q8_q8(acts: &QuantActs, q: &[i8], scales: &[f32]) -> f32 {
    if crate::simd_x86::available() {
        // SAFETY: available() verified AVX2+FMA support on this CPU.
        return unsafe { crate::simd_x86::dot_q8_q8(acts, q, scales) };
    }
    dot_q8_q8_scalar(acts, q, scales)
}

#[inline]
pub(crate) fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 4];
    let mut ai = a.chunks_exact(4);
    let mut bi = b.chunks_exact(4);
    for (x, y) in ai.by_ref().zip(bi.by_ref()) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for (x, y) in ai.remainder().iter().zip(bi.remainder()) {
        sum += x * y;
    }
    sum
}

/// Scalar integer Q8×Q8 dot: `Σ_g sw[g]·sa[g]·Σ_32 qw·qa`.
#[inline]
pub(crate) fn dot_q8_q8_scalar(acts: &QuantActs, q: &[i8], scales: &[f32]) -> f32 {
    let mut sum = 0f32;
    for (g, &sw) in scales.iter().enumerate() {
        let base = g * Q8_BLOCK;
        let mut acc = 0i32;
        for l in 0..Q8_BLOCK {
            acc += acts.q[base + l] as i32 * q[base + l] as i32;
        }
        sum += sw * acts.scales[g] * acc as f32;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_matmul_matches_reference() {
        // a: 2x3, w: 4x3 (n_rows=4, k=3)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![
            1.0, 0.0, 0.0, // row 0 -> picks a[.,0]
            0.0, 1.0, 0.0, // row 1 -> picks a[.,1]
            0.0, 0.0, 1.0, // row 2 -> picks a[.,2]
            1.0, 1.0, 1.0, // row 3 -> sum
        ];
        let out = matmul_f32(&a, &w, 2, 3, 4);
        // token 0: [1,2,3,6]; token 1: [4,5,6,15]
        assert_eq!(out, vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }

    #[test]
    fn q8_roundtrip_is_close() {
        let rows = 8;
        let cols = Q8_BLOCK * 2;
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.013).sin()).collect();
        let qm = quantize_q8(rows, cols, &w);
        let dq = dequantize_q8(&qm);
        let max_err = w.iter().zip(&dq).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // int8 with per-32 block scaling on |x|<=1 data: error well under 1%.
        assert!(max_err < 0.01, "max_err = {max_err}");
        // And it actually saves memory vs dense f32.
        assert!(qm.stored_bytes() < w.len() * 4);
    }

    #[test]
    fn q8_matmul_tracks_f32_matmul() {
        let n_rows = 6;
        let k = Q8_BLOCK * 3;
        let n_tokens = 1;
        let a: Vec<f32> = (0..n_tokens * k).map(|i| ((i as f32) * 0.02).cos()).collect();
        let w: Vec<f32> = (0..n_rows * k).map(|i| ((i as f32) * 0.017).sin()).collect();

        let dense = matmul_f32(&a, &w, n_tokens, k, n_rows);
        let qm = quantize_q8(n_rows, k, &w);
        let quant = matmul_q8_f32(&a, &qm, n_tokens);

        for (d, q) in dense.iter().zip(&quant) {
            assert!((d - q).abs() < 0.05, "dense {d} vs quant {q}");
        }
    }

    #[test]
    fn prefill_and_decode_agree() {
        // Same weights, feed 3 tokens as a batch vs one at a time.
        let n_rows = 5;
        let k = Q8_BLOCK;
        let w: Vec<f32> = (0..n_rows * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let batch: Vec<f32> = (0..3 * k).map(|i| (i as f32 * 0.011).cos()).collect();

        let all = matmul_f32(&batch, &w, 3, k, n_rows);
        for t in 0..3 {
            let one = matmul_f32(&batch[t * k..(t + 1) * k], &w, 1, k, n_rows);
            assert_eq!(&all[t * n_rows..(t + 1) * n_rows], &one[..]);
        }
    }
}
