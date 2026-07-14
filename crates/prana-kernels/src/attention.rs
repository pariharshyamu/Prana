//! Rotary position embedding (RoPE) and scaled-dot-product attention — the
//! remaining core of a transformer block beyond matmul/norm.
//!
//! Cactus implements these in `cactus-kernels/src/norms_rope.cpp` (RoPE) and a
//! fused attention kernel with a Metal fast path. The numerics here are the
//! standard ones (Llama-style `rotate_half` RoPE, causal masked softmax
//! attention with grouped-query support); the point is that the whole thing is
//! expressible in safe, autovectorizable Rust with an obvious seam for a
//! target-gated SIMD/Metal tier behind the same signatures.
//!
//! Layout convention (seq-major, matching a matmul output `[seq, features]`):
//! Q is `[seq_q, n_heads * head_dim]`, K/V are `[seq_kv, n_kv_heads * head_dim]`,
//! all row-major. This is the layout the projection matmuls produce directly.

use crate::norms::softmax;

/// Apply Llama-style `rotate_half` RoPE in place to a `[seq, n_heads * head_dim]`
/// tensor. Position `p` for row index `p` (0-based), pairing dim `j` with
/// `j + head_dim/2`. `head_dim` must be even.
///
/// `theta_base` is the usual RoPE base (10000.0 for most models).
pub fn rope(x: &mut [f32], seq: usize, n_heads: usize, head_dim: usize, theta_base: f32) {
    assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");
    assert_eq!(x.len(), seq * n_heads * head_dim);
    let half = head_dim / 2;
    let row_stride = n_heads * head_dim;

    for p in 0..seq {
        for h in 0..n_heads {
            let base = p * row_stride + h * head_dim;
            for j in 0..half {
                // freq = theta_base^(-2j/head_dim); angle = p * freq.
                let freq = (theta_base).powf(-(2.0 * j as f32) / head_dim as f32);
                let angle = p as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                let x1 = x[base + j];
                let x2 = x[base + j + half];
                x[base + j] = x1 * cos - x2 * sin;
                x[base + j + half] = x1 * sin + x2 * cos;
            }
        }
    }
}

/// Apply interleaved (adjacent-pair) RoPE in place to a single position's
/// activation row `[n_heads * head_dim]`, rotating pairs `(x[i], x[i+1])` —
/// the llama2.c / original-Llama convention, as opposed to [`rope`]'s
/// `rotate_half` pairing. Checkpoints trained with one convention are wrong
/// under the other, so both exist.
///
/// `pos` is the absolute token position; frequency depends on `i % head_dim`
/// so every head sees the same rotation schedule.
pub fn rope_interleaved(x: &mut [f32], pos: usize, head_dim: usize, theta_base: f32) {
    assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");
    assert_eq!(x.len() % head_dim, 0, "row must be a whole number of heads");
    for i in (0..x.len()).step_by(2) {
        let j = i % head_dim;
        let freq = theta_base.powf(-(j as f32) / head_dim as f32);
        let (sin, cos) = (pos as f32 * freq).sin_cos();
        let (x0, x1) = (x[i], x[i + 1]);
        x[i] = x0 * cos - x1 * sin;
        x[i + 1] = x0 * sin + x1 * cos;
    }
}

/// Apply NeoX-style (`rotate_half`) RoPE in place to a single position's
/// activation row `[n_heads * head_dim]` — the convention Qwen2, Gemma, and
/// other GPT-NeoX-lineage models use (ggml `ROPE_TYPE_NEOX`), pairing dim `j`
/// with `j + head_dim/2` instead of adjacent elements.
pub fn rope_neox(x: &mut [f32], pos: usize, head_dim: usize, theta_base: f32) {
    assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");
    assert_eq!(x.len() % head_dim, 0, "row must be a whole number of heads");
    let half = head_dim / 2;
    for h in 0..x.len() / head_dim {
        let base = h * head_dim;
        for j in 0..half {
            let freq = theta_base.powf(-(2.0 * j as f32) / head_dim as f32);
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            let (x1, x2) = (x[base + j], x[base + j + half]);
            x[base + j] = x1 * cos - x2 * sin;
            x[base + j + half] = x1 * sin + x2 * cos;
        }
    }
}

/// One decode step of causal (grouped-query) attention against a KV cache.
///
/// `q` is this position's query row `[n_heads * head_dim]`; `k_cache`/`v_cache`
/// are `[capacity, n_kv_heads * head_dim]` row-major with rows `0..=pos`
/// already filled (including this position's K/V). Returns the attention
/// output `[n_heads * head_dim]` — softmax(q·K/√d)·V over positions `0..=pos`.
pub fn attention_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    assert!(n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads));
    let kv_stride = n_kv_heads * head_dim;
    assert!(k_cache.len() >= (pos + 1) * kv_stride, "k_cache not filled to pos");
    assert!(v_cache.len() >= (pos + 1) * kv_stride, "v_cache not filled to pos");

    let mut out = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        decode_one_head(
            q,
            k_cache,
            v_cache,
            pos,
            h,
            n_heads / n_kv_heads,
            kv_stride,
            head_dim,
            &mut out[h * head_dim..(h + 1) * head_dim],
        );
    }
    out
}

/// One attention head of a single decode step, written into `dst`
/// (`head_dim` wide). Shared by the serial and team decode paths.
#[allow(clippy::too_many_arguments)]
fn decode_one_head(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    pos: usize,
    h: usize,
    group: usize,
    kv_stride: usize,
    head_dim: usize,
    dst: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_h = h / group;
    let q_vec = &q[h * head_dim..(h + 1) * head_dim];

    let mut scores = vec![0f32; pos + 1];
    for (t, s) in scores.iter_mut().enumerate() {
        let k_vec = &k_cache[t * kv_stride + kv_h * head_dim..t * kv_stride + (kv_h + 1) * head_dim];
        let dot: f32 = q_vec.iter().zip(k_vec).map(|(a, b)| a * b).sum();
        *s = dot * scale;
    }
    let probs = softmax(&scores);

    dst.fill(0.0);
    for (t, &p) in probs.iter().enumerate() {
        let v_vec = &v_cache[t * kv_stride + kv_h * head_dim..t * kv_stride + (kv_h + 1) * head_dim];
        for (d, &vv) in dst.iter_mut().zip(v_vec) {
            *d += p * vv;
        }
    }
}

/// Team version of [`attention_decode`]: heads are the parallel unit (their
/// output slices are disjoint `head_dim` runs). Fills `out[..n_heads*head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_team(
    team: &crate::team::Team,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &crate::team::TeamCell<Vec<f32>>,
) {
    assert_eq!(q.len(), n_heads * head_dim);
    assert!(n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads));
    let kv_stride = n_kv_heads * head_dim;
    assert!(k_cache.len() >= (pos + 1) * kv_stride, "k_cache not filled to pos");
    assert!(v_cache.len() >= (pos + 1) * kv_stride, "v_cache not filled to pos");
    let group = n_heads / n_kv_heads;

    // One head scans (pos+1) K rows and V rows of head_dim each.
    let head_macs = 2 * (pos + 1) * head_dim;
    crate::team::team_fill_slices(team, out, n_heads, head_dim, head_macs, |h, dst| {
        decode_one_head(q, k_cache, v_cache, pos, h, group, kv_stride, head_dim, dst);
    });
}

/// Causal scaled-dot-product self-attention.
///
/// `q`  is `[seq, n_heads * head_dim]`,
/// `k`/`v` are `[seq, n_kv_heads * head_dim]` (same seq — a single forward pass /
/// prefill window). `n_heads` must be a multiple of `n_kv_heads` (grouped-query
/// attention; set them equal for classic multi-head).
///
/// Returns `[seq, n_heads * head_dim]`. Every query position `i` attends only to
/// key positions `t <= i` (causal mask), softmax-normalized, scaled by
/// `1/sqrt(head_dim)`.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq * n_heads * head_dim);
    assert_eq!(k.len(), seq * n_kv_heads * head_dim);
    assert_eq!(v.len(), seq * n_kv_heads * head_dim);
    assert!(n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads), "n_heads must be a multiple of n_kv_heads");

    let group = n_heads / n_kv_heads; // query heads per kv head
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv_heads * head_dim;
    let mut out = vec![0f32; seq * q_stride];

    for i in 0..seq {
        for h in 0..n_heads {
            let kv_h = h / group;
            let q_vec = &q[i * q_stride + h * head_dim..i * q_stride + h * head_dim + head_dim];

            // Scores against all causal-visible keys (t <= i).
            let mut scores = vec![0f32; i + 1];
            for (t, s) in scores.iter_mut().enumerate() {
                let k_vec =
                    &k[t * kv_stride + kv_h * head_dim..t * kv_stride + kv_h * head_dim + head_dim];
                let dot: f32 = q_vec.iter().zip(k_vec).map(|(a, b)| a * b).sum();
                *s = dot * scale;
            }

            let probs = softmax(&scores);

            // Weighted sum of value vectors.
            let dst = &mut out[i * q_stride + h * head_dim..i * q_stride + h * head_dim + head_dim];
            for (t, &p) in probs.iter().enumerate() {
                let v_vec =
                    &v[t * kv_stride + kv_h * head_dim..t * kv_stride + kv_h * head_dim + head_dim];
                for (d, &vv) in dst.iter_mut().zip(v_vec) {
                    *d += p * vv;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_preserves_norm_per_pair() {
        // A rotation preserves the length of each (x1, x2) pair.
        let seq = 3;
        let n_heads = 2;
        let head_dim = 4;
        let mut x: Vec<f32> =
            (0..seq * n_heads * head_dim).map(|i| (i as f32 * 0.1).sin() + 0.3).collect();
        let before = x.clone();
        rope(&mut x, seq, n_heads, head_dim, 10000.0);

        let half = head_dim / 2;
        for p in 0..seq {
            for h in 0..n_heads {
                let base = (p * n_heads + h) * head_dim;
                for j in 0..half {
                    let n0 = before[base + j].powi(2) + before[base + j + half].powi(2);
                    let n1 = x[base + j].powi(2) + x[base + j + half].powi(2);
                    assert!((n0 - n1).abs() < 1e-5, "pair norm changed: {n0} vs {n1}");
                }
            }
        }
    }

    #[test]
    fn rope_is_identity_at_position_zero() {
        // Angle = 0 for p=0, so the first row must be unchanged.
        let head_dim = 8;
        let mut x: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
        let before = x.clone();
        rope(&mut x, 1, 1, head_dim, 10000.0);
        for (a, b) in before.iter().zip(&x) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn attention_rows_are_convex_combinations_of_v() {
        // With causal masking, position 0 attends only to itself, so out[0] == v[0].
        let seq = 4;
        let n_heads = 2;
        let head_dim = 3;
        let q: Vec<f32> = (0..seq * n_heads * head_dim).map(|i| (i as f32 * 0.05).cos()).collect();
        let k: Vec<f32> = (0..seq * n_heads * head_dim).map(|i| (i as f32 * 0.03).sin()).collect();
        let v: Vec<f32> = (0..seq * n_heads * head_dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let out = attention(&q, &k, &v, seq, n_heads, n_heads, head_dim);

        // First query position sees only key/value 0 -> output equals v[0] per head.
        for h in 0..n_heads {
            for d in 0..head_dim {
                let o = out[h * head_dim + d];
                let expected = v[h * head_dim + d];
                assert!((o - expected).abs() < 1e-5, "pos0 head{h} dim{d}: {o} vs {expected}");
            }
        }
    }

    #[test]
    fn attention_output_is_bounded_by_v_range() {
        // A softmax-weighted average can never exceed the min/max of the values
        // it averages, a good invariant that catches masking/scaling bugs.
        let seq = 5;
        let n_heads = 1;
        let head_dim = 4;
        let q: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32 * 0.2).sin()).collect();
        let k: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32 * 0.15).cos()).collect();
        let v: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = attention(&q, &k, &v, seq, n_heads, n_heads, head_dim);

        for i in 0..seq {
            // Values visible to position i are rows 0..=i.
            let visible = &v[0..(i + 1) * head_dim];
            let vmin = visible.iter().copied().fold(f32::INFINITY, f32::min);
            let vmax = visible.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for d in 0..head_dim {
                let o = out[i * head_dim + d];
                assert!(o >= vmin - 1e-5 && o <= vmax + 1e-5, "out {o} not in [{vmin},{vmax}]");
            }
        }
    }

    #[test]
    fn rope_interleaved_preserves_pair_norm_and_pos0_identity() {
        let head_dim = 8;
        let mut x: Vec<f32> = (0..2 * head_dim).map(|i| (i as f32 * 0.3).sin() + 0.1).collect();
        let before = x.clone();
        rope_interleaved(&mut x, 0, head_dim, 10000.0);
        assert_eq!(x, before, "position 0 must be identity");

        rope_interleaved(&mut x, 7, head_dim, 10000.0);
        for i in (0..x.len()).step_by(2) {
            let n0 = before[i].powi(2) + before[i + 1].powi(2);
            let n1 = x[i].powi(2) + x[i + 1].powi(2);
            assert!((n0 - n1).abs() < 1e-4, "pair norm changed at {i}: {n0} vs {n1}");
        }
    }

    #[test]
    fn rope_neox_pairs_across_halves_and_matches_batch_rope() {
        // rope_neox on a single row at position p must equal the batch rope()
        // (also rotate_half) applied to a seq where that row sits at index p.
        let n_heads = 2;
        let head_dim = 8;
        let w = n_heads * head_dim;
        let row: Vec<f32> = (0..w).map(|i| (i as f32 * 0.17).sin() + 0.2).collect();

        let pos = 3;
        let mut batch = vec![0f32; 4 * w];
        batch[pos * w..(pos + 1) * w].copy_from_slice(&row);
        rope(&mut batch, 4, n_heads, head_dim, 10000.0);

        let mut single = row.clone();
        rope_neox(&mut single, pos, head_dim, 10000.0);
        for (a, b) in single.iter().zip(&batch[pos * w..(pos + 1) * w]) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn decode_with_cache_matches_full_attention() {
        // Filling a KV cache position-by-position and querying the last position
        // must reproduce the last row of the batch (prefill) attention.
        let seq = 5;
        let n_heads = 2;
        let head_dim = 4;
        let w = n_heads * head_dim;
        let q: Vec<f32> = (0..seq * w).map(|i| (i as f32 * 0.07).sin()).collect();
        let k: Vec<f32> = (0..seq * w).map(|i| (i as f32 * 0.05).cos()).collect();
        let v: Vec<f32> = (0..seq * w).map(|i| (i as f32 * 0.03).sin()).collect();

        let full = attention(&q, &k, &v, seq, n_heads, n_heads, head_dim);
        for pos in 0..seq {
            let step = attention_decode(&q[pos * w..(pos + 1) * w], &k, &v, pos, n_heads, n_heads, head_dim);
            for (a, b) in full[pos * w..(pos + 1) * w].iter().zip(&step) {
                assert!((a - b).abs() < 1e-5, "pos {pos}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn grouped_query_attention_shapes_work() {
        // 4 query heads sharing 2 kv heads (GQA) must produce full q-shaped output.
        let seq = 3;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 2;
        let q: Vec<f32> = (0..seq * n_heads * head_dim).map(|i| i as f32 * 0.01).collect();
        let k: Vec<f32> = (0..seq * n_kv_heads * head_dim).map(|i| i as f32 * 0.02).collect();
        let v: Vec<f32> = (0..seq * n_kv_heads * head_dim).map(|i| i as f32 * 0.03).collect();
        let out = attention(&q, &k, &v, seq, n_heads, n_kv_heads, head_dim);
        assert_eq!(out.len(), seq * n_heads * head_dim);
    }
}
