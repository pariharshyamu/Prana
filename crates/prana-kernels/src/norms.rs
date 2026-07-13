//! Normalization and softmax — the elementwise glue between matmuls.
//!
//! Cactus implements these in `cactus-kernels/src/norms_rope.cpp` and the
//! sampling ops; the numerically-careful structure (max-subtracted softmax,
//! mean-square RMS) is identical here, just expressed in safe Rust.

/// RMSNorm over the last dimension: `y = x / sqrt(mean(x^2) + eps) * weight`.
/// `x` is `[rows, dim]` row-major; `weight` is length `dim`.
pub fn rmsnorm(x: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(weight.len(), dim);
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let dst = &mut out[r * dim..(r + 1) * dim];
        for i in 0..dim {
            dst[i] = row[i] * inv * weight[i];
        }
    }
    out
}

/// SiLU (swish) activation in place: `x = x * sigmoid(x)` — the gate
/// nonlinearity in Llama-family SwiGLU MLPs.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// Tanh-approximated GELU in place — the MLP activation in Gemma (and GPT
/// lineage): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
pub fn gelu_tanh(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        *v = 0.5 * *v * (1.0 + (SQRT_2_OVER_PI * (*v + 0.044715 * x3)).tanh());
    }
}

/// Numerically stable softmax over a single vector (subtract max before exp).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in &mut out {
        *v *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weight_normalizes() {
        // Constant row -> every normalized value equals 1 (rms of constant c is c).
        let x = vec![2.0, 2.0, 2.0, 2.0];
        let w = vec![1.0; 4];
        let y = rmsnorm(&x, &w, 1, 4, 0.0);
        for v in y {
            assert!((v - 1.0).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        // Monotonic: larger logit -> larger prob.
        assert!(p[0] < p[1] && p[1] < p[2]);
    }

    #[test]
    fn silu_matches_definition() {
        let mut x = vec![0.0f32, 1.0, -1.0, 4.0];
        silu(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 0.731_058_6).abs() < 1e-5); // 1*sigmoid(1)
        assert!((x[2] + 0.268_941_4).abs() < 1e-5); // -1*sigmoid(-1)
        assert!(x[3] > 3.9 && x[3] < 4.0); // large x -> ~identity
    }

    #[test]
    fn gelu_tanh_matches_reference_points() {
        let mut x = vec![0.0f32, 1.0, -1.0, 3.0];
        gelu_tanh(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 0.841_192).abs() < 1e-4); // gelu(1) ≈ 0.8412
        assert!((x[2] + 0.158_808).abs() < 1e-4); // gelu(-1) ≈ -0.1588
        assert!((x[3] - 2.995_9).abs() < 1e-3); // large x → ~identity
    }

    #[test]
    fn softmax_is_overflow_safe() {
        // Without max-subtraction this would be inf/inf = NaN.
        let p = softmax(&[1000.0, 1001.0]);
        assert!(p.iter().all(|v| v.is_finite()));
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
