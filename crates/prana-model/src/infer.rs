//! The KV-cached transformer forward pass and generation loop, composed
//! entirely from `prana-kernels` safe kernels. This is the Llama architecture
//! as in llama2.c's `run.c`: RMSNorm → QKV → interleaved RoPE → causal
//! attention over the cache → output projection → residual, then
//! RMSNorm → SwiGLU MLP → residual; final RMSNorm → vocab logits.

use prana_kernels::{attention_decode, rmsnorm, rope_interleaved, silu};

use crate::checkpoint::Model;
use crate::sampler::Sampler;
use crate::tokenizer::{Tokenizer, BOS, EOS};

/// Mutable inference state: the per-layer KV caches.
pub struct KvCache {
    k: Vec<Vec<f32>>, // per layer: [seq_len, kv_dim]
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(model: &Model) -> Self {
        let c = &model.config;
        let per_layer = c.seq_len * c.kv_dim();
        Self {
            k: (0..c.n_layers).map(|_| vec![0f32; per_layer]).collect(),
            v: (0..c.n_layers).map(|_| vec![0f32; per_layer]).collect(),
        }
    }
}

/// Run one token through the model at position `pos`, updating the cache.
/// Returns the vocabulary logits.
pub fn forward(model: &Model, cache: &mut KvCache, token: u32, pos: usize) -> Vec<f32> {
    let c = &model.config;
    let dim = c.dim;
    let head_dim = c.head_dim();
    let kv_dim = c.kv_dim();
    const ROPE_THETA: f32 = 10000.0;

    let mut x = model.tok_emb[token as usize * dim..(token as usize + 1) * dim].to_vec();

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention block ---
        let xb = rmsnorm(&x, &layer.rms_att, 1, dim, 1e-5);
        let mut q = layer.wq.apply(&xb);
        let mut k = layer.wk.apply(&xb);
        let v = layer.wv.apply(&xb);
        rope_interleaved(&mut q, pos, head_dim, ROPE_THETA);
        rope_interleaved(&mut k, pos, head_dim, ROPE_THETA);

        cache.k[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        cache.v[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);

        let attn = attention_decode(&q, &cache.k[l], &cache.v[l], pos, c.n_heads, c.n_kv_heads, head_dim);
        for (xi, oi) in x.iter_mut().zip(layer.wo.apply(&attn)) {
            *xi += oi;
        }

        // --- SwiGLU MLP block ---
        let xb = rmsnorm(&x, &layer.rms_ffn, 1, dim, 1e-5);
        let mut gate = layer.w1.apply(&xb);
        let up = layer.w3.apply(&xb);
        silu(&mut gate);
        for (g, u) in gate.iter_mut().zip(&up) {
            *g *= u;
        }
        for (xi, di) in x.iter_mut().zip(layer.w2.apply(&gate)) {
            *xi += di;
        }
    }

    let x = rmsnorm(&x, &model.rms_final, 1, dim, 1e-5);
    model.logits(&x)
}

/// Statistics from a generation run.
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub seconds: f64,
}

/// Generate up to `steps` tokens continuing `prompt`, streaming each decoded
/// piece to `on_piece`. Stops at EOS/BOS or the model's context limit.
pub fn generate(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    steps: usize,
    sampler: &mut Sampler,
    mut on_piece: impl FnMut(&[u8]),
) -> GenStats {
    let prompt_tokens = tokenizer.encode(prompt, true, false);
    assert!(!prompt_tokens.is_empty());
    let max_pos = model.config.seq_len.min(prompt_tokens.len() + steps);

    let mut cache = KvCache::new(model);
    let start = std::time::Instant::now();
    let mut token = prompt_tokens[0];
    let mut generated = 0usize;

    for pos in 0..max_pos {
        let logits = forward(model, &mut cache, token, pos);
        if pos + 1 >= max_pos {
            break; // context full: the next token would have nowhere to sit
        }
        let next = if pos + 1 < prompt_tokens.len() {
            prompt_tokens[pos + 1] // teacher-force the rest of the prompt
        } else {
            sampler.sample(&logits)
        };
        if pos + 1 >= prompt_tokens.len() {
            if next == EOS || next == BOS {
                break;
            }
            generated += 1;
        }
        on_piece(&tokenizer.decode(token, next));
        token = next;
    }

    GenStats {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated,
        seconds: start.elapsed().as_secs_f64(),
    }
}
