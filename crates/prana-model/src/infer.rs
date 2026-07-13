//! The KV-cached transformer forward pass and generation loop, composed
//! entirely from `prana-kernels` safe kernels. Covers the Llama family plus
//! the knobs Qwen2 and Gemma turn: QKV biases, NeoX vs interleaved RoPE,
//! SiLU vs GELU MLP gates, decoupled head_dim, per-arch norm epsilon, and
//! Gemma's √dim embedding scale.

use prana_kernels::{attention_decode, gelu_tanh, rmsnorm, rope_interleaved, rope_neox, silu};

use crate::checkpoint::{Activation, Model};
use crate::tokenizer::Tokenize;

/// Mutable inference state: the per-layer KV caches.
pub struct KvCache {
    k: Vec<Vec<f32>>, // per layer: [seq_len, kv_dim]
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(model: &Model) -> Self {
        Self::with_len(model, model.config.seq_len)
    }

    /// Cache sized for `positions` tokens. Modern models declare 32k+
    /// contexts; a short generation shouldn't pay for cache it never fills.
    pub fn with_len(model: &Model, positions: usize) -> Self {
        let c = &model.config;
        let per_layer = positions.min(c.seq_len) * c.kv_dim();
        Self {
            k: (0..c.n_layers).map(|_| vec![0f32; per_layer]).collect(),
            v: (0..c.n_layers).map(|_| vec![0f32; per_layer]).collect(),
        }
    }
}

fn add_bias(x: &mut [f32], bias: &Option<Vec<f32>>) {
    if let Some(b) = bias {
        for (xi, bi) in x.iter_mut().zip(b) {
            *xi += bi;
        }
    }
}

/// Run one token through the model at position `pos`, updating the cache.
/// Returns the vocabulary logits.
pub fn forward(model: &Model, cache: &mut KvCache, token: u32, pos: usize) -> Vec<f32> {
    let c = &model.config;
    let dim = c.dim;
    let head_dim = c.head_dim;
    let kv_dim = c.kv_dim();

    let mut x = model.tok_emb.row(token as usize, dim);
    if c.emb_scale != 1.0 {
        for v in x.iter_mut() {
            *v *= c.emb_scale;
        }
    }

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention block ---
        let xb = rmsnorm(&x, &layer.rms_att, 1, dim, c.norm_eps);
        let mut q = layer.wq.apply(&xb);
        let mut k = layer.wk.apply(&xb);
        let mut v = layer.wv.apply(&xb);
        add_bias(&mut q, &layer.bq);
        add_bias(&mut k, &layer.bk);
        add_bias(&mut v, &layer.bv);
        if c.rope_neox {
            rope_neox(&mut q, pos, head_dim, c.rope_theta);
            rope_neox(&mut k, pos, head_dim, c.rope_theta);
        } else {
            rope_interleaved(&mut q, pos, head_dim, c.rope_theta);
            rope_interleaved(&mut k, pos, head_dim, c.rope_theta);
        }

        cache.k[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        cache.v[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);

        let attn = attention_decode(&q, &cache.k[l], &cache.v[l], pos, c.n_heads, c.n_kv_heads, head_dim);
        for (xi, oi) in x.iter_mut().zip(layer.wo.apply(&attn)) {
            *xi += oi;
        }

        // --- gated MLP block ---
        let xb = rmsnorm(&x, &layer.rms_ffn, 1, dim, c.norm_eps);
        let mut gate = layer.w1.apply(&xb);
        let up = layer.w3.apply(&xb);
        match c.act {
            Activation::Silu => silu(&mut gate),
            Activation::GeluTanh => gelu_tanh(&mut gate),
        }
        for (g, u) in gate.iter_mut().zip(&up) {
            *g *= u;
        }
        for (xi, di) in x.iter_mut().zip(layer.w2.apply(&gate)) {
            *xi += di;
        }
    }

    let x = rmsnorm(&x, &model.rms_final, 1, dim, c.norm_eps);
    model.logits(&x)
}

/// Statistics from a generation run.
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub seconds: f64,
}

/// Generate up to `steps` tokens continuing `prompt`, streaming each decoded
/// piece to `on_piece` (second arg is true while the piece belongs to the
/// prompt prefill — chat UIs hide those). Stops on the tokenizer's stop
/// tokens or the model's context limit.
pub fn generate(
    model: &Model,
    tokenizer: &dyn Tokenize,
    prompt: &str,
    steps: usize,
    sampler: &mut crate::sampler::Sampler,
    mut on_piece: impl FnMut(&[u8], bool),
) -> GenStats {
    let prompt_tokens = tokenizer.encode_prompt(prompt);
    assert!(!prompt_tokens.is_empty(), "prompt encoded to zero tokens");
    let max_pos = model.config.seq_len.min(prompt_tokens.len() + steps);

    let mut cache = KvCache::with_len(model, max_pos);
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
            if tokenizer.is_stop(next) {
                break;
            }
            generated += 1;
        }
        on_piece(&tokenizer.decode(token, next), pos + 1 < prompt_tokens.len());
        token = next;
    }

    GenStats {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated,
        seconds: start.elapsed().as_secs_f64(),
    }
}
