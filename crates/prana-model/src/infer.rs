//! The KV-cached transformer forward pass and generation loop, composed
//! entirely from `prana-kernels` safe kernels. Covers the Llama family plus
//! the knobs Qwen2 and Gemma turn: QKV biases, NeoX vs interleaved RoPE,
//! SiLU vs GELU MLP gates, decoupled head_dim, per-arch norm epsilon, and
//! Gemma's √dim embedding scale.

use prana_kernels::{
    attention_decode, attention_decode_team, gelu_tanh, matmul_f32_team, matmul_kquant_team,
    quantize_acts, rmsnorm, rope_interleaved, rope_neox, run_team, silu, QuantActs, Team,
    TeamCell,
};

use crate::checkpoint::{Activation, Embedding, Model};
use crate::tokenizer::Tokenize;

/// Mutable inference state: the per-layer KV caches. The buffers live in
/// `TeamCell`s so team members can read them during attention while serial
/// sections fill them — see `forward_team`.
pub struct KvCache {
    k: Vec<TeamCell<Vec<f32>>>, // per layer: [seq_len, kv_dim]
    v: Vec<TeamCell<Vec<f32>>>,
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
            k: (0..c.n_layers).map(|_| TeamCell::new(vec![0f32; per_layer])).collect(),
            v: (0..c.n_layers).map(|_| TeamCell::new(vec![0f32; per_layer])).collect(),
        }
    }
}

/// Preallocated intermediate buffers for `forward_team`, shared across the
/// team through guard-checked cells.
pub struct Scratch {
    x: TeamCell<Vec<f32>>,     // residual stream [dim]
    xb: TeamCell<Vec<f32>>,    // normed input [dim]
    q: TeamCell<Vec<f32>>,     // [q_dim]
    k: TeamCell<Vec<f32>>,     // [kv_dim]
    v: TeamCell<Vec<f32>>,     // [kv_dim]
    attn: TeamCell<Vec<f32>>,  // [q_dim]
    o: TeamCell<Vec<f32>>,     // matmul output back into the stream [dim]
    gate: TeamCell<Vec<f32>>,  // [hidden]
    up: TeamCell<Vec<f32>>,    // [hidden]
    logits: TeamCell<Vec<f32>>, // [vocab]
    /// Quantized activations for the current matmul input.
    acts: TeamCell<QuantActs>,
}

impl Scratch {
    pub fn new(model: &Model) -> Self {
        let c = &model.config;
        let empty = || QuantActs { q: Vec::new(), scales: Vec::new(), sums: Vec::new() };
        Self {
            x: TeamCell::new(vec![0f32; c.dim]),
            xb: TeamCell::new(vec![0f32; c.dim]),
            q: TeamCell::new(vec![0f32; c.q_dim()]),
            k: TeamCell::new(vec![0f32; c.kv_dim()]),
            v: TeamCell::new(vec![0f32; c.kv_dim()]),
            attn: TeamCell::new(vec![0f32; c.q_dim()]),
            o: TeamCell::new(vec![0f32; c.dim]),
            gate: TeamCell::new(vec![0f32; c.hidden_dim]),
            up: TeamCell::new(vec![0f32; c.hidden_dim]),
            logits: TeamCell::new(vec![0f32; c.vocab_size]),
            acts: TeamCell::new(empty()),
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

/// Cumulative per-section nanoseconds, filled only when `PRANA_TIMING` is
/// set (one branch per section otherwise). Indices: see `timing_report`.
static TIMING_NS: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PRANA_TIMING").is_ok())
}

/// Opt into team execution (`PRANA_TEAM=1`) — see the note in [`generate`].
fn team_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PRANA_TEAM").is_ok())
}

/// Where the forward passes spent their time so far (needs `PRANA_TIMING`).
pub fn timing_report() -> Option<String> {
    if !timing_enabled() {
        return None;
    }
    let labels = ["attn matmuls (qkvo)", "ffn matmuls", "lm head", "attention+glue", "embed+norms"];
    let ms: Vec<f64> = TIMING_NS
        .iter()
        .map(|n| n.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6)
        .collect();
    let total: f64 = ms.iter().sum();
    let mut out = format!("  forward-pass time  : {total:.0} ms total\n");
    for (l, m) in labels.iter().zip(&ms) {
        out.push_str(&format!("    {l:<20}: {m:>8.1} ms  ({:>4.1}%)\n", m / total * 100.0));
    }
    Some(out)
}

/// Time `f` into timing slot `slot` when enabled.
#[inline]
fn timed<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    if !timing_enabled() {
        return f();
    }
    let t = std::time::Instant::now();
    let out = f();
    TIMING_NS[slot].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

/// Run one token through the model at position `pos`, updating the cache.
/// Returns the vocabulary logits.
pub fn forward(model: &Model, cache: &mut KvCache, token: u32, pos: usize) -> Vec<f32> {
    let c = &model.config;
    let dim = c.dim;
    let head_dim = c.head_dim;
    let kv_dim = c.kv_dim();

    let mut x = timed(4, || model.tok_emb.row(token as usize, dim));
    if c.emb_scale != 1.0 {
        for v in x.iter_mut() {
            *v *= c.emb_scale;
        }
    }

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention block ---
        let xb = timed(4, || rmsnorm(&x, &layer.rms_att, 1, dim, c.norm_eps));
        let (mut q, mut k, mut v) =
            timed(0, || (layer.wq.apply(&xb), layer.wk.apply(&xb), layer.wv.apply(&xb)));
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

        cache.k[l].get_mut()[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        cache.v[l].get_mut()[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);

        let attn = timed(3, || {
            let (ck, cv) = (cache.k[l].read(), cache.v[l].read());
            attention_decode(&q, &ck, &cv, pos, c.n_heads, c.n_kv_heads, head_dim)
        });
        let o = timed(0, || layer.wo.apply(&attn));
        for (xi, oi) in x.iter_mut().zip(o) {
            *xi += oi;
        }

        // --- gated MLP block ---
        let xb = timed(4, || rmsnorm(&x, &layer.rms_ffn, 1, dim, c.norm_eps));
        let (mut gate, up) = timed(1, || (layer.w1.apply(&xb), layer.w3.apply(&xb)));
        match c.act {
            Activation::Silu => silu(&mut gate),
            Activation::GeluTanh => gelu_tanh(&mut gate),
        }
        for (g, u) in gate.iter_mut().zip(&up) {
            *g *= u;
        }
        let down = timed(1, || layer.w2.apply(&gate));
        for (xi, di) in x.iter_mut().zip(down) {
            *xi += di;
        }
    }

    let x = timed(4, || rmsnorm(&x, &model.rms_final, 1, dim, c.norm_eps));
    timed(2, || model.logits(&x))
}

/// One token through the model as a **team**: a single pool dispatch per
/// token, all threads walking every layer together with µs spin barriers
/// between ops (llama.cpp's `ggml_graph_compute` model). Matmuls and
/// attention heads split across the team; norms/rope/bias glue runs on
/// thread 0 inside `serial` sections. Numerically identical to [`forward`]
/// (same per-row dots in the same order) — asserted by a parity test.
pub fn forward_team(
    model: &Model,
    cache: &KvCache,
    scratch: &Scratch,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    run_team(|team| forward_team_member(model, cache, scratch, token, pos, &team));
    scratch.logits.read().clone()
}

/// The per-member body of [`forward_team`]. Every member runs this whole
/// function; `serial` and the team kernels internally keep them in step.
/// Barrier discipline: every member must reach every `serial`/team-kernel
/// call (no member-dependent control flow around them), and `TeamCell`
/// read guards live only inside the parallel block that needs them.
fn forward_team_member(
    model: &Model,
    cache: &KvCache,
    scratch: &Scratch,
    token: u32,
    pos: usize,
    team: &Team,
) {
    let c = &model.config;
    let (dim, head_dim, kv_dim) = (c.dim, c.head_dim, c.kv_dim());

    team.serial(|| {
        let mut x = scratch.x.get_mut();
        x.copy_from_slice(&model.tok_emb.row(token as usize, dim));
        if c.emb_scale != 1.0 {
            for v in x.iter_mut() {
                *v *= c.emb_scale;
            }
        }
    });

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention block ---
        team.serial(|| {
            let x = scratch.x.read();
            let xb = rmsnorm(&x, &layer.rms_att, 1, dim, c.norm_eps);
            *scratch.acts.get_mut() = quantize_acts(&xb);
            *scratch.xb.get_mut() = xb;
        });
        {
            let xb = scratch.xb.read();
            let acts = scratch.acts.read();
            layer.wq.apply_team(team, &xb, &acts, &scratch.q);
            layer.wk.apply_team(team, &xb, &acts, &scratch.k);
            layer.wv.apply_team(team, &xb, &acts, &scratch.v);
        }
        team.serial(|| {
            let mut q = scratch.q.get_mut();
            let mut k = scratch.k.get_mut();
            let mut v = scratch.v.get_mut();
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
            cache.k[l].get_mut()[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            cache.v[l].get_mut()[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
        });
        {
            let q = scratch.q.read();
            let (ck, cv) = (cache.k[l].read(), cache.v[l].read());
            attention_decode_team(team, &q, &ck, &cv, pos, c.n_heads, c.n_kv_heads, head_dim, &scratch.attn);
        }
        team.serial(|| {
            *scratch.acts.get_mut() = quantize_acts(&scratch.attn.read());
        });
        {
            let attn = scratch.attn.read();
            let acts = scratch.acts.read();
            layer.wo.apply_team(team, &attn, &acts, &scratch.o);
        }

        // --- gated MLP block ---
        team.serial(|| {
            let mut x = scratch.x.get_mut();
            for (xi, oi) in x.iter_mut().zip(scratch.o.read().iter()) {
                *xi += oi;
            }
            let xb = rmsnorm(&x, &layer.rms_ffn, 1, dim, c.norm_eps);
            *scratch.acts.get_mut() = quantize_acts(&xb);
            *scratch.xb.get_mut() = xb;
        });
        {
            let xb = scratch.xb.read();
            let acts = scratch.acts.read();
            layer.w1.apply_team(team, &xb, &acts, &scratch.gate);
            layer.w3.apply_team(team, &xb, &acts, &scratch.up);
        }
        team.serial(|| {
            let mut gate = scratch.gate.get_mut();
            match c.act {
                Activation::Silu => silu(&mut gate),
                Activation::GeluTanh => gelu_tanh(&mut gate),
            }
            for (g, u) in gate.iter_mut().zip(scratch.up.read().iter()) {
                *g *= u;
            }
            *scratch.acts.get_mut() = quantize_acts(&gate);
        });
        {
            let gate = scratch.gate.read();
            let acts = scratch.acts.read();
            layer.w2.apply_team(team, &gate, &acts, &scratch.o);
        }
        team.serial(|| {
            let mut x = scratch.x.get_mut();
            for (xi, oi) in x.iter_mut().zip(scratch.o.read().iter()) {
                *xi += oi;
            }
        });
    }

    // --- classifier ---
    team.serial(|| {
        let x = scratch.x.read();
        let xb = rmsnorm(&x, &model.rms_final, 1, dim, c.norm_eps);
        *scratch.acts.get_mut() = quantize_acts(&xb);
        *scratch.xb.get_mut() = xb;
    });
    {
        let xb = scratch.xb.read();
        let acts = scratch.acts.read();
        match (&model.wcls, &model.tok_emb) {
            (Some(cls), _) => cls.apply_team(team, &xb, &acts, &scratch.logits),
            (None, Embedding::F32(w)) => {
                matmul_f32_team(team, &xb, w, c.vocab_size, &scratch.logits)
            }
            (None, Embedding::KQuant(m)) => matmul_kquant_team(team, &acts, m, &scratch.logits),
        }
    }
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
    let scratch = Scratch::new(model);
    let start = std::time::Instant::now();
    let mut token = prompt_tokens[0];
    let mut generated = 0usize;

    for pos in 0..max_pos {
        // Measured on Ultra-5-class hybrid Windows hardware, the per-op pool
        // path outruns team execution (~29 vs ~21 tok/s on a 0.5B q4_0):
        // both leave the small QKV/O projections serial, but the team's
        // members busy-spin through those sections and through every barrier
        // tail, which eats the power/SMT headroom the working threads need.
        // Team execution (PRANA_TEAM=1) stays as parity-tested groundwork —
        // its barrier costs measure healthy (~1-2µs), so it should win once
        // the serial glue itself is parallelized. See EVALUATION.md.
        let logits = if team_enabled() {
            forward_team(model, &cache, &scratch, token, pos)
        } else {
            forward(model, &mut cache, token, pos)
        };
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
