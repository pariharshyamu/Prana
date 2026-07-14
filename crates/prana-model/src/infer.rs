//! The KV-cached transformer forward pass and generation loop, composed
//! entirely from `prana-kernels` safe kernels. Covers the Llama family plus
//! the knobs Qwen2 and Gemma turn: QKV biases, NeoX vs interleaved RoPE,
//! SiLU vs GELU MLP gates, decoupled head_dim, per-arch norm epsilon, and
//! Gemma's √dim embedding scale.

use prana_kernels::{
    attention, attention_decode, attention_decode_team, attention_verify, f16_kv_write, gelu_tanh,
    matmul_f32_team, matmul_kquant_team, quantize_acts, rmsnorm, rope_interleaved, rope_neox,
    run_team, silu, F16KvView, QuantActs, Team, TeamCell,
};

use crate::checkpoint::{Activation, Embedding, Model};
use crate::tokenizer::Tokenize;

/// Mutable inference state: the per-layer KV caches. The buffers live in
/// `TeamCell`s so team members can read them during attention while serial
/// sections fill them — see `forward_team`.
///
/// K and V are stored at **half precision** (`u16` f16 bits): attention reads
/// the whole cache every decode step, and that read grows with context, so
/// halving it is the KV-cache optimization that scales. Values are converted
/// f32→f16 on write ([`f16_kv_write`]) and f16→f32 on read (`F16KvView`); the
/// added rounding is well below the model's existing quantization noise.
pub struct KvCache {
    k: Vec<TeamCell<Vec<u16>>>, // per layer: [seq_len, kv_dim] f16 bits
    v: Vec<TeamCell<Vec<u16>>>,
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
            k: (0..c.n_layers).map(|_| TeamCell::new(vec![0u16; per_layer])).collect(),
            v: (0..c.n_layers).map(|_| TeamCell::new(vec![0u16; per_layer])).collect(),
        }
    }
}

/// Shared intermediate buffers for `forward_team` — only the tensors that
/// parallel ops *write* live here (matmul / attention outputs). Everything
/// small (norms, rope, activation quantization) is recomputed redundantly
/// per member on private buffers: a few µs of duplicate arithmetic instead
/// of a barrier plus idle spinning.
pub struct Scratch {
    qkv: TeamCell<Vec<f32>>,    // fused Q|K|V projection [q_dim + 2*kv_dim]
    attn: TeamCell<Vec<f32>>,   // attention output [q_dim]
    o: TeamCell<Vec<f32>>,      // projection back into the stream [dim]
    gu: TeamCell<Vec<f32>>,     // fused gate|up [2*hidden]
    logits: TeamCell<Vec<f32>>, // [vocab]
}

impl Scratch {
    pub fn new(model: &Model) -> Self {
        let c = &model.config;
        Self {
            qkv: TeamCell::new(vec![0f32; c.q_dim() + 2 * c.kv_dim()]),
            attn: TeamCell::new(vec![0f32; c.q_dim()]),
            o: TeamCell::new(vec![0f32; c.dim]),
            gu: TeamCell::new(vec![0f32; 2 * c.hidden_dim]),
            logits: TeamCell::new(vec![0f32; c.vocab_size]),
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
    forward_impl(model, cache, token, pos, true).expect("logits requested")
}

/// [`forward`] with the LM head optional: teacher-forced prefill positions
/// never look at their logits, and the head is the single largest matmul in
/// the model (vocab × dim — ~30% of per-token weight traffic on a 152k-vocab
/// 0.5B). llama.cpp's graphs do the same via `inp_out_ids`/`get_rows`.
fn forward_impl(
    model: &Model,
    cache: &mut KvCache,
    token: u32,
    pos: usize,
    need_logits: bool,
) -> Option<Vec<f32>> {
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

        f16_kv_write(&mut cache.k[l].get_mut(), pos, kv_dim, &k);
        f16_kv_write(&mut cache.v[l].get_mut(), pos, kv_dim, &v);

        let attn = timed(3, || {
            let (ck, cv) = (cache.k[l].read(), cache.v[l].read());
            let (kv, vv) = (F16KvView { data: &ck }, F16KvView { data: &cv });
            attention_decode(&q, kv, vv, pos, c.n_heads, c.n_kv_heads, head_dim)
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

    if !need_logits {
        return None;
    }
    let x = timed(4, || rmsnorm(&x, &model.rms_final, 1, dim, c.norm_eps));
    Some(timed(2, || model.logits(&x)))
}

/// Batch-process `tokens` at positions `0..tokens.len()`, filling the KV
/// cache. No logits — the caller's decode loop re-enters at the last
/// prompt position. Weight rows stream once per matmul for the whole batch
/// (row-outer kernels), which is what makes prompt processing several times
/// faster than token-by-token forwards; the batch attention kernel shares
/// its accumulation order with the decode path, so the resulting cache and
/// downstream logits are bit-identical to sequential prefill.
pub fn prefill(model: &Model, cache: &mut KvCache, tokens: &[u32]) {
    let c = &model.config;
    let (dim, head_dim, kv_dim, q_dim) = (c.dim, c.head_dim, c.kv_dim(), c.q_dim());
    let n = tokens.len();

    let mut x = Vec::with_capacity(n * dim);
    for &t in tokens {
        x.extend(model.tok_emb.row(t as usize, dim));
    }
    if c.emb_scale != 1.0 {
        for v in x.iter_mut() {
            *v *= c.emb_scale;
        }
    }

    let row_acts = |xs: &[f32], width: usize| -> Vec<QuantActs> {
        xs.chunks_exact(width).map(quantize_acts).collect()
    };

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention block ---
        let xb = rmsnorm(&x, &layer.rms_att, n, dim, c.norm_eps);
        let acts = row_acts(&xb, dim);
        let mut q = layer.wq.apply_prefill(&xb, &acts);
        let mut k = layer.wk.apply_prefill(&xb, &acts);
        let mut v = layer.wv.apply_prefill(&xb, &acts);
        for t in 0..n {
            add_bias(&mut q[t * q_dim..(t + 1) * q_dim], &layer.bq);
            add_bias(&mut k[t * kv_dim..(t + 1) * kv_dim], &layer.bk);
            add_bias(&mut v[t * kv_dim..(t + 1) * kv_dim], &layer.bv);
            if c.rope_neox {
                rope_neox(&mut q[t * q_dim..(t + 1) * q_dim], t, head_dim, c.rope_theta);
                rope_neox(&mut k[t * kv_dim..(t + 1) * kv_dim], t, head_dim, c.rope_theta);
            } else {
                rope_interleaved(&mut q[t * q_dim..(t + 1) * q_dim], t, head_dim, c.rope_theta);
                rope_interleaved(&mut k[t * kv_dim..(t + 1) * kv_dim], t, head_dim, c.rope_theta);
            }
        }
        f16_kv_write(&mut cache.k[l].get_mut(), 0, kv_dim, &k);
        f16_kv_write(&mut cache.v[l].get_mut(), 0, kv_dim, &v);

        let attn = attention(&q, &k, &v, n, c.n_heads, c.n_kv_heads, head_dim);
        let acts_attn = row_acts(&attn, q_dim);
        let o = layer.wo.apply_prefill(&attn, &acts_attn);
        for (xi, oi) in x.iter_mut().zip(o) {
            *xi += oi;
        }

        // --- gated MLP block ---
        let xb = rmsnorm(&x, &layer.rms_ffn, n, dim, c.norm_eps);
        let acts = row_acts(&xb, dim);
        let mut gate = layer.w1.apply_prefill(&xb, &acts);
        let up = layer.w3.apply_prefill(&xb, &acts);
        match c.act {
            Activation::Silu => silu(&mut gate),
            Activation::GeluTanh => gelu_tanh(&mut gate),
        }
        for (g, u) in gate.iter_mut().zip(&up) {
            *g *= u;
        }
        let acts_h = row_acts(&gate, c.hidden_dim);
        let down = layer.w2.apply_prefill(&gate, &acts_h);
        for (xi, di) in x.iter_mut().zip(down) {
            *xi += di;
        }
    }
}

/// Process `tokens` at absolute positions `pos0..pos0+tokens.len()` in one
/// batched pass, writing their K/V into the cache and returning the logits
/// for **every** position (`[n, vocab]`). Each position attends to the full
/// cache (rows `0..=pos0+i`), so the cache must already hold rows `0..pos0`.
///
/// This is the speculative-decoding verification kernel: verifying k drafted
/// tokens costs one weight-streaming pass (each weight row read once for all
/// k positions) instead of k separate decode steps.
pub fn forward_batch(model: &Model, cache: &mut KvCache, tokens: &[u32], pos0: usize) -> Vec<f32> {
    let c = &model.config;
    let (dim, head_dim, kv_dim, q_dim) = (c.dim, c.head_dim, c.kv_dim(), c.q_dim());
    let n = tokens.len();

    let mut x = Vec::with_capacity(n * dim);
    for &t in tokens {
        x.extend(model.tok_emb.row(t as usize, dim));
    }
    if c.emb_scale != 1.0 {
        for v in x.iter_mut() {
            *v *= c.emb_scale;
        }
    }

    let row_acts = |xs: &[f32], width: usize| -> Vec<QuantActs> {
        xs.chunks_exact(width).map(quantize_acts).collect()
    };

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention block ---
        let xb = rmsnorm(&x, &layer.rms_att, n, dim, c.norm_eps);
        let acts = row_acts(&xb, dim);
        let mut q = layer.wq.apply_prefill(&xb, &acts);
        let mut k = layer.wk.apply_prefill(&xb, &acts);
        let mut v = layer.wv.apply_prefill(&xb, &acts);
        for t in 0..n {
            let pos = pos0 + t;
            add_bias(&mut q[t * q_dim..(t + 1) * q_dim], &layer.bq);
            add_bias(&mut k[t * kv_dim..(t + 1) * kv_dim], &layer.bk);
            add_bias(&mut v[t * kv_dim..(t + 1) * kv_dim], &layer.bv);
            if c.rope_neox {
                rope_neox(&mut q[t * q_dim..(t + 1) * q_dim], pos, head_dim, c.rope_theta);
                rope_neox(&mut k[t * kv_dim..(t + 1) * kv_dim], pos, head_dim, c.rope_theta);
            } else {
                rope_interleaved(&mut q[t * q_dim..(t + 1) * q_dim], pos, head_dim, c.rope_theta);
                rope_interleaved(&mut k[t * kv_dim..(t + 1) * kv_dim], pos, head_dim, c.rope_theta);
            }
        }
        // Write the new K/V rows into the cache at their absolute positions,
        // then attend against the whole cache.
        f16_kv_write(&mut cache.k[l].get_mut(), pos0, kv_dim, &k);
        f16_kv_write(&mut cache.v[l].get_mut(), pos0, kv_dim, &v);
        let attn = {
            let (ck, cv) = (cache.k[l].read(), cache.v[l].read());
            let (kv, vv) = (F16KvView { data: &ck }, F16KvView { data: &cv });
            attention_verify(&q, kv, vv, pos0, n, c.n_heads, c.n_kv_heads, head_dim)
        };
        let acts_attn = row_acts(&attn, q_dim);
        let o = layer.wo.apply_prefill(&attn, &acts_attn);
        for (xi, oi) in x.iter_mut().zip(o) {
            *xi += oi;
        }

        // --- gated MLP block ---
        let xb = rmsnorm(&x, &layer.rms_ffn, n, dim, c.norm_eps);
        let acts = row_acts(&xb, dim);
        let mut gate = layer.w1.apply_prefill(&xb, &acts);
        let up = layer.w3.apply_prefill(&xb, &acts);
        match c.act {
            Activation::Silu => silu(&mut gate),
            Activation::GeluTanh => gelu_tanh(&mut gate),
        }
        for (g, u) in gate.iter_mut().zip(&up) {
            *g *= u;
        }
        let acts_h = row_acts(&gate, c.hidden_dim);
        let down = layer.w2.apply_prefill(&gate, &acts_h);
        for (xi, di) in x.iter_mut().zip(down) {
            *xi += di;
        }
    }

    // Classifier for every position, batched: the LM head streams once for
    // all n rows (row-outer), not once per row — the difference between
    // verifying k tokens for one head-stream vs k.
    let xb = rmsnorm(&x, &model.rms_final, n, dim, c.norm_eps);
    model.logits_batch(&xb, n)
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
    forward_team_impl(model, cache, scratch, token, pos, true).expect("logits requested")
}

fn forward_team_impl(
    model: &Model,
    cache: &KvCache,
    scratch: &Scratch,
    token: u32,
    pos: usize,
    need_logits: bool,
) -> Option<Vec<f32>> {
    run_team(|team| forward_team_member(model, cache, scratch, token, pos, need_logits, &team));
    need_logits.then(|| scratch.logits.read().clone())
}

/// The per-member body of [`forward_team`]. Every member runs this whole
/// function; the team kernels' internal barriers keep them in step.
///
/// Structure (A1/A2 of the llama.cpp-gap plan): the *only* shared writes
/// are the five parallel ops per layer — fused QKV, attention, `wo`, fused
/// gate|up, `w2` — each a barrier-bracketed team op (≈12 barriers/layer).
/// All scalar glue (norms, rope, bias, silu·mul, activation quantization,
/// residual adds) is computed **redundantly by every member** on private
/// buffers: the inputs are identical, so the results are identical, and a
/// few µs of duplicated arithmetic beats a barrier plus (nth−1) idle
/// spinners. TeamCell read guards live only inside the block that needs
/// them, always dropped before the next team op's entry barrier.
fn forward_team_member(
    model: &Model,
    cache: &KvCache,
    scratch: &Scratch,
    token: u32,
    pos: usize,
    need_logits: bool,
    team: &Team,
) {
    let c = &model.config;
    let (dim, head_dim, kv_dim, q_dim) = (c.dim, c.head_dim, c.kv_dim(), c.q_dim());

    // Member-private residual stream (identical on every member).
    let mut x = model.tok_emb.row(token as usize, dim);
    if c.emb_scale != 1.0 {
        for v in x.iter_mut() {
            *v *= c.emb_scale;
        }
    }

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention block ---
        let xb = rmsnorm(&x, &layer.rms_att, 1, dim, c.norm_eps);
        let acts = quantize_acts(&xb);
        {
            // Q, K and V share the input, so they run as ONE parallel op
            // over q_dim + 2*kv_dim fused output rows.
            let (wq, wk, wv) = (&layer.wq, &layer.wk, &layer.wv);
            prana_kernels::team_fill_rows_weighted(team, &scratch.qkv, q_dim + 2 * kv_dim, dim, |r| {
                if r < q_dim {
                    wq.row_dot(&xb, &acts, r)
                } else if r < q_dim + kv_dim {
                    wk.row_dot(&xb, &acts, r - q_dim)
                } else {
                    wv.row_dot(&xb, &acts, r - q_dim - kv_dim)
                }
            });
        }
        // Redundant per member: bias + rope on private copies.
        let (mut q, mut k, mut v);
        {
            let qkv = scratch.qkv.read();
            q = qkv[..q_dim].to_vec();
            k = qkv[q_dim..q_dim + kv_dim].to_vec();
            v = qkv[q_dim + kv_dim..].to_vec();
        }
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
        // The KV cache is the one shared write the glue needs: thread 0
        // stores its (identical) copy.
        team.serial(|| {
            f16_kv_write(&mut cache.k[l].get_mut(), pos, kv_dim, &k);
            f16_kv_write(&mut cache.v[l].get_mut(), pos, kv_dim, &v);
        });
        {
            let (ck, cv) = (cache.k[l].read(), cache.v[l].read());
            let (kv, vv) = (F16KvView { data: &ck }, F16KvView { data: &cv });
            attention_decode_team(team, &q, kv, vv, pos, c.n_heads, c.n_kv_heads, head_dim, &scratch.attn);
        }
        let (attn, acts_attn) = {
            let a = scratch.attn.read();
            (a.to_vec(), quantize_acts(&a))
        };
        layer.wo.apply_team(team, &attn, &acts_attn, &scratch.o);
        {
            let o = scratch.o.read();
            for (xi, oi) in x.iter_mut().zip(o.iter()) {
                *xi += oi;
            }
        }

        // --- gated MLP block ---
        let xb = rmsnorm(&x, &layer.rms_ffn, 1, dim, c.norm_eps);
        let acts = quantize_acts(&xb);
        {
            let hidden = c.hidden_dim;
            let (w1, w3) = (&layer.w1, &layer.w3);
            prana_kernels::team_fill_rows_weighted(team, &scratch.gu, 2 * hidden, dim, |r| {
                if r < hidden {
                    w1.row_dot(&xb, &acts, r)
                } else {
                    w3.row_dot(&xb, &acts, r - hidden)
                }
            });
        }
        // Redundant per member: gate activation, elementwise product,
        // re-quantization for the down projection.
        let (hb, acts_h) = {
            let gu = scratch.gu.read();
            let mut g = gu[..c.hidden_dim].to_vec();
            match c.act {
                Activation::Silu => silu(&mut g),
                Activation::GeluTanh => gelu_tanh(&mut g),
            }
            for (gi, ui) in g.iter_mut().zip(gu[c.hidden_dim..].iter()) {
                *gi *= ui;
            }
            let acts_h = quantize_acts(&g);
            (g, acts_h)
        };
        layer.w2.apply_team(team, &hb, &acts_h, &scratch.o);
        {
            let o = scratch.o.read();
            for (xi, oi) in x.iter_mut().zip(o.iter()) {
                *xi += oi;
            }
        }
    }

    // --- classifier (skipped for teacher-forced prefill positions; the
    // condition is identical on every member, so barrier counts stay
    // uniform) ---
    if !need_logits {
        return;
    }
    let xb = rmsnorm(&x, &model.rms_final, 1, dim, c.norm_eps);
    let acts = quantize_acts(&xb);
    match (&model.wcls, &model.tok_emb) {
        (Some(cls), _) => cls.apply_team(team, &xb, &acts, &scratch.logits),
        (None, Embedding::F32(w)) => matmul_f32_team(team, &xb, w, c.vocab_size, &scratch.logits),
        (None, Embedding::KQuant(m)) => matmul_kquant_team(team, &acts, m, &scratch.logits),
    }
}

/// Statistics from a generation run.
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub seconds: f64,
    /// Speculative decoding only: (draft tokens proposed, tokens accepted).
    /// `accepted / proposed` is the acceptance rate; the speedup is roughly
    /// `(accepted + verify_passes) / verify_passes` capped by the cost ratio.
    pub spec: Option<(usize, usize)>,
}

/// Greedy argmax of one logits row.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in logits.iter().enumerate() {
        if *x > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Greedy **speculative decoding**: a small `draft` model proposes `k` tokens
/// per round, the `target` model verifies all `k` in a single batched pass
/// ([`forward_batch`]), and the longest correct prefix is accepted. Because
/// acceptance requires the draft token to equal the target's greedy argmax,
/// the emitted sequence is **exactly** the target's own greedy decode — just
/// produced in fewer target passes. Only greedy is supported (temperature
/// needs rejection sampling; documented as future work).
///
/// Both models must share a tokenizer/vocabulary. Returns the same
/// [`GenStats`] as [`generate`], with `spec` populated.
#[allow(clippy::too_many_arguments)]
pub fn generate_speculative(
    target: &Model,
    draft: &Model,
    tokenizer: &dyn Tokenize,
    prompt: &str,
    steps: usize,
    k: usize,
    mut on_piece: impl FnMut(&[u8], bool),
) -> GenStats {
    assert!(k >= 1, "draft length must be >= 1");
    assert_eq!(
        target.config.vocab_size, draft.config.vocab_size,
        "draft and target must share a vocabulary"
    );
    let prompt_tokens = tokenizer.encode_prompt(prompt);
    assert!(!prompt_tokens.is_empty(), "prompt encoded to zero tokens");
    let max_pos = target.config.seq_len.min(prompt_tokens.len() + steps);
    let p = prompt_tokens.len();

    let mut tcache = KvCache::with_len(target, max_pos);
    let mut dcache = KvCache::with_len(draft, max_pos);
    let start = std::time::Instant::now();

    // Prefill both caches with the prompt except its last token, streaming
    // the prompt pieces (hidden by chat UIs). After this, cache positions
    // 0..p-1 hold correct K/V, and `last` (the p-th prompt token, logical
    // position p-1) has not yet been fed through.
    if p > 1 {
        prefill(target, &mut tcache, &prompt_tokens[..p - 1]);
        prefill(draft, &mut dcache, &prompt_tokens[..p - 1]);
        for w in prompt_tokens.windows(2) {
            on_piece(&tokenizer.decode(w[0], w[1]), true);
        }
    }

    // Invariant at the top of each round: caches hold correct K/V for
    // positions 0..pos; `last` is the (uncached) token at logical position
    // `pos`. Every round writes cache positions starting at `pos`, so any
    // stale K/V left past a partial accept is always overwritten before it
    // is attended to.
    let mut pos = p - 1;
    let mut last = prompt_tokens[p - 1];
    let mut generated = 0usize;
    let (mut proposed, mut accepted) = (0usize, 0usize);

    'outer: while pos + 1 < max_pos {
        // --- draft: greedily propose up to `budget` tokens from `last` ---
        // Draft forward at logical position pos+j writes dcache[pos+j].
        let budget = (max_pos - 1 - pos).min(k);
        let mut draft_toks = Vec::with_capacity(budget);
        let mut d_tok = last;
        for j in 0..budget {
            let logits = forward(draft, &mut dcache, d_tok, pos + j);
            d_tok = argmax(&logits);
            draft_toks.push(d_tok);
        }
        proposed += draft_toks.len();

        // --- target: verify [last, draft_toks...] in one batched pass ---
        // Input row i sits at logical position pos+i; its argmax is the
        // target's greedy successor of that input. Row 0's successor is the
        // target's own next token; row i>0's is the successor of
        // draft_toks[i-1]. So target_next[i] should equal draft_toks[i] for
        // the draft to be accepted at step i.
        let mut inputs = Vec::with_capacity(draft_toks.len() + 1);
        inputs.push(last);
        inputs.extend_from_slice(&draft_toks);
        let logits = forward_batch(target, &mut tcache, &inputs, pos);
        let v = target.config.vocab_size;

        // Walk the predictions: accept while they match the draft, then emit
        // the first non-matching target token (the correction, or the bonus
        // token after a full accept) and start the next round from it.
        for i in 0..=draft_toks.len() {
            let target_next = argmax(&logits[i * v..(i + 1) * v]);

            if i < draft_toks.len() && target_next == draft_toks[i] {
                accepted += 1;
            }

            // Emit `target_next` as the token following logical position
            // pos+i (its K/V is already in tcache at pos+i, written by
            // forward_batch, and correct because inputs 0..=i were correct).
            if tokenizer.is_stop(target_next) {
                break 'outer;
            }
            on_piece(&tokenizer.decode(inputs[i], target_next), false);
            generated += 1;
            // Advance: the emitted token becomes `last` at logical pos+i+1.
            pos += 1;
            last = target_next;
            if generated >= steps || pos + 1 >= max_pos {
                break 'outer;
            }

            // If the draft diverged here, discard the rest of this batch and
            // re-draft from the corrected token.
            if i == draft_toks.len() || target_next != draft_toks[i] {
                break;
            }
        }
    }

    GenStats {
        prompt_tokens: p,
        generated_tokens: generated,
        seconds: start.elapsed().as_secs_f64(),
        spec: Some((proposed, accepted)),
    }
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

    // Batch-prefill all but the last prompt token in one pass (the decode
    // loop re-enters at the last one and produces the first logits). The
    // per-token path is kept for PRANA_TIMING so the profiler sees the
    // whole model.
    let p = prompt_tokens.len();
    let mut start_pos = 0;
    if p > 1 && p <= max_pos && !timing_enabled() {
        prefill(model, &mut cache, &prompt_tokens[..p - 1]);
        for w in prompt_tokens.windows(2) {
            on_piece(&tokenizer.decode(w[0], w[1]), true);
        }
        start_pos = p - 1;
        token = prompt_tokens[p - 1];
    }

    for pos in start_pos..max_pos {
        // Teacher-forced prefill positions never read their logits, so the
        // LM head (the model's single largest matmul) is skipped for them —
        // llama.cpp's `inp_out_ids` trick.
        let need_logits = pos + 1 >= prompt_tokens.len() && pos + 1 < max_pos;
        // Measured on Ultra-5-class hybrid Windows hardware, the per-op pool
        // path outruns team execution (~29 vs ~21 tok/s on a 0.5B q4_0):
        // both leave the small QKV/O projections serial, but the team's
        // members busy-spin through those sections and through every barrier
        // tail, which eats the power/SMT headroom the working threads need.
        // Team execution (PRANA_TEAM=1) stays as parity-tested groundwork —
        // its barrier costs measure healthy (~1-2µs), so it should win once
        // the serial glue itself is parallelized. See EVALUATION.md.
        let logits = if team_enabled() {
            forward_team_impl(model, &cache, &scratch, token, pos, need_logits)
        } else {
            forward_impl(model, &mut cache, token, pos, need_logits)
        };
        if pos + 1 >= max_pos {
            break; // context full: the next token would have nowhere to sit
        }
        let next = if pos + 1 < prompt_tokens.len() {
            prompt_tokens[pos + 1] // teacher-force the rest of the prompt
        } else {
            sampler.sample(&logits.expect("logits computed for sampled positions"))
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
        spec: None,
    }
}
