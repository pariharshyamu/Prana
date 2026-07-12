//! Loader for the `llama2.c` checkpoint format — the simplest real-model
//! container in circulation: a 7-int32 header followed by raw little-endian
//! f32 weight tensors in a fixed order. Karpathy's `stories*` TinyStories
//! models (15M/42M/110M) all ship in it, which makes it the ideal target for
//! proving Prana runs an *actual trained model*, not synthetic weights.
//!
//! Weights can optionally be re-quantized to Prana's Q8 block format at load
//! (`Precision::Q8`), exercising the quantized matmul path end-to-end with
//! real weights.

use std::fs;
use std::io;
use std::path::Path;

use prana_kernels::{matmul_f32, matmul_q8_f32, quantize_q8, QuantMatrix, Q8_BLOCK};

/// Model hyperparameters, from the checkpoint header.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// Classifier shares the token-embedding matrix (header vocab_size > 0).
    pub shared_classifier: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }
}

/// Weight precision selected at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    /// Re-quantize all projection matrices to Q8 blocks at load.
    Q8,
}

/// One projection matrix, dense or quantized, behind a single `apply`.
pub enum Linear {
    F32 { w: Vec<f32>, rows: usize, cols: usize },
    Q8(QuantMatrix),
}

impl Linear {
    fn new(w: Vec<f32>, rows: usize, cols: usize, precision: Precision) -> Self {
        match precision {
            // Q8 needs cols % block == 0; fall back to dense for odd shapes.
            Precision::Q8 if cols.is_multiple_of(Q8_BLOCK) => Linear::Q8(quantize_q8(rows, cols, &w)),
            _ => Linear::F32 { w, rows, cols },
        }
    }

    /// `y = W x` for a single activation row `x` of length `cols`.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Linear::F32 { w, rows, cols } => matmul_f32(x, w, 1, *cols, *rows),
            Linear::Q8(qm) => matmul_q8_f32(x, qm, 1),
        }
    }

    /// Bytes of weight storage.
    pub fn stored_bytes(&self) -> usize {
        match self {
            Linear::F32 { w, .. } => w.len() * 4,
            Linear::Q8(qm) => qm.stored_bytes(),
        }
    }
}

/// Per-layer weights.
pub struct Layer {
    pub rms_att: Vec<f32>,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub rms_ffn: Vec<f32>,
    pub w1: Linear, // gate
    pub w2: Linear, // down
    pub w3: Linear, // up
}

/// A loaded model.
pub struct Model {
    pub config: Config,
    /// `[vocab_size, dim]` token embedding table, always f32 (row lookups).
    pub tok_emb: Vec<f32>,
    pub layers: Vec<Layer>,
    pub rms_final: Vec<f32>,
    /// LM head; `None` means classifier shares `tok_emb`.
    pub wcls: Option<Linear>,
}

impl Model {
    /// Total bytes of projection-weight storage (the part precision changes).
    pub fn projection_bytes(&self) -> usize {
        let per_layer: usize = self
            .layers
            .iter()
            .map(|l| {
                l.wq.stored_bytes()
                    + l.wk.stored_bytes()
                    + l.wv.stored_bytes()
                    + l.wo.stored_bytes()
                    + l.w1.stored_bytes()
                    + l.w2.stored_bytes()
                    + l.w3.stored_bytes()
            })
            .sum();
        per_layer + self.wcls.as_ref().map_or(0, |c| c.stored_bytes())
    }

    /// Compute vocabulary logits for the final hidden state.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        match &self.wcls {
            Some(c) => c.apply(x),
            None => matmul_f32(x, &self.tok_emb, 1, self.config.dim, self.config.vocab_size),
        }
    }
}

/// Sequential little-endian reader over the checkpoint bytes.
struct Cursor<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn i32(&mut self) -> io::Result<i32> {
        let b = self
            .data
            .get(self.off..self.off + 4)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "checkpoint truncated"))?;
        self.off += 4;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32s(&mut self, n: usize) -> io::Result<Vec<f32>> {
        let bytes = n * 4;
        let b = self
            .data
            .get(self.off..self.off + bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "checkpoint truncated"))?;
        self.off += bytes;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Load a llama2.c checkpoint. Weight order matches `run.c`:
/// tok_emb, rms_att×L, wq×L, wk×L, wv×L, wo×L, rms_ffn×L, w1×L, w2×L, w3×L,
/// rms_final, (skipped rope tables), [wcls if unshared].
pub fn load(path: &Path, precision: Precision) -> io::Result<Model> {
    let data = fs::read(path)?;
    load_bytes(&data, precision)
}

/// Load from in-memory bytes (also lets tests build synthetic checkpoints).
pub fn load_bytes(data: &[u8], precision: Precision) -> io::Result<Model> {
    let mut cur = Cursor { data, off: 0 };
    let dim = cur.i32()? as usize;
    let hidden_dim = cur.i32()? as usize;
    let n_layers = cur.i32()? as usize;
    let n_heads = cur.i32()? as usize;
    let n_kv_heads = cur.i32()? as usize;
    let vocab_raw = cur.i32()?;
    let seq_len = cur.i32()? as usize;

    let shared_classifier = vocab_raw > 0;
    let vocab_size = vocab_raw.unsigned_abs() as usize;
    let config = Config {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        seq_len,
        shared_classifier,
    };
    if dim == 0 || n_heads == 0 || !dim.is_multiple_of(n_heads) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad checkpoint header"));
    }
    let head_dim = config.head_dim();
    let kv_dim = config.kv_dim();

    let tok_emb = cur.f32s(vocab_size * dim)?;

    // Weights are stored grouped by kind across layers, not by layer.
    let rms_att: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(dim)).collect::<io::Result<_>>()?;
    let wq: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(dim * n_heads * head_dim)).collect::<io::Result<_>>()?;
    let wk: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(dim * kv_dim)).collect::<io::Result<_>>()?;
    let wv: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(dim * kv_dim)).collect::<io::Result<_>>()?;
    let wo: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(n_heads * head_dim * dim)).collect::<io::Result<_>>()?;
    let rms_ffn: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(dim)).collect::<io::Result<_>>()?;
    let w1: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(hidden_dim * dim)).collect::<io::Result<_>>()?;
    let w2: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(dim * hidden_dim)).collect::<io::Result<_>>()?;
    let w3: Vec<Vec<f32>> = (0..n_layers).map(|_| cur.f32s(hidden_dim * dim)).collect::<io::Result<_>>()?;
    let rms_final = cur.f32s(dim)?;

    // Skip the precomputed RoPE tables (freq_cis_real/imag); we compute RoPE.
    let _ = cur.f32s(seq_len * head_dim / 2 * 2)?;

    let wcls = if shared_classifier {
        None
    } else {
        Some(Linear::new(cur.f32s(vocab_size * dim)?, vocab_size, dim, precision))
    };
    // Shared classifier + Q8: quantize a copy of the embedding table so the
    // big vocab matmul also runs the quantized path.
    let wcls = match (&wcls, precision) {
        (None, Precision::Q8) if dim.is_multiple_of(Q8_BLOCK) => {
            Some(Linear::new(tok_emb.clone(), vocab_size, dim, Precision::Q8))
        }
        _ => wcls,
    };

    let (mut rms_att, mut wq, mut wk, mut wv) = (rms_att.into_iter(), wq.into_iter(), wk.into_iter(), wv.into_iter());
    let (mut wo, mut rms_ffn, mut w1, mut w2, mut w3) =
        (wo.into_iter(), rms_ffn.into_iter(), w1.into_iter(), w2.into_iter(), w3.into_iter());
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(Layer {
            rms_att: rms_att.next().unwrap(),
            wq: Linear::new(wq.next().unwrap(), n_heads * head_dim, dim, precision),
            wk: Linear::new(wk.next().unwrap(), kv_dim, dim, precision),
            wv: Linear::new(wv.next().unwrap(), kv_dim, dim, precision),
            wo: Linear::new(wo.next().unwrap(), dim, n_heads * head_dim, precision),
            rms_ffn: rms_ffn.next().unwrap(),
            w1: Linear::new(w1.next().unwrap(), hidden_dim, dim, precision),
            w2: Linear::new(w2.next().unwrap(), dim, hidden_dim, precision),
            w3: Linear::new(w3.next().unwrap(), hidden_dim, dim, precision),
        });
    }

    Ok(Model { config, tok_emb, layers, rms_final, wcls })
}
