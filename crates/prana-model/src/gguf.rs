//! GGUF loader — the container format the modern local-model ecosystem
//! (llama.cpp, Ollama, LM Studio) distributes weights in.
//!
//! Supports GGUF v2/v3 little-endian files with F32, F16, and Q8_0 tensors,
//! the `llama` architecture metadata keys, and the embedded SentencePiece
//! tokenizer (`tokenizer.ggml.tokens` / `.scores`). Q8_0 blocks (32 × i8 +
//! one f16 scale) map 1:1 onto Prana's `QuantMatrix` layout, so quantized
//! GGUF tensors are used *natively* — no dequantize-requantize round trip.
//!
//! GGUF llama models use the interleaved (adjacent-pair) RoPE convention
//! (ggml `ROPE_TYPE_NORM`; HF-permuted checkpoints are un-permuted by the
//! conversion scripts), which is exactly `rope_interleaved` — the same
//! convention as llama2.c checkpoints.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use prana_kernels::{
    dequant_q4k_block, dequant_q6k_block, KQuantKind, KQuantMatrix, QuantMatrix,
    Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_BLOCK, QK_K,
};
pub use prana_kernels::f16_to_f32;

use crate::checkpoint::{Activation, Config, Layer, Linear, Model, Precision};
use crate::tokenizer::{AnyTokenizer, Tokenizer};

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

// ggml tensor dtypes we support.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;
const GGML_Q6_K: u32 = 14;


fn err(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A parsed metadata value. Only the shapes the llama keys use get accessors.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(Vec<u8>),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    fn as_usize(&self) -> Option<usize> {
        match *self {
            Value::U8(v) => Some(v as usize),
            Value::U16(v) => Some(v as usize),
            Value::U32(v) => Some(v as usize),
            Value::U64(v) => usize::try_from(v).ok(),
            Value::I8(v) if v >= 0 => Some(v as usize),
            Value::I16(v) if v >= 0 => Some(v as usize),
            Value::I32(v) if v >= 0 => Some(v as usize),
            Value::I64(v) if v >= 0 => usize::try_from(v).ok(),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let b = self.data.get(self.off..self.off + n).ok_or_else(|| err("gguf truncated"))?;
        self.off += n;
        Ok(b)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn str_(&mut self) -> io::Result<Vec<u8>> {
        let n = self.u64()? as usize;
        Ok(self.bytes(n)?.to_vec())
    }

    fn value(&mut self, ty: u32) -> io::Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.bytes(1)?[0]),
            1 => Value::I8(self.bytes(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap())),
            7 => Value::Bool(self.bytes(1)?[0] != 0),
            8 => Value::Str(self.str_()?),
            9 => {
                let elem_ty = self.u32()?;
                let count = self.u64()? as usize;
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(self.value(elem_ty)?);
                }
                Value::Arr(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap())),
            t => return Err(err(format!("unsupported gguf value type {t}"))),
        })
    }
}

struct TensorInfo {
    /// ggml order: dims[0] is the contiguous (column) dimension.
    dims: Vec<usize>,
    ggml_type: u32,
    offset: usize,
}

/// The parsed file: metadata + tensor directory + the raw data section.
pub struct Gguf {
    pub metadata: HashMap<String, Value>,
    tensors: HashMap<String, TensorInfo>,
    data: Vec<u8>,
    data_start: usize,
}

impl Gguf {
    pub fn read(path: &Path) -> io::Result<Self> {
        Self::from_bytes(fs::read(path)?)
    }

    pub fn from_bytes(data: Vec<u8>) -> io::Result<Self> {
        let mut r = Reader { data: &data, off: 0 };
        if r.u32()? != MAGIC {
            return Err(err("not a GGUF file (bad magic)"));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(err(format!("unsupported GGUF version {version}")));
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = String::from_utf8_lossy(&r.str_()?).into_owned();
            let ty = r.u32()?;
            metadata.insert(key, r.value(ty)?);
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = String::from_utf8_lossy(&r.str_()?).into_owned();
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()? as usize);
            }
            let ggml_type = r.u32()?;
            let offset = r.u64()? as usize;
            tensors.insert(name, TensorInfo { dims, ggml_type, offset });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_usize)
            .filter(|&a| a.is_power_of_two())
            .unwrap_or(32);
        let data_start = r.off.div_ceil(alignment) * alignment;
        Ok(Self { metadata, tensors, data, data_start })
    }

    fn info(&self, name: &str) -> io::Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| err(format!("missing tensor '{name}'")))
    }

    fn raw(&self, info: &TensorInfo, bytes: usize) -> io::Result<&[u8]> {
        let start = self.data_start + info.offset;
        self.data.get(start..start + bytes).ok_or_else(|| err("tensor data out of range"))
    }

    /// Read a tensor as f32, whatever its storage type.
    pub fn tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let info = self.info(name)?;
        let n: usize = info.dims.iter().product();
        match info.ggml_type {
            GGML_F32 => Ok(self
                .raw(info, n * 4)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()),
            GGML_F16 => Ok(self
                .raw(info, n * 2)?
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect()),
            GGML_Q8_0 => {
                if !n.is_multiple_of(Q8_BLOCK) {
                    return Err(err(format!("Q8_0 tensor '{name}' not block-aligned")));
                }
                let blocks = n / Q8_BLOCK;
                let raw = self.raw(info, blocks * 34)?; // f16 scale + 32 i8
                let mut out = Vec::with_capacity(n);
                for b in raw.chunks_exact(34) {
                    let scale = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                    out.extend(b[2..].iter().map(|&q| q as i8 as f32 * scale));
                }
                Ok(out)
            }
            GGML_Q4_K | GGML_Q6_K => {
                let ty = info.ggml_type;
                if !n.is_multiple_of(QK_K) {
                    return Err(err(format!("K-quant tensor '{name}' not {QK_K}-aligned")));
                }
                let blocks = n / QK_K;
                let block_bytes = if ty == GGML_Q4_K { Q4_K_BLOCK_BYTES } else { Q6_K_BLOCK_BYTES };
                let raw = self.raw(info, blocks * block_bytes)?;
                let mut out = Vec::with_capacity(n);
                for b in raw.chunks_exact(block_bytes) {
                    if ty == GGML_Q4_K {
                        dequant_q4k_block(b, &mut out);
                    } else {
                        dequant_q6k_block(b, &mut out);
                    }
                }
                Ok(out)
            }
            t => Err(err(format!("tensor '{name}': unsupported ggml type {t}"))),
        }
    }

    /// Read a 2-D tensor as a `Linear` at the requested precision. ggml dims
    /// are `[cols, rows]` (dims[0] contiguous), matching our row-major layout.
    fn linear(&self, name: &str, precision: Precision) -> io::Result<Linear> {
        let info = self.info(name)?;
        if info.dims.len() != 2 {
            return Err(err(format!("tensor '{name}' is not 2-D")));
        }
        let (cols, rows) = (info.dims[0], info.dims[1]);
        // Q8_0 storage maps directly onto QuantMatrix: keep it quantized
        // regardless of requested precision (dequantizing regains nothing).
        if info.ggml_type == GGML_Q8_0 && cols.is_multiple_of(Q8_BLOCK) {
            let blocks = rows * cols / Q8_BLOCK;
            let raw = self.raw(info, blocks * 34)?;
            let mut q = Vec::with_capacity(rows * cols);
            let mut scales = Vec::with_capacity(blocks);
            for b in raw.chunks_exact(34) {
                scales.push(f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
                q.extend(b[2..].iter().map(|&x| x as i8));
            }
            return Ok(Linear::Q8(QuantMatrix::from_parts(rows, cols, q, scales)));
        }
        // K-quant tensors likewise stay packed and run on the native
        // K-quant matmul (4.5 / 6.6 bits per weight in RAM).
        if matches!(info.ggml_type, GGML_Q4_K | GGML_Q6_K) && cols.is_multiple_of(QK_K) {
            let kind = if info.ggml_type == GGML_Q4_K { KQuantKind::Q4K } else { KQuantKind::Q6K };
            let blocks = rows * cols / QK_K;
            let raw = self.raw(info, blocks * kind.block_bytes())?.to_vec();
            return Ok(Linear::KQuant(KQuantMatrix::from_raw(rows, cols, kind, raw)));
        }
        Ok(Linear::new(self.tensor_f32(name)?, rows, cols, precision))
    }

    fn meta_usize(&self, key: &str) -> io::Result<usize> {
        self.metadata
            .get(key)
            .and_then(Value::as_usize)
            .ok_or_else(|| err(format!("missing metadata '{key}'")))
    }
}

/// Load a GGUF model (llama, qwen2, or gemma architecture) plus its
/// embedded tokenizer.
pub fn load(path: &Path, precision: Precision) -> io::Result<(Model, AnyTokenizer)> {
    let g = Gguf::read(path)?;

    let arch = match g.metadata.get("general.architecture") {
        Some(Value::Str(s)) => String::from_utf8_lossy(s).into_owned(),
        _ => return Err(err("missing general.architecture")),
    };
    if !matches!(arch.as_str(), "llama" | "qwen2" | "gemma") {
        return Err(err(format!("unsupported architecture '{arch}' (llama, qwen2, gemma)")));
    }
    let is_gemma = arch == "gemma";

    let key = |suffix: &str| format!("{arch}.{suffix}");
    let dim = g.meta_usize(&key("embedding_length"))?;
    let n_layers = g.meta_usize(&key("block_count"))?;
    let n_heads = g.meta_usize(&key("attention.head_count"))?;
    let n_kv_heads = g
        .metadata
        .get(&key("attention.head_count_kv"))
        .and_then(Value::as_usize)
        .unwrap_or(n_heads);
    let hidden_dim = g.meta_usize(&key("feed_forward_length"))?;
    let seq_len = g.meta_usize(&key("context_length"))?;
    let rope_theta = g
        .metadata
        .get(&key("rope.freq_base"))
        .and_then(Value::as_f32)
        .unwrap_or(10000.0);
    // Gemma decouples per-head width from dim/n_heads.
    let head_dim = g
        .metadata
        .get(&key("attention.key_length"))
        .and_then(Value::as_usize)
        .unwrap_or(dim / n_heads);
    let norm_eps = g
        .metadata
        .get(&key("attention.layer_norm_rms_epsilon"))
        .and_then(Value::as_f32)
        .unwrap_or(1e-5);

    let tokenizer = load_tokenizer(&g)?;
    let vocab_size = tokenizer.vocab_size();

    let mut tok_emb = g.tensor_f32("token_embd.weight")?;
    if tok_emb.len() != vocab_size * dim {
        return Err(err("token_embd.weight size mismatch with vocab"));
    }
    // llama.cpp stores gemma norm weights as (w - 1) relative to their
    // effective value: the model computes rmsnorm(x) * (1 + w). Fold the +1
    // in at load so the runtime norm stays uniform across architectures.
    let fold_one = |mut v: Vec<f32>| -> Vec<f32> {
        if is_gemma {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        v
    };

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let t = |suffix: &str| format!("blk.{i}.{suffix}");
        let bias = |name: &str| -> io::Result<Option<Vec<f32>>> {
            if g.tensors.contains_key(name) {
                Ok(Some(g.tensor_f32(name)?))
            } else {
                Ok(None)
            }
        };
        layers.push(Layer {
            rms_att: fold_one(g.tensor_f32(&t("attn_norm.weight"))?),
            wq: g.linear(&t("attn_q.weight"), precision)?,
            wk: g.linear(&t("attn_k.weight"), precision)?,
            wv: g.linear(&t("attn_v.weight"), precision)?,
            wo: g.linear(&t("attn_output.weight"), precision)?,
            bq: bias(&t("attn_q.bias"))?,
            bk: bias(&t("attn_k.bias"))?,
            bv: bias(&t("attn_v.bias"))?,
            rms_ffn: fold_one(g.tensor_f32(&t("ffn_norm.weight"))?),
            w1: g.linear(&t("ffn_gate.weight"), precision)?,
            w2: g.linear(&t("ffn_down.weight"), precision)?,
            w3: g.linear(&t("ffn_up.weight"), precision)?,
        });
    }

    let rms_final = fold_one(g.tensor_f32("output_norm.weight")?);

    // Tied-embedding models omit output.weight.
    let shared_classifier = !g.tensors.contains_key("output.weight");
    let wcls = if shared_classifier {
        match precision {
            Precision::Q8 if dim.is_multiple_of(Q8_BLOCK) => {
                Some(Linear::new(tok_emb.clone(), vocab_size, dim, Precision::Q8))
            }
            _ => None,
        }
    } else {
        Some(g.linear("output.weight", precision)?)
    };

    let config = Config {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        seq_len,
        shared_classifier,
        rope_theta,
        head_dim,
        norm_eps,
        // llama GGUFs are stored un-permuted for interleaved rope (NORM);
        // qwen2/gemma use the NeoX rotate_half convention.
        rope_neox: !matches!(arch.as_str(), "llama"),
        act: if is_gemma { Activation::GeluTanh } else { Activation::Silu },
        emb_scale: if is_gemma { (dim as f32).sqrt() } else { 1.0 },
    };
    if is_gemma {
        // Gemma ties the classifier and scales embeddings; nothing extra to
        // do here (scale applies at lookup), but keep tok_emb unscaled.
        let _ = &mut tok_emb;
    }
    Ok((Model { config, tok_emb, layers, rms_final, wcls }, tokenizer))
}

/// Build the right tokenizer from GGUF metadata: SentencePiece ("llama") or
/// byte-level BPE with merges ("gpt2", used by Qwen2).
fn load_tokenizer(g: &Gguf) -> io::Result<AnyTokenizer> {
    let tokens = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Arr(items)) => items,
        _ => return Err(err("missing tokenizer.ggml.tokens")),
    };
    let model = match g.metadata.get("tokenizer.ggml.model") {
        Some(Value::Str(s)) => String::from_utf8_lossy(s).into_owned(),
        _ => "llama".to_string(),
    };
    let bos = g.metadata.get("tokenizer.ggml.bos_token_id").and_then(Value::as_usize);
    let eos = g.metadata.get("tokenizer.ggml.eos_token_id").and_then(Value::as_usize);
    let add_bos = match g.metadata.get("tokenizer.ggml.add_bos_token") {
        Some(Value::Bool(b)) => *b,
        _ => model == "llama", // SPM models default to BOS, BPE models to none
    };

    match model.as_str() {
        "llama" => {
            // SentencePiece: pieces carry U+2581 word markers; ours use spaces.
            let vocab: Vec<Vec<u8>> = tokens
                .iter()
                .map(|v| match v {
                    Value::Str(s) => {
                        String::from_utf8_lossy(s).replace('\u{2581}', " ").into_bytes()
                    }
                    _ => Vec::new(),
                })
                .collect();
            let scores: Vec<f32> = match g.metadata.get("tokenizer.ggml.scores") {
                Some(Value::Arr(items)) => items.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect(),
                _ => vec![0.0; vocab.len()],
            };
            let tok = Tokenizer::from_parts(vocab, scores).with_special(
                bos.unwrap_or(1) as u32,
                eos.unwrap_or(2) as u32,
                add_bos,
            );
            Ok(AnyTokenizer::Spm(Box::new(tok)))
        }
        "gpt2" => {
            let vocab: Vec<String> = tokens
                .iter()
                .map(|v| match v {
                    Value::Str(s) => String::from_utf8_lossy(s).into_owned(),
                    _ => String::new(),
                })
                .collect();
            let merges: Vec<String> = match g.metadata.get("tokenizer.ggml.merges") {
                Some(Value::Arr(items)) => items
                    .iter()
                    .map(|v| match v {
                        Value::Str(s) => String::from_utf8_lossy(s).into_owned(),
                        _ => String::new(),
                    })
                    .collect(),
                _ => return Err(err("gpt2 tokenizer requires tokenizer.ggml.merges")),
            };
            let tok = crate::bpe::BpeTokenizer::new(vocab, &merges, bos.map(|v| v as u32), eos.map(|v| v as u32), add_bos);
            Ok(AnyTokenizer::Bpe(Box::new(tok)))
        }
        other => Err(err(format!("unsupported tokenizer.ggml.model '{other}'"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prana_kernels::q4k_scale_min;

    /// Minimal GGUF writer for tests.
    struct W(Vec<u8>);
    impl W {
        fn str_(&mut self, s: &[u8]) {
            self.0.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.0.extend_from_slice(s);
        }
        fn kv_u32(&mut self, k: &str, v: u32) {
            self.str_(k.as_bytes());
            self.0.extend_from_slice(&4u32.to_le_bytes());
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        fn kv_str(&mut self, k: &str, v: &str) {
            self.str_(k.as_bytes());
            self.0.extend_from_slice(&8u32.to_le_bytes());
            self.str_(v.as_bytes());
        }
    }

    /// Build a complete tiny model file for an architecture, with an embedded
    /// SPM tokenizer (arch mechanics and tokenizer family are orthogonal —
    /// the BPE tokenizer has its own unit tests in `bpe.rs`).
    fn synthetic_model_gguf(arch: &str, with_biases: bool, head_dim: usize) -> Vec<u8> {
        let (dim, hidden, n_layers, n_heads, vocab, ctx) = (32usize, 64usize, 2usize, 2usize, 48usize, 16usize);
        let q_dim = n_heads * head_dim;
        let kv_dim = q_dim; // MHA in the test

        let mut rng_state = 0xABCDEF12345u64;
        let mut f32s = |n: usize, scale: f32| -> Vec<u8> {
            let mut out = Vec::with_capacity(n * 4);
            for _ in 0..n {
                rng_state ^= rng_state >> 12;
                rng_state ^= rng_state << 25;
                rng_state ^= rng_state >> 27;
                let r = ((rng_state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                out.extend_from_slice(&(r * scale).to_le_bytes());
            }
            out
        };

        // (name, ne, data)
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        tensors.push(("token_embd.weight".into(), vec![dim, vocab], f32s(vocab * dim, 0.1)));
        for i in 0..n_layers {
            tensors.push((format!("blk.{i}.attn_norm.weight"), vec![dim], f32s(dim, 0.5)));
            tensors.push((format!("blk.{i}.attn_q.weight"), vec![dim, q_dim], f32s(q_dim * dim, 0.1)));
            tensors.push((format!("blk.{i}.attn_k.weight"), vec![dim, kv_dim], f32s(kv_dim * dim, 0.1)));
            tensors.push((format!("blk.{i}.attn_v.weight"), vec![dim, kv_dim], f32s(kv_dim * dim, 0.1)));
            tensors.push((format!("blk.{i}.attn_output.weight"), vec![q_dim, dim], f32s(dim * q_dim, 0.1)));
            if with_biases {
                tensors.push((format!("blk.{i}.attn_q.bias"), vec![q_dim], f32s(q_dim, 0.05)));
                tensors.push((format!("blk.{i}.attn_k.bias"), vec![kv_dim], f32s(kv_dim, 0.05)));
                tensors.push((format!("blk.{i}.attn_v.bias"), vec![kv_dim], f32s(kv_dim, 0.05)));
            }
            tensors.push((format!("blk.{i}.ffn_norm.weight"), vec![dim], f32s(dim, 0.5)));
            tensors.push((format!("blk.{i}.ffn_gate.weight"), vec![dim, hidden], f32s(hidden * dim, 0.1)));
            tensors.push((format!("blk.{i}.ffn_down.weight"), vec![hidden, dim], f32s(dim * hidden, 0.1)));
            tensors.push((format!("blk.{i}.ffn_up.weight"), vec![dim, hidden], f32s(hidden * dim, 0.1)));
        }
        tensors.push(("output_norm.weight".into(), vec![dim], f32s(dim, 0.5)));

        // --- header + metadata ---
        let mut w = W(Vec::new());
        w.0.extend_from_slice(&MAGIC.to_le_bytes());
        w.0.extend_from_slice(&3u32.to_le_bytes());
        w.0.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        w.0.extend_from_slice(&11u64.to_le_bytes()); // kv count below
        w.kv_str("general.architecture", arch);
        w.kv_u32(&format!("{arch}.embedding_length"), dim as u32);
        w.kv_u32(&format!("{arch}.block_count"), n_layers as u32);
        w.kv_u32(&format!("{arch}.attention.head_count"), n_heads as u32);
        w.kv_u32(&format!("{arch}.attention.head_count_kv"), n_heads as u32);
        w.kv_u32(&format!("{arch}.feed_forward_length"), hidden as u32);
        w.kv_u32(&format!("{arch}.context_length"), ctx as u32);
        w.kv_u32(&format!("{arch}.attention.key_length"), head_dim as u32);
        w.kv_str("tokenizer.ggml.model", "llama");
        // tokens: specials + ascii singles
        w.str_(b"tokenizer.ggml.tokens");
        w.0.extend_from_slice(&9u32.to_le_bytes()); // type arr
        w.0.extend_from_slice(&8u32.to_le_bytes()); // elem type str
        w.0.extend_from_slice(&(vocab as u64).to_le_bytes());
        for i in 0..vocab as u8 {
            let piece: Vec<u8> = match i {
                0 => b"<unk>".to_vec(),
                1 => b"<s>".to_vec(),
                2 => b"</s>".to_vec(),
                3 => b" ".to_vec(),
                i => vec![b'a' + (i - 4) % 26],
            };
            w.str_(&piece);
        }
        w.str_(b"tokenizer.ggml.scores");
        w.0.extend_from_slice(&9u32.to_le_bytes());
        w.0.extend_from_slice(&6u32.to_le_bytes()); // elem type f32
        w.0.extend_from_slice(&(vocab as u64).to_le_bytes());
        for _ in 0..vocab {
            w.0.extend_from_slice(&(-1.0f32).to_le_bytes());
        }

        // --- tensor directory + data ---
        let mut data_off = 0usize;
        for (name, ne, data) in &tensors {
            w.str_(name.as_bytes());
            w.0.extend_from_slice(&(ne.len() as u32).to_le_bytes());
            for d in ne {
                w.0.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            w.0.extend_from_slice(&GGML_F32.to_le_bytes());
            w.0.extend_from_slice(&(data_off as u64).to_le_bytes());
            data_off = (data_off + data.len()).div_ceil(32) * 32;
        }
        while !w.0.len().is_multiple_of(32) {
            w.0.push(0);
        }
        for (_, _, data) in &tensors {
            while !w.0.len().is_multiple_of(32) {
                w.0.push(0);
            }
            w.0.extend_from_slice(data);
        }
        w.0
    }

    fn load_from_bytes(bytes: Vec<u8>) -> (Model, AnyTokenizer) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0); // tests run in parallel
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("prana-test-{}-{n}.gguf", std::process::id()));
        std::fs::write(&dir, bytes).unwrap();
        let out = load(&dir, Precision::F32).unwrap();
        let _ = std::fs::remove_file(&dir);
        out
    }

    #[test]
    fn qwen2_arch_loads_with_biases_and_runs() {
        let (model, tok) = load_from_bytes(synthetic_model_gguf("qwen2", true, 16));
        assert!(model.config.rope_neox, "qwen2 must use NeoX rope");
        assert_eq!(model.config.act, Activation::Silu);
        assert_eq!(model.config.emb_scale, 1.0);
        assert!(model.layers[0].bq.is_some(), "qwen2 QKV biases must load");
        assert_eq!(tok.vocab_size(), 48);

        let mut cache = crate::KvCache::new(&model);
        let a = crate::forward(&model, &mut cache, 5, 0);
        let b = crate::forward(&model, &mut cache, 5, 1);
        assert_eq!(a.len(), 48);
        assert!(a.iter().chain(&b).all(|v| v.is_finite()));
        assert_ne!(a, b);
    }

    #[test]
    fn qwen2_biases_change_the_output() {
        let (with_bias, _) = load_from_bytes(synthetic_model_gguf("qwen2", true, 16));
        let (no_bias, _) = load_from_bytes(synthetic_model_gguf("qwen2", false, 16));
        let mut c1 = crate::KvCache::new(&with_bias);
        let mut c2 = crate::KvCache::new(&no_bias);
        let a = crate::forward(&with_bias, &mut c1, 7, 0);
        let b = crate::forward(&no_bias, &mut c2, 7, 0);
        assert_ne!(a, b, "bias tensors must affect logits");
    }

    #[test]
    fn gemma_arch_gets_its_knobs_and_decoupled_head_dim() {
        // head_dim 8 with dim=32, heads=2 -> q_dim 16 != dim, like real Gemma.
        let (model, _) = load_from_bytes(synthetic_model_gguf("gemma", false, 8));
        assert!(model.config.rope_neox);
        assert_eq!(model.config.act, Activation::GeluTanh);
        assert_eq!(model.config.head_dim, 8);
        assert_eq!(model.config.q_dim(), 16);
        assert!((model.config.emb_scale - (32f32).sqrt()).abs() < 1e-6);

        let mut cache = crate::KvCache::new(&model);
        for pos in 0..4 {
            let logits = crate::forward(&model, &mut cache, (pos + 3) as u32, pos);
            assert!(logits.iter().all(|v| v.is_finite()), "pos {pos}");
        }
    }

    #[test]
    fn gemma_norm_weights_are_folded_plus_one() {
        // The raw file stores rms weights ~N(0, 0.5); after folding they must
        // center near 1.0. Compare against the llama load of identical bytes.
        let (gemma, _) = load_from_bytes(synthetic_model_gguf("gemma", false, 16));
        let (llama, _) = load_from_bytes(synthetic_model_gguf("llama", false, 16));
        let g_mean: f32 = gemma.rms_final.iter().sum::<f32>() / gemma.rms_final.len() as f32;
        let l_mean: f32 = llama.rms_final.iter().sum::<f32>() / llama.rms_final.len() as f32;
        assert!((g_mean - (l_mean + 1.0)).abs() < 1e-5, "gemma {g_mean} vs llama {l_mean}");
    }

    #[test]
    fn f16_conversion_covers_the_cases() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        // subnormal: smallest positive half = 2^-24
        assert!((f16_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-12);
        assert!((f16_to_f32(0x3555) - 0.333_25).abs() < 1e-4);
    }

    #[test]
    fn parses_metadata_and_f32_tensor() {
        let mut w = W(Vec::new());
        w.0.extend_from_slice(&MAGIC.to_le_bytes());
        w.0.extend_from_slice(&3u32.to_le_bytes()); // version
        w.0.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        w.0.extend_from_slice(&2u64.to_le_bytes()); // 2 kv
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 6);
        // tensor info: "t" [2 x 3] f32 at offset 0
        w.str_(b"t");
        w.0.extend_from_slice(&2u32.to_le_bytes());
        w.0.extend_from_slice(&2u64.to_le_bytes()); // dims[0]=cols=2
        w.0.extend_from_slice(&3u64.to_le_bytes()); // dims[1]=rows=3
        w.0.extend_from_slice(&GGML_F32.to_le_bytes());
        w.0.extend_from_slice(&0u64.to_le_bytes());
        // align to 32 and write 6 f32s
        while !w.0.len().is_multiple_of(32) {
            w.0.push(0);
        }
        for i in 0..6 {
            w.0.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let g = Gguf::from_bytes(w.0).unwrap();
        assert_eq!(g.meta_usize("llama.block_count").unwrap(), 6);
        assert_eq!(g.tensor_f32("t").unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(g.tensor_f32("nope").is_err());
    }

    #[test]
    fn q4k_block_dequantizes_known_values() {
        // Craft a super-block with d=1.0, dmin=0.0 so y = sc[group] * nibble.
        // 6-bit scales: groups 0..3 live in scales[0..4] (low 6 bits).
        let mut b = vec![0u8; Q4_K_BLOCK_BYTES];
        b[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        b[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0
        b[4] = 2; // group 0 scale = 2
        b[5] = 3; // group 1 scale = 3 (second half of same 32 bytes)
        // qs: first 32 bytes cover groups 0 (low nibbles) and 1 (high nibbles)
        for i in 0..32 {
            b[16 + i] = ((i % 16) as u8) | (((15 - i % 16) as u8) << 4);
        }
        let mut out = Vec::new();
        dequant_q4k_block(&b, &mut out);
        assert_eq!(out.len(), 256);
        for i in 0..32 {
            assert_eq!(out[i], 2.0 * (i % 16) as f32, "group0[{i}]");
            assert_eq!(out[32 + i], 3.0 * (15 - i % 16) as f32, "group1[{i}]");
        }
        // dmin=0 and zero scales elsewhere -> rest decodes to 0.
        assert!(out[64..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4k_packed_scale_min_high_groups_unpack() {
        // Groups 4..7 pack 6-bit values across bytes; verify the bit surgery.
        let mut s = [0u8; 12];
        // Encode group 4: sc = 0b110101 (=53), m = 0b101011 (=43).
        // Layout: low 4 bits of sc in s[8]&0xF, high 2 bits in s[0]>>6;
        //         low 4 bits of m  in s[8]>>4,  high 2 bits in s[4]>>6.
        s[8] = (53 & 0xF) | ((43 & 0xF) << 4);
        s[0] = (53u8 >> 4) << 6;
        s[4] = (43u8 >> 4) << 6;
        let (sc, m) = q4k_scale_min(4, &s);
        assert_eq!((sc, m), (53.0, 43.0));
    }

    #[test]
    fn q6k_block_dequantizes_known_values() {
        // d = 1.0, all 16 sub-block scales = 1: y[l] = q - 32 where q is the
        // 6-bit value. Set ql[0] = 5 (low nibble), qh[0] = 2 (bits 0-1 -> +32
        // ... actually bits<<4): q1 = 5 | (2<<4) = 37 -> y[0] = 5.
        let mut b = vec![0u8; Q6_K_BLOCK_BYTES];
        b[192..208].fill(1); // all 16 sub-block scales = 1
        b[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        b[0] = 5; // ql[0]
        b[128] = 2; // qh[0]: bits 0-1 = 2
        let mut out = Vec::new();
        dequant_q6k_block(&b, &mut out);
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], (5 + (2 << 4) - 32) as f32); // = 5.0
        // Everything else in group 0's lane had q=0 -> value -32.
        assert_eq!(out[1], -32.0);
        // High-nibble group of the same byte: ql[0]>>4 = 0, qh bits 4-5 = 0.
        assert_eq!(out[64], -32.0);
    }

    #[test]
    fn q8_0_tensor_loads_natively_as_quantmatrix() {
        // One row of 32 values stored as a single Q8_0 block: scale=1.0(f16),
        // q[i] = i - 16.
        let mut w = W(Vec::new());
        w.0.extend_from_slice(&MAGIC.to_le_bytes());
        w.0.extend_from_slice(&3u32.to_le_bytes());
        w.0.extend_from_slice(&1u64.to_le_bytes());
        w.0.extend_from_slice(&0u64.to_le_bytes());
        w.str_(b"w");
        w.0.extend_from_slice(&2u32.to_le_bytes());
        w.0.extend_from_slice(&32u64.to_le_bytes()); // cols
        w.0.extend_from_slice(&1u64.to_le_bytes()); // rows
        w.0.extend_from_slice(&GGML_Q8_0.to_le_bytes());
        w.0.extend_from_slice(&0u64.to_le_bytes());
        while !w.0.len().is_multiple_of(32) {
            w.0.push(0);
        }
        w.0.extend_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        for i in 0..32i8 {
            w.0.push((i - 16) as u8);
        }

        let g = Gguf::from_bytes(w.0).unwrap();
        let lin = g.linear("w", Precision::F32).unwrap();
        match &lin {
            Linear::Q8(qm) => assert_eq!((qm.rows, qm.cols), (1, 32)),
            _ => panic!("expected native Q8"),
        }
        let x = vec![1.0f32; 32];
        // sum of (i-16) for i in 0..32 = -16
        assert!((lin.apply(&x)[0] + 16.0).abs() < 1e-4);
    }
}
