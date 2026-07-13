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

use prana_kernels::{QuantMatrix, Q8_BLOCK};

use crate::checkpoint::{Config, Layer, Linear, Model, Precision};
use crate::tokenizer::Tokenizer;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

// ggml tensor dtypes we support.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q8_0: u32 = 8;

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

/// IEEE 754 half → single conversion (handles subnormals, inf, NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        // subnormal: exact value is frac * 2^-24 (sign applied numerically)
        (0, f) => {
            let v = f as f32 * 2f32.powi(-24);
            return if sign == 1 { -v } else { v };
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        (e, f) => (sign << 31) | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
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
        Ok(Linear::new(self.tensor_f32(name)?, rows, cols, precision))
    }

    fn meta_usize(&self, key: &str) -> io::Result<usize> {
        self.metadata
            .get(key)
            .and_then(Value::as_usize)
            .ok_or_else(|| err(format!("missing metadata '{key}'")))
    }
}

/// Load a GGUF llama model plus its embedded tokenizer.
pub fn load(path: &Path, precision: Precision) -> io::Result<(Model, Tokenizer)> {
    let g = Gguf::read(path)?;

    let arch = match g.metadata.get("general.architecture") {
        Some(Value::Str(s)) => String::from_utf8_lossy(s).into_owned(),
        _ => return Err(err("missing general.architecture")),
    };
    if arch != "llama" {
        return Err(err(format!("unsupported architecture '{arch}' (only llama)")));
    }

    let dim = g.meta_usize("llama.embedding_length")?;
    let n_layers = g.meta_usize("llama.block_count")?;
    let n_heads = g.meta_usize("llama.attention.head_count")?;
    let n_kv_heads = g
        .metadata
        .get("llama.attention.head_count_kv")
        .and_then(Value::as_usize)
        .unwrap_or(n_heads);
    let hidden_dim = g.meta_usize("llama.feed_forward_length")?;
    let seq_len = g.meta_usize("llama.context_length")?;
    let rope_theta = g
        .metadata
        .get("llama.rope.freq_base")
        .and_then(Value::as_f32)
        .unwrap_or(10000.0);

    // Tokenizer from the embedded SentencePiece vocab. GGUF stores pieces
    // with U+2581 (▁) word markers; our tokenizer uses plain spaces.
    let tokens = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Arr(items)) => items,
        _ => return Err(err("missing tokenizer.ggml.tokens")),
    };
    let scores: Vec<f32> = match g.metadata.get("tokenizer.ggml.scores") {
        Some(Value::Arr(items)) => items.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect(),
        _ => vec![0.0; tokens.len()],
    };
    let vocab: Vec<Vec<u8>> = tokens
        .iter()
        .map(|v| match v {
            Value::Str(s) => {
                let s = String::from_utf8_lossy(s).replace('\u{2581}', " ");
                s.into_bytes()
            }
            _ => Vec::new(),
        })
        .collect();
    let vocab_size = vocab.len();
    let tokenizer = Tokenizer::from_parts(vocab, scores);

    let tok_emb = g.tensor_f32("token_embd.weight")?;
    if tok_emb.len() != vocab_size * dim {
        return Err(err("token_embd.weight size mismatch with vocab"));
    }

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let t = |suffix: &str| format!("blk.{i}.{suffix}");
        layers.push(Layer {
            rms_att: g.tensor_f32(&t("attn_norm.weight"))?,
            wq: g.linear(&t("attn_q.weight"), precision)?,
            wk: g.linear(&t("attn_k.weight"), precision)?,
            wv: g.linear(&t("attn_v.weight"), precision)?,
            wo: g.linear(&t("attn_output.weight"), precision)?,
            rms_ffn: g.tensor_f32(&t("ffn_norm.weight"))?,
            w1: g.linear(&t("ffn_gate.weight"), precision)?,
            w2: g.linear(&t("ffn_down.weight"), precision)?,
            w3: g.linear(&t("ffn_up.weight"), precision)?,
        });
    }

    let rms_final = g.tensor_f32("output_norm.weight")?;

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
    };
    Ok((Model { config, tok_emb, layers, rms_final, wcls }, tokenizer))
}

#[cfg(test)]
mod tests {
    use super::*;

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
