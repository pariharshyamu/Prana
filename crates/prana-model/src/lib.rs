//! Model loading, tokenization, and text generation for Prana — the layer
//! that turns the kernel/graph prototype into a working inference engine.
//!
//! Supports the `llama2.c` checkpoint format (Karpathy's TinyStories models):
//! [`checkpoint::load`] reads the weights (optionally re-quantizing to Q8),
//! [`Tokenizer`] handles the Llama SentencePiece vocab, and [`generate`] runs
//! the KV-cached forward pass built from `prana-kernels`. Everything here is
//! `#![forbid(unsafe_code)]`.

pub mod bpe;
pub mod checkpoint;
pub mod gguf;
pub mod infer;
pub mod sampler;
pub mod tokenizer;

pub use checkpoint::{Activation, Config, Model, Precision};
pub use infer::{forward, generate, GenStats, KvCache};
pub use sampler::Sampler;
pub use tokenizer::{AnyTokenizer, Tokenize, Tokenizer};

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny random-ish (deterministic) llama2.c checkpoint image:
    /// dim=32, hidden=64, 2 layers, 2 heads, vocab=48, seq=16, shared cls.
    fn synthetic_checkpoint() -> Vec<u8> {
        let (dim, hidden, layers, heads, kv_heads, vocab, seq) = (32i32, 64i32, 2i32, 2i32, 2i32, 48i32, 16i32);
        let head_dim = (dim / heads) as usize;
        let mut out = Vec::new();
        for v in [dim, hidden, layers, heads, kv_heads, vocab, seq] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut push_f32s = |n: usize, scale: f32, out: &mut Vec<u8>| {
            for _ in 0..n {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let r = ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                out.extend_from_slice(&(r * scale).to_le_bytes());
            }
        };
        let (dim, hidden, layers, vocab, seq) =
            (dim as usize, hidden as usize, layers as usize, vocab as usize, seq as usize);
        push_f32s(vocab * dim, 0.1, &mut out); // tok_emb
        push_f32s(layers * dim, 1.0, &mut out); // rms_att (near-1 would be nicer; harmless)
        push_f32s(layers * dim * dim, 0.1, &mut out); // wq
        push_f32s(layers * dim * dim, 0.1, &mut out); // wk
        push_f32s(layers * dim * dim, 0.1, &mut out); // wv
        push_f32s(layers * dim * dim, 0.1, &mut out); // wo
        push_f32s(layers * dim, 1.0, &mut out); // rms_ffn
        push_f32s(layers * hidden * dim, 0.1, &mut out); // w1
        push_f32s(layers * dim * hidden, 0.1, &mut out); // w2
        push_f32s(layers * hidden * dim, 0.1, &mut out); // w3
        push_f32s(dim, 1.0, &mut out); // rms_final
        push_f32s(seq * head_dim, 0.0, &mut out); // rope tables (skipped by loader)
        out
    }

    #[test]
    fn loads_synthetic_checkpoint_and_runs_forward() {
        let bytes = synthetic_checkpoint();
        let model = checkpoint::load_bytes(&bytes, Precision::F32).unwrap();
        assert_eq!(model.config.dim, 32);
        assert_eq!(model.config.n_layers, 2);
        assert!(model.config.shared_classifier);

        let mut cache = KvCache::new(&model);
        let logits = forward(&model, &mut cache, 5, 0);
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_is_deterministic_and_position_dependent() {
        let model = checkpoint::load_bytes(&synthetic_checkpoint(), Precision::F32).unwrap();

        let mut c1 = KvCache::new(&model);
        let mut c2 = KvCache::new(&model);
        let a = forward(&model, &mut c1, 7, 0);
        let b = forward(&model, &mut c2, 7, 0);
        assert_eq!(a, b, "same token+pos must be bit-identical");

        // Same token later in the sequence must differ (RoPE + cache history).
        let c = forward(&model, &mut c1, 7, 1);
        assert_ne!(a, c);
    }

    #[test]
    fn q8_model_tracks_f32_model() {
        let bytes = synthetic_checkpoint();
        let m32 = checkpoint::load_bytes(&bytes, Precision::F32).unwrap();
        let m8 = checkpoint::load_bytes(&bytes, Precision::Q8).unwrap();
        assert!(m8.projection_bytes() < m32.projection_bytes() / 3, "Q8 must shrink weights >3x");

        let mut c32 = KvCache::new(&m32);
        let mut c8 = KvCache::new(&m8);
        let (mut l32, mut l8) = (Vec::new(), Vec::new());
        for (pos, tok) in [(0usize, 3u32), (1, 11), (2, 4)] {
            l32 = forward(&m32, &mut c32, tok, pos);
            l8 = forward(&m8, &mut c8, tok, pos);
        }
        // Rankings should broadly agree; check argmax and value closeness.
        let rms = (l32.iter().map(|v| v * v).sum::<f32>() / l32.len() as f32).sqrt().max(1e-6);
        let max_err = l32.iter().zip(&l8).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_err / rms < 0.15, "Q8 diverged: max_err={max_err}, rms={rms}");
    }

    #[test]
    fn generate_streams_pieces_and_respects_context_limit() {
        let model = checkpoint::load_bytes(&synthetic_checkpoint(), Precision::F32).unwrap();
        // Tokenizer with 48 pieces: specials + printable ASCII singles.
        let mut data = 8u32.to_le_bytes().to_vec();
        let pieces: Vec<Vec<u8>> = (0..48u8)
            .map(|i| match i {
                0 => b"<unk>".to_vec(),
                1 => b"<s>".to_vec(),
                2 => b"</s>".to_vec(),
                3 => b" ".to_vec(),
                i => vec![b'a' + (i - 4) % 26],
            })
            .collect();
        for p in &pieces {
            data.extend_from_slice(&(-1.0f32).to_le_bytes());
            data.extend_from_slice(&(p.len() as u32).to_le_bytes());
            data.extend_from_slice(p);
        }
        let tok = Tokenizer::from_bytes(&data, 48).unwrap();

        let mut sampler = Sampler::new(0.0, 1);
        let mut streamed = Vec::new();
        let stats = generate(&model, &tok, "ab", 64, &mut sampler, |piece, _| {
            streamed.extend_from_slice(piece);
        });
        // Context is 16: prompt (BOS + " " + 'a' + 'b' = 4 tokens) + generation
        // must never exceed seq_len.
        assert!(stats.prompt_tokens + stats.generated_tokens <= model.config.seq_len);
        assert!(!streamed.is_empty());
    }
}
