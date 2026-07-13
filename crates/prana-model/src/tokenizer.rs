//! The Llama SentencePiece tokenizer, loaded from llama2.c's `tokenizer.bin`
//! (a flat dump of the 32000-entry vocab: merge scores + piece bytes).
//!
//! Encoding is greedy BPE exactly as `run.c` does it: seed with one token per
//! UTF-8 codepoint (byte-fallback tokens `<0xNN>` for unknowns), then
//! repeatedly merge the adjacent pair whose concatenation is the
//! highest-scoring vocab entry. SentencePiece marks word boundaries with a
//! leading space, so a dummy space token is prepended to non-empty text.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

pub const BOS: u32 = 1;
pub const EOS: u32 = 2;

pub struct Tokenizer {
    /// Piece bytes per token id.
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    /// Piece bytes -> id (first occurrence wins, like run.c's sorted lookup).
    lookup: HashMap<Vec<u8>, u32>,
}

impl Tokenizer {
    pub fn load(path: &Path, vocab_size: usize) -> io::Result<Self> {
        Self::from_bytes(&fs::read(path)?, vocab_size)
    }

    /// Build directly from piece/score lists (e.g. GGUF's embedded
    /// `tokenizer.ggml.tokens` / `.scores` metadata arrays).
    pub fn from_parts(vocab: Vec<Vec<u8>>, scores: Vec<f32>) -> Self {
        assert_eq!(vocab.len(), scores.len());
        let mut lookup = HashMap::with_capacity(vocab.len());
        for (id, piece) in vocab.iter().enumerate() {
            lookup.entry(piece.clone()).or_insert(id as u32);
        }
        Self { vocab, scores, lookup }
    }

    /// Parse the tokenizer.bin layout: `u32 max_token_length`, then
    /// `vocab_size ×  (f32 score, u32 len, len bytes)`.
    pub fn from_bytes(data: &[u8], vocab_size: usize) -> io::Result<Self> {
        let eof = || io::Error::new(io::ErrorKind::UnexpectedEof, "tokenizer.bin truncated");
        let mut off = 4; // skip max_token_length
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let s = data.get(off..off + 4).ok_or_else(eof)?;
            scores.push(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
            let l = data.get(off + 4..off + 8).ok_or_else(eof)?;
            let len = u32::from_le_bytes([l[0], l[1], l[2], l[3]]) as usize;
            let bytes = data.get(off + 8..off + 8 + len).ok_or_else(eof)?;
            vocab.push(bytes.to_vec());
            off += 8 + len;
        }
        Ok(Self::from_parts(vocab, scores))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode text to token ids, optionally wrapping in BOS/EOS.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();
        if bos {
            tokens.push(BOS);
        }
        // SentencePiece dummy prefix: word-initial pieces carry a leading
        // space, so non-empty text starts with the " " token.
        if !text.is_empty() {
            if let Some(&sp) = self.lookup.get(b" ".as_slice()) {
                tokens.push(sp);
            }
        }
        // Seed: one token per codepoint, with byte fallback (<0xNN> tokens
        // occupy ids 3..259, hence the +3 offset).
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let piece = c.encode_utf8(&mut buf).as_bytes();
            match self.lookup.get(piece) {
                Some(&id) => tokens.push(id),
                None => tokens.extend(piece.iter().map(|&b| b as u32 + 3)),
            }
        }

        // Greedy BPE: keep merging the best-scoring adjacent pair.
        loop {
            let mut best: Option<(f32, usize, u32)> = None; // (score, index, merged id)
            for i in 0..tokens.len().saturating_sub(1) {
                let (Some(a), Some(b)) =
                    (self.vocab.get(tokens[i] as usize), self.vocab.get(tokens[i + 1] as usize))
                else {
                    continue; // byte-fallback id beyond this vocab: never mergeable
                };
                let mut merged = a.clone();
                merged.extend_from_slice(b);
                if let Some(&id) = self.lookup.get(&merged) {
                    let score = self.scores[id as usize];
                    if best.is_none_or(|(s, _, _)| score > s) {
                        best = Some((score, i, id));
                    }
                }
            }
            match best {
                Some((_, i, id)) => {
                    tokens[i] = id;
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }

        if eos {
            tokens.push(EOS);
        }
        tokens
    }

    /// Decode one token into its piece bytes, given the previous token
    /// (SentencePiece strips the leading space of the piece right after BOS).
    pub fn decode(&self, prev: u32, token: u32) -> Vec<u8> {
        let piece = match self.vocab.get(token as usize) {
            Some(p) => p.as_slice(),
            None => return Vec::new(),
        };
        // Raw-byte tokens are stored as the literal text "<0xNN>".
        if piece.len() == 6 && piece.starts_with(b"<0x") && piece.ends_with(b">") {
            if let Ok(b) = u8::from_str_radix(std::str::from_utf8(&piece[3..5]).unwrap_or(""), 16) {
                return vec![b];
            }
        }
        if prev == BOS && piece.first() == Some(&b' ') {
            return piece[1..].to_vec();
        }
        piece.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tokenizer.bin image from (score, piece) entries.
    fn build(entries: &[(f32, &[u8])]) -> Tokenizer {
        let mut data = 8u32.to_le_bytes().to_vec(); // max_token_length (unused)
        for (score, piece) in entries {
            data.extend_from_slice(&score.to_le_bytes());
            data.extend_from_slice(&(piece.len() as u32).to_le_bytes());
            data.extend_from_slice(piece);
        }
        Tokenizer::from_bytes(&data, entries.len()).unwrap()
    }

    fn tiny_vocab() -> Tokenizer {
        // ids: 0 <unk>, 1 <s>, 2 </s>, then pieces.
        build(&[
            (0.0, b"<unk>"),
            (0.0, b"<s>"),
            (0.0, b"</s>"),
            (-1.0, b" "),
            (-2.0, b"h"),
            (-2.0, b"i"),
            (-1.5, b"hi"),
            (-1.2, b" hi"),
        ])
    }

    #[test]
    fn encodes_with_greedy_merges_and_dummy_prefix() {
        let t = tiny_vocab();
        // " " + "h" + "i" -> merge "h"+"i"='hi' (-1.5) ... then " "+"hi"=' hi' (-1.2)
        let ids = t.encode("hi", true, false);
        assert_eq!(ids, vec![BOS, 7], "expected [BOS, ' hi']: {ids:?}");
    }

    #[test]
    fn byte_fallback_for_unknown_chars() {
        let t = tiny_vocab();
        let ids = t.encode("z", false, false);
        // dummy prefix " " (id 3), then byte fallback: 'z' = 0x7A -> 0x7A + 3
        assert_eq!(ids, vec![3, b'z' as u32 + 3]);
    }

    #[test]
    fn decode_strips_space_after_bos_and_parses_byte_tokens() {
        let t = build(&[(0.0, b"<unk>"), (0.0, b"<s>"), (0.0, b"</s>"), (-1.0, b" word"), (0.0, b"<0x0A>")]);
        assert_eq!(t.decode(BOS, 3), b"word");
        assert_eq!(t.decode(9, 3), b" word");
        assert_eq!(t.decode(3, 4), vec![0x0A]);
    }
}
