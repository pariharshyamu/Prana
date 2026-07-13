//! Byte-level BPE (GPT-2 lineage) — the tokenizer family Qwen2, SmolLM, and
//! most non-SentencePiece models use, reconstructed from GGUF's embedded
//! `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`.
//!
//! GPT-2 BPE first maps raw bytes to a printable unicode alphabet
//! (`bytes_to_unicode`), so vocab entries and merges are strings over that
//! alphabet. Encoding pre-splits text into word-ish pieces, seeds each piece
//! as one symbol per (mapped) byte, then repeatedly applies the lowest-ranked
//! merge from the merges list.
//!
//! Honest caveat: real GPT-2 pre-tokenization is a specific regex with
//! contraction rules and unicode categories. This implementation approximates
//! it (leading-space words, letter/digit/punctuation runs), which can split
//! rare inputs differently than the reference. For the models we can actually
//! verify in this environment that distinction is untestable; the structure
//! and merges application follow the reference algorithm exactly.

use std::collections::HashMap;

use crate::tokenizer::Tokenize;

/// GPT-2's byte -> printable-unicode mapping.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable = |b: u16| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut n = 0u16;
    for b in 0..256u16 {
        if printable(b) {
            table[b as usize] = char::from_u32(b as u32).unwrap();
        } else {
            table[b as usize] = char::from_u32(256 + n as u32).unwrap();
            n += 1;
        }
    }
    table
}

pub struct BpeTokenizer {
    /// Vocab strings (over the byte-unicode alphabet) by id.
    vocab: Vec<String>,
    lookup: HashMap<String, u32>,
    /// (left, right) -> merge rank (index in the merges list).
    ranks: HashMap<(String, String), u32>,
    byte_enc: [char; 256],
    /// Mapped char -> original byte.
    byte_dec: HashMap<char, u8>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    /// Control/special tokens (`<|im_start|>`, ...) matched verbatim before
    /// pre-tokenization, longest first. Stored raw in the vocab (not
    /// byte-mapped), so they must bypass the BPE path entirely.
    specials: Vec<(String, u32)>,
}

impl BpeTokenizer {
    /// `merges` entries are "left right" pairs in rank order (GGUF layout).
    pub fn new(
        vocab: Vec<String>,
        merges: &[String],
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        add_bos: bool,
    ) -> Self {
        let mut lookup = HashMap::with_capacity(vocab.len());
        for (id, piece) in vocab.iter().enumerate() {
            lookup.entry(piece.clone()).or_insert(id as u32);
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            if let Some((l, r)) = m.split_once(' ') {
                ranks.insert((l.to_string(), r.to_string()), rank as u32);
            }
        }
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        Self { vocab, lookup, ranks, byte_enc, byte_dec, bos_id, eos_id, add_bos, specials: Vec::new() }
    }

    /// Register special tokens (from GGUF `tokenizer.ggml.token_type`).
    pub fn with_specials(mut self, mut specials: Vec<(String, u32)>) -> Self {
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len())); // longest match wins
        self.specials = specials;
        self
    }

    /// Look up a token id by its literal vocab string (e.g. `<|im_start|>`).
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.lookup.get(piece).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Approximate GPT-2 pre-tokenization: split into pieces that keep one
    /// leading space, grouped by rough character class.
    fn pre_tokenize(text: &str) -> Vec<String> {
        #[derive(PartialEq, Clone, Copy)]
        enum Class {
            Letter,
            Digit,
            Other,
            Space,
        }
        let class = |c: char| {
            if c.is_alphabetic() {
                Class::Letter
            } else if c.is_ascii_digit() {
                Class::Digit
            } else if c == ' ' {
                Class::Space
            } else {
                Class::Other // incl. non-space whitespace
            }
        };

        let mut pieces = Vec::new();
        let mut cur = String::new();
        let mut cur_class = Class::Space;
        for ch in text.chars() {
            let cl = class(ch);
            let starts_new = match (cur.as_str(), cl) {
                ("", _) => false,
                // a single pending space attaches to a following word
                (" ", Class::Letter | Class::Digit | Class::Other) => false,
                _ => cl != cur_class || cl == Class::Space && ch != ' ',
            };
            if starts_new || (cl == Class::Space && !cur.ends_with(' ') && !cur.is_empty()) {
                pieces.push(std::mem::take(&mut cur));
            }
            cur.push(ch);
            cur_class = cl;
        }
        if !cur.is_empty() {
            pieces.push(cur);
        }
        pieces
    }

    /// Apply ranked merges to one pre-tokenized piece, returning token ids.
    fn bpe_piece(&self, piece: &str) -> Vec<u32> {
        // Map to the byte-unicode alphabet, one symbol per byte.
        let mut symbols: Vec<String> =
            piece.bytes().map(|b| self.byte_enc[b as usize].to_string()).collect();

        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self.ranks.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                    symbols[i] = merged;
                    symbols.remove(i + 1);
                }
                None => break,
            }
        }

        // Unknown symbols are dropped; a well-formed byte-level vocab contains
        // every single mapped byte, so post-merge symbols always resolve.
        symbols.iter().filter_map(|s| self.lookup.get(s).copied()).collect()
    }
}

impl Tokenize for BpeTokenizer {
    fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            if let Some(b) = self.bos_id {
                out.push(b);
            }
        }
        // Specials are matched verbatim before BPE sees the text; `specials`
        // is sorted longest-first, so ties at one position take the longest.
        let mut rest = text;
        while !rest.is_empty() {
            let hit = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by_key(|&(at, _, _)| at);
            let (plain, special, tail) = match hit {
                Some((at, len, id)) => (&rest[..at], Some(id), &rest[at + len..]),
                None => (rest, None, ""),
            };
            for piece in Self::pre_tokenize(plain) {
                out.extend(self.bpe_piece(&piece));
            }
            out.extend(special);
            rest = tail;
        }
        out
    }

    fn decode(&self, _prev: u32, token: u32) -> Vec<u8> {
        let Some(s) = self.vocab.get(token as usize) else {
            return Vec::new();
        };
        s.chars().filter_map(|c| self.byte_dec.get(&c).copied()).collect()
    }

    fn is_stop(&self, token: u32) -> bool {
        Some(token) == self.eos_id || Some(token) == self.bos_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny byte-level vocab: single mapped bytes for "h", "i", " " plus
    /// merged entries, with merges ranked.
    fn tiny() -> BpeTokenizer {
        let enc = bytes_to_unicode();
        let sp = enc[b' ' as usize]; // 'Ġ' in GPT-2 convention
        let vocab = vec![
            "h".to_string(),
            "i".to_string(),
            sp.to_string(),
            "hi".to_string(),
            format!("{sp}hi"),
            "<|endoftext|>".to_string(),
        ];
        let merges = vec!["h i".to_string(), format!("{sp} hi")];
        BpeTokenizer::new(vocab, &merges, None, Some(5), false)
    }

    #[test]
    fn merges_apply_in_rank_order() {
        let t = tiny();
        // "hi" -> [h, i] -> merge rank 0 -> "hi" (id 3)
        assert_eq!(t.encode_prompt("hi"), vec![3]);
        // " hi" -> [Ġ, h, i] -> "h i" first (rank 0), then "Ġ hi" (rank 1)
        assert_eq!(t.encode_prompt(" hi"), vec![4]);
    }

    #[test]
    fn decode_roundtrips_bytes() {
        let t = tiny();
        let ids = t.encode_prompt(" hi");
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.decode(0, i)).collect();
        assert_eq!(bytes, b" hi");
    }

    #[test]
    fn eos_is_stop() {
        let t = tiny();
        assert!(t.is_stop(5));
        assert!(!t.is_stop(3));
    }

    #[test]
    fn pre_tokenizer_keeps_leading_spaces_on_words() {
        let pieces = BpeTokenizer::pre_tokenize("hello world 42!");
        assert_eq!(pieces, vec!["hello", " world", " 42", "!"]);
    }

    #[test]
    fn special_tokens_encode_verbatim_not_split() {
        let t = tiny().with_specials(vec![("<|endoftext|>".to_string(), 5)]);
        // The special must come out as one id, with normal BPE around it.
        assert_eq!(t.encode_prompt("hi<|endoftext|> hi"), vec![3, 5, 4]);
        assert_eq!(t.encode_prompt("<|endoftext|>"), vec![5]);
        // Without registration the same text falls through to byte BPE
        // (and here mostly drops: tiny vocab lacks '<', '|', ...).
        assert_ne!(tiny().encode_prompt("hi<|endoftext|> hi"), vec![3, 5, 4]);
    }
}
