//! Minimal JSON *encoding* — just enough to render the `messages_json` and
//! `options_json` payloads the Cactus ABI takes, with correct string escaping.
//! Kept dependency-free on purpose (the whole workspace has zero external
//! deps); Phase 1 would swap this for `serde_json` when parsing responses.

/// Escape a string for inclusion inside JSON double quotes.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_quotes_backslashes_and_control_chars() {
        assert_eq!(escape(r#"say "hi"\now"#), r#"say \"hi\"\\now"#);
        assert_eq!(escape("a\nb\tc"), "a\\nb\\tc");
        assert_eq!(escape("\u{1}"), "\\u0001");
    }

    #[test]
    fn passes_plain_text_and_unicode_through() {
        assert_eq!(escape("héllo 世界"), "héllo 世界");
    }
}
