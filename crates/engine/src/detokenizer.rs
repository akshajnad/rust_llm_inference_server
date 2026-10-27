use anyhow::Result;

/// Turns token ids into text.
pub trait Decode {
    fn decode(&self, ids: &[u32]) -> Result<String>;
}

/// Streams text as tokens arrive, without breaking multi-byte characters.
///
/// Decoding a single token in isolation is wrong for byte-level BPE: one character can span
/// several tokens, and leading-space handling depends on context. Instead, decode a short
/// window ending at the newest token and emit only what it adds beyond the text already
/// emitted. While the window ends in U+FFFD (an incomplete UTF-8 sequence), hold output back.
#[derive(Debug, Default)]
pub struct Detokenizer {
    tokens: Vec<u32>,
    /// Start of the context window decoded together with the new tokens.
    prefix_offset: usize,
    /// Tokens before this index have been emitted.
    read_offset: usize,
}

impl Detokenizer {
    /// `prompt` tokens give leading context but are never emitted.
    pub fn new(prompt: &[u32]) -> Self {
        Self {
            tokens: prompt.to_vec(),
            // A few tokens of context are enough for the decoder's spacing rules.
            prefix_offset: prompt.len().saturating_sub(5),
            read_offset: prompt.len(),
        }
    }

    /// Adds one token and returns the newly completed text, possibly empty.
    pub fn push(&mut self, decoder: &impl Decode, id: u32) -> Result<String> {
        self.tokens.push(id);
        let prefix = decoder.decode(&self.tokens[self.prefix_offset..self.read_offset])?;
        let full = decoder.decode(&self.tokens[self.prefix_offset..])?;
        if full.len() > prefix.len() && !full.ends_with('\u{FFFD}') {
            let new = full.get(prefix.len()..).unwrap_or_default().to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.tokens.len();
            return Ok(new);
        }
        Ok(String::new())
    }

    /// Emits whatever is still held back at the end of generation, even if incomplete.
    pub fn flush(&mut self, decoder: &impl Decode) -> Result<String> {
        if self.read_offset == self.tokens.len() {
            return Ok(String::new());
        }
        let prefix = decoder.decode(&self.tokens[self.prefix_offset..self.read_offset])?;
        let full = decoder.decode(&self.tokens[self.prefix_offset..])?;
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        Ok(full.get(prefix.len()..).unwrap_or_default().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each token id is one raw byte, the worst case for split characters.
    struct Bytes;

    impl Decode for Bytes {
        fn decode(&self, ids: &[u32]) -> Result<String> {
            let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }

    fn stream(text: &str) -> Vec<String> {
        let mut d = Detokenizer::new(&[]);
        let mut out: Vec<String> = text
            .bytes()
            .map(|b| d.push(&Bytes, b as u32).unwrap())
            .collect();
        out.push(d.flush(&Bytes).unwrap());
        out
    }

    #[test]
    fn ascii_streams_one_char_per_token() {
        assert_eq!(stream("hi!").concat(), "hi!");
        assert_eq!(stream("hi!")[..3], ["h", "i", "!"]);
    }

    #[test]
    fn multibyte_chars_are_held_until_complete() {
        let text = "a€😀b";
        let chunks = stream(text);
        assert_eq!(chunks.concat(), text);
        assert!(chunks.iter().all(|c| !c.contains('\u{FFFD}')));
        // '€' is 3 bytes: nothing is emitted for its first two.
        assert_eq!(chunks[1..4], ["", "", "€"]);
    }

    #[test]
    fn prompt_is_not_emitted() {
        let mut d = Detokenizer::new(&[b'x' as u32, b'y' as u32]);
        assert_eq!(d.push(&Bytes, b'z' as u32).unwrap(), "z");
    }

    #[test]
    fn flush_emits_truncated_tail() {
        let mut d = Detokenizer::new(&[]);
        let euro = "€".as_bytes();
        assert_eq!(d.push(&Bytes, euro[0] as u32).unwrap(), "");
        assert_eq!(d.flush(&Bytes).unwrap(), "\u{FFFD}");
    }
}
