use std::path::Path;

use anyhow::{Context, Result};

use crate::detokenizer::Decode;

/// Hugging Face tokenizer plus the stop tokens from `generation_config.json`.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos_ids: Vec<u32>,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)
            .context("loading tokenizer.json")?;
        let eos_ids = read_eos_ids(model_dir)?;
        Ok(Self { inner, eos_ids })
    }

    /// Encodes `text` with special tokens, which for Llama 3 prepends `<|begin_of_text|>`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, true).map_err(anyhow::Error::msg)?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
}

impl Decode for Tokenizer {
    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true).map_err(anyhow::Error::msg)
    }
}

/// `eos_token_id` from `generation_config.json`, falling back to `config.json`; either may be
/// a single id or a list.
fn read_eos_ids(dir: &Path) -> Result<Vec<u32>> {
    for name in ["generation_config.json", "config.json"] {
        let Ok(text) = std::fs::read_to_string(dir.join(name)) else {
            continue;
        };
        let json: serde_json::Value = serde_json::from_str(&text)?;
        match &json["eos_token_id"] {
            serde_json::Value::Number(n) => return Ok(vec![n.as_u64().unwrap_or(0) as u32]),
            serde_json::Value::Array(a) => {
                return Ok(a
                    .iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as u32)
                    .collect());
            }
            _ => continue,
        }
    }
    anyhow::bail!("no eos_token_id in {}", dir.display())
}
