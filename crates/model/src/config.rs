use serde::Deserialize;

/// The subset of a Hugging Face Llama `config.json` the forward pass needs.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub bos_token_id: Option<u32>,
    /// Compute attention scores in f32 even when the KV cache is f16. Qwen2's Q/K biases make
    /// q·k overflow f16 (|k| reaches ~360 in layer 0); llama.cpp forces f32 there too.
    #[serde(default)]
    pub attention_f32: bool,
}

/// Llama 3 style frequency-dependent RoPE scaling (`rope_type: "llama3"`).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

impl LlamaConfig {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Bytes of K plus V per token across all layers for the given element size.
    pub fn kv_bytes_per_token(&self, elem_size: usize) -> usize {
        2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim() * elem_size
    }

    /// A tiny config for tests that run without real weights.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(16),
            vocab_size: 97,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            max_position_embeddings: 128,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            attention_f32: false,
        }
    }
}
