use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::attention::{Attention, causal_mask};
use crate::config::LlamaConfig;
use crate::kv::KvStore;
use crate::layers::{Linear, Mlp, RmsNorm};
use crate::rope::Rope;

#[derive(Debug, Clone)]
struct Block {
    input_norm: RmsNorm,
    attn: Attention,
    post_attn_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &LlamaConfig) -> Result<Self> {
        let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        Ok(Self {
            input_norm: RmsNorm::load(vb.pp("input_layernorm"), h, eps)?,
            attn: Attention::load(vb.pp("self_attn"), cfg)?,
            post_attn_norm: RmsNorm::load(vb.pp("post_attention_layernorm"), h, eps)?,
            mlp: Mlp::load(vb.pp("mlp"), h, cfg.intermediate_size)?,
        })
    }
}

/// Llama-architecture decoder (RMSNorm, RoPE, grouped-query attention, SwiGLU).
#[derive(Debug, Clone)]
pub struct Llama {
    cfg: LlamaConfig,
    embed: Tensor,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    dtype: DType,
    device: Device,
}

impl Llama {
    /// Loads HF safetensors from `dir`, casting weights to `dtype`. RoPE tables cover
    /// `max_seq_len` positions.
    pub fn load(
        dir: &Path,
        dtype: DType,
        device: &Device,
        max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        let cfg = LlamaConfig::from_file(dir.join("config.json"))?;
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        anyhow::ensure!(
            !files.is_empty(),
            "no .safetensors files in {}",
            dir.display()
        );
        // SAFETY: the files are opened read-only and not modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
        Ok(Self::new(cfg, vb, max_seq_len)?)
    }

    pub fn new(cfg: LlamaConfig, vb: VarBuilder, max_seq_len: usize) -> Result<Self> {
        let (dtype, device) = (vb.dtype(), vb.device().clone());
        let m = vb.pp("model");
        let embed = m
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(m.pp(format!("layers.{i}")), &cfg))
            .collect::<Result<_>>()?;
        let norm = RmsNorm::load(m.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::Dense(embed.clone())
        } else {
            Linear::load(vb.pp("lm_head"), cfg.hidden_size, cfg.vocab_size)?
        };
        let max_pos = max_seq_len.min(cfg.max_position_embeddings);
        let rope = Rope::new(&cfg, max_pos, dtype, &device)?;
        Ok(Self {
            cfg,
            embed,
            blocks,
            norm,
            lm_head,
            rope,
            dtype,
            device,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn max_seq_len(&self) -> usize {
        self.rope.max_positions()
    }

    /// Runs `tokens` (one sequence) through the model, continuing after the positions already in
    /// `kv`, and returns final hidden states `(1, seq, hidden)`.
    fn hidden(&self, tokens: &[u32], kv: &mut dyn KvStore) -> Result<Tensor> {
        let start_pos = kv.seq_len();
        let seq = tokens.len();
        if start_pos + seq > self.max_seq_len() {
            candle_core::bail!(
                "sequence of {} positions exceeds max_seq_len {}",
                start_pos + seq,
                self.max_seq_len()
            );
        }
        let ids = Tensor::new(tokens, &self.device)?;
        let mut x = self.embed.index_select(&ids, 0)?.unsqueeze(0)?;
        let mask = match seq {
            1 => None,
            _ => Some(causal_mask(seq, start_pos, &self.device)?),
        };
        for (i, block) in self.blocks.iter().enumerate() {
            let h = block.input_norm.forward(&x)?;
            let h = block
                .attn
                .forward(&h, i, start_pos, &self.rope, kv, mask.as_ref())?;
            x = (x + h)?;
            let h = block.mlp.forward(&block.post_attn_norm.forward(&x)?)?;
            x = (x + h)?;
        }
        self.norm.forward(&x)
    }

    /// Logits for the last position only, shape `(vocab,)`, in f32.
    pub fn forward(&self, tokens: &[u32], kv: &mut dyn KvStore) -> Result<Tensor> {
        let h = self.hidden(tokens, kv)?;
        let last = h.narrow(1, tokens.len() - 1, 1)?;
        self.lm_head
            .forward(&last)?
            .to_dtype(DType::F32)?
            .flatten_all()
    }

    /// Logits for every position, shape `(seq, vocab)`, in f32.
    pub fn forward_all_logits(&self, tokens: &[u32], kv: &mut dyn KvStore) -> Result<Tensor> {
        let h = self.hidden(tokens, kv)?;
        self.lm_head.forward(&h)?.to_dtype(DType::F32)?.squeeze(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::ContiguousKvCache;
    use candle_nn::VarMap;

    fn tiny_model() -> Result<Llama> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        Llama::new(LlamaConfig::tiny(), vb, 64)
    }

    /// Prefill followed by token-by-token decode through the cache must reproduce the logits
    /// of one full-sequence forward pass.
    #[test]
    fn incremental_decode_matches_full_forward() -> Result<()> {
        let model = tiny_model()?;
        let cfg = model.config().clone();
        let tokens: Vec<u32> = vec![1, 5, 9, 33, 2, 70, 41, 8, 96, 3];
        let new_cache = || ContiguousKvCache::new(&cfg, 64, DType::F32, &Device::Cpu);

        let full = model.forward_all_logits(&tokens, &mut new_cache()?)?;

        let mut kv = new_cache()?;
        let prefill = 4;
        let mut rows = vec![model.forward_all_logits(&tokens[..prefill], &mut kv)?];
        for &t in &tokens[prefill..] {
            rows.push(model.forward(&[t], &mut kv)?.unsqueeze(0)?);
        }
        let incremental = Tensor::cat(&rows, 0)?;
        let full_tail = full.narrow(0, prefill - 1, tokens.len() - prefill + 1)?;
        let inc_tail = incremental.narrow(0, prefill - 1, tokens.len() - prefill + 1)?;
        let diff = (full_tail - inc_tail)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "max logit diff {diff}");
        assert_eq!(kv.seq_len(), tokens.len());
        Ok(())
    }

    #[test]
    fn rejects_sequences_past_max_len() -> Result<()> {
        let model = tiny_model()?;
        let mut kv = ContiguousKvCache::new(model.config(), 128, DType::F32, &Device::Cpu)?;
        assert!(model.forward(&vec![1; 65], &mut kv).is_err());
        Ok(())
    }
}
