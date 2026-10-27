use candle_core::{DType, Device, Result, Tensor};

use crate::config::LlamaConfig;

/// Precomputed cos/sin tables, shape `(max_positions, head_dim / 2)`.
///
/// Rotation uses the non-interleaved ("rotate_half") layout of HF checkpoints. GGUF files
/// converted by llama.cpp permute Q/K for the interleaved layout and need `rope_i` instead.
#[derive(Debug, Clone)]
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

/// Inverse frequencies, with Llama 3 scaling applied when configured.
/// Port of `_compute_llama3_parameters` in transformers' `modeling_rope_utils.py`,
/// done in f32 like the reference so the tables agree to the last bit.
pub fn inv_freqs(cfg: &LlamaConfig) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    let theta = cfg.rope_theta as f32;
    let base: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f32 / head_dim as f32))
        .collect();
    match &cfg.rope_scaling {
        Some(s) if s.rope_type == "llama3" => {
            let factor = s.factor as f32;
            let (low, high) = (s.low_freq_factor as f32, s.high_freq_factor as f32);
            let old_ctx = s.original_max_position_embeddings as f32;
            let low_freq_wavelen = old_ctx / low;
            let high_freq_wavelen = old_ctx / high;
            base.iter()
                .map(|&f| {
                    let wavelen = 2.0 * std::f32::consts::PI / f;
                    if wavelen < high_freq_wavelen {
                        f
                    } else if wavelen > low_freq_wavelen {
                        f / factor
                    } else {
                        let smooth = (old_ctx / wavelen - low) / (high - low);
                        (1.0 - smooth) * f / factor + smooth * f
                    }
                })
                .collect()
        }
        _ => base,
    }
}

impl Rope {
    pub fn new(
        cfg: &LlamaConfig,
        max_positions: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let inv = inv_freqs(cfg);
        let half = inv.len();
        let inv = Tensor::from_vec(inv, (1, half), dev)?;
        let pos = Tensor::arange(0u32, max_positions as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_positions, 1))?;
        let angles = pos.broadcast_mul(&inv)?;
        Ok(Self {
            cos: angles.cos()?.to_dtype(dtype)?,
            sin: angles.sin()?.to_dtype(dtype)?,
        })
    }

    pub fn max_positions(&self) -> usize {
        self.cos.dim(0).unwrap_or(0)
    }

    /// Gathers the cos/sin rows for each token's position, once per forward pass.
    pub fn batch(&self, positions: &Tensor) -> Result<RopeBatch> {
        Ok(RopeBatch {
            cos: self.cos.index_select(positions, 0)?.contiguous()?,
            sin: self.sin.index_select(positions, 0)?.contiguous()?,
        })
    }
}

/// cos/sin rows `(total_tokens, head_dim / 2)` for one batch, in token order.
#[derive(Debug, Clone)]
pub struct RopeBatch {
    cos: Tensor,
    sin: Tensor,
}

impl RopeBatch {
    /// Rotates `x` of shape `(1, heads, total_tokens, head_dim)`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, &self.cos, &self.sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from transformers' `_compute_llama3_parameters` for the Llama 3.2 1B
    /// config (theta 500000, head_dim 64, factor 32, low 1, high 4, original ctx 8192).
    #[test]
    fn llama3_scaling_matches_transformers() {
        let cfg: LlamaConfig = serde_json::from_str(
            r#"{"hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":16,
            "num_attention_heads":32,"num_key_value_heads":8,"head_dim":64,"vocab_size":128256,
            "rms_norm_eps":1e-5,"rope_theta":500000.0,"max_position_embeddings":131072,
            "rope_scaling":{"rope_type":"llama3","factor":32.0,"low_freq_factor":1.0,
            "high_freq_factor":4.0,"original_max_position_embeddings":8192}}"#,
        )
        .unwrap();
        let got = inv_freqs(&cfg);
        assert_eq!(got.len(), 32);
        for (i, want) in LLAMA32_INV_FREQ.iter().enumerate() {
            let rel = ((got[i] - want) / want).abs();
            assert!(rel < 1e-6, "inv_freq[{i}] = {} want {want}", got[i]);
        }
    }

    // Printed by scripts/parity/dump_reference.py (transformers 5.21.0).
    #[rustfmt::skip]
    #[allow(clippy::excessive_precision)]
    const LLAMA32_INV_FREQ: [f32; 32] = [
        1.000000000e+00, 6.636012793e-01, 4.403666258e-01, 2.922278345e-01,
        1.939227581e-01, 1.286873817e-01, 8.539710194e-02, 5.666961893e-02,
        3.760603070e-02, 2.495540865e-02, 1.656044088e-02, 1.098952908e-02,
        7.292665076e-03, 4.839421250e-03, 3.211446106e-03, 1.290548011e-03,
        4.295567051e-04, 9.708286234e-05, 1.946163866e-05, 1.291476747e-05,
        8.570255886e-06, 5.687232260e-06, 3.774054449e-06, 2.504467147e-06,
        1.661967417e-06, 1.102883630e-06, 7.318749340e-07, 4.856731266e-07,
        3.222932889e-07, 2.138742303e-07, 1.419272024e-07, 9.418306490e-08,
    ];
}
