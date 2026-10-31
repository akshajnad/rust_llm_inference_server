//! Loads llama.cpp GGUF checkpoints (Q8_0, Q4_K_M, ...) into the same `Llama` forward pass.
//! Three architectures: `llama`, `qwen2` (Llama plus Q/K/V biases) and `k2-horizon` (Llama with
//! RMS norms over equal groups of the hidden dimension).
//!
//! Differences from HF safetensors, all handled here:
//! - Config comes from `<arch>.*` metadata.
//! - Matrices stay block-quantized (`QMatMul`, f32 activations); the token embedding stays
//!   quantized too and doubles as the tied LM head.
//! - Llama 3 RoPE scaling is stored as `rope_freqs.weight`, per-frequency divisors of the base
//!   inverse frequencies.
//! - For `llama` the converter permutes Q/K rows for interleaved RoPE, so rotation uses
//!   adjacent pairs. `qwen2` and `k2-horizon` are converted unpermuted and keep HF's rotate-half
//!   (NEOX) layout.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::quantized::{QMatMul, QTensor, gguf_file};
use candle_core::{DType, Device};

use crate::config::LlamaConfig;
use crate::layers::{Embedding, Linear, QuantLinear, RmsNorm};
use crate::llama::{BlockWeights, Llama, LlamaParts};
use crate::rope::Rope;

/// Reads the model config from GGUF metadata.
pub fn config(content: &gguf_file::Content) -> Result<LlamaConfig> {
    let md = |k: &str| {
        content
            .metadata
            .get(k)
            .with_context(|| format!("GGUF metadata missing {k}"))
    };
    let arch = architecture(content)?;
    let u = |k: &str| -> Result<usize> { Ok(md(&format!("{arch}.{k}"))?.to_u32()? as usize) };
    let f = |k: &str| -> Result<f64> { Ok(md(&format!("{arch}.{k}"))?.to_f32()? as f64) };
    let embd = content
        .tensor_infos
        .get("token_embd.weight")
        .context("no token_embd.weight")?;
    Ok(LlamaConfig {
        hidden_size: u("embedding_length")?,
        intermediate_size: u("feed_forward_length")?,
        num_hidden_layers: u("block_count")?,
        num_attention_heads: u("attention.head_count")?,
        num_key_value_heads: u("attention.head_count_kv")?,
        // qwen2 files omit it; then it is hidden_size / heads.
        head_dim: u("rope.dimension_count").ok(),
        vocab_size: embd.shape.dims()[0],
        rms_norm_eps: f("attention.layer_norm_rms_epsilon")?,
        rope_theta: f("rope.freq_base")?,
        rope_scaling: None,
        max_position_embeddings: u("context_length")?,
        tie_word_embeddings: !content.tensor_infos.contains_key("output.weight"),
        bos_token_id: md("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.to_u32().ok()),
        attention_f32: arch != "llama",
    })
}

/// `general.architecture`, if it is one this loader supports.
pub fn architecture(content: &gguf_file::Content) -> Result<String> {
    let arch = content
        .metadata
        .get("general.architecture")
        .context("GGUF metadata missing general.architecture")?
        .to_string()?
        .clone();
    anyhow::ensure!(
        matches!(arch.as_str(), "llama" | "qwen2" | "k2-horizon"),
        "unsupported GGUF architecture {arch:?}"
    );
    Ok(arch)
}

/// Base inverse frequencies divided by the `rope_freqs` factors, when present.
pub fn inv_freqs(cfg: &LlamaConfig, rope_freqs: Option<&[f32]>) -> Vec<f32> {
    let base = crate::rope::inv_freqs(cfg);
    match rope_freqs {
        Some(f) => base.iter().zip(f).map(|(b, f)| b / f).collect(),
        None => base,
    }
}

struct Reader<'a> {
    content: &'a gguf_file::Content,
    file: std::fs::File,
    device: &'a Device,
}

impl Reader<'_> {
    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        self.content
            .tensor(&mut self.file, name, self.device)
            .with_context(|| format!("loading {name}"))
    }

    fn dense_f32(&mut self, name: &str) -> Result<candle_core::Tensor> {
        Ok(self
            .qtensor(name)?
            .dequantize(self.device)?
            .to_dtype(DType::F32)?)
    }

    fn linear(&mut self, name: &str) -> Result<Linear> {
        #[cfg(feature = "metal")]
        if self.device.is_metal()
            && self
                .content
                .tensor_infos
                .get(name)
                .is_some_and(|t| t.ggml_dtype == candle_core::quantized::GgmlDType::Q8_0)
        {
            let cpu = self.content.tensor(&mut self.file, name, &Device::Cpu)?;
            let (q, w) = upload_q8_0(cpu, self.device)?;
            let mut l = QuantLinear::new(QMatMul::from_qtensor(q)?);
            l.gemv = Some(w);
            return Ok(Linear::Quantized(l));
        }
        Ok(Linear::Quantized(QuantLinear::new(QMatMul::from_qtensor(
            self.qtensor(name)?,
        )?)))
    }
}

pub fn load(path: &Path, device: &Device, max_seq_len: usize) -> Result<Llama> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)?;
    let cfg = config(&content)?;
    let arch = architecture(&content)?;
    let interleaved_rope = arch == "llama";
    let groups = match content
        .metadata
        .get(&format!("{arch}.attention.group_norm_groups"))
    {
        Some(v) => v.to_u32()? as usize,
        None => 1,
    };
    let mut r = Reader {
        content: &content,
        file,
        device,
    };
    let rope_freqs = match content.tensor_infos.contains_key("rope_freqs.weight") {
        true => Some(r.dense_f32("rope_freqs.weight")?.to_vec1::<f32>()?),
        false => None,
    };
    let norm = |t| RmsNorm::grouped(t, cfg.rms_norm_eps, groups);
    let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let b = |n: &str| format!("blk.{i}.{n}.weight");
        let bias = |n: &str| format!("blk.{i}.{n}.bias");
        let qkv_bias = match content.tensor_infos.contains_key(&bias("attn_q")) {
            true => Some([
                r.dense_f32(&bias("attn_q"))?,
                r.dense_f32(&bias("attn_k"))?,
                r.dense_f32(&bias("attn_v"))?,
            ]),
            false => None,
        };
        blocks.push(BlockWeights {
            q: r.linear(&b("attn_q"))?,
            k: r.linear(&b("attn_k"))?,
            v: r.linear(&b("attn_v"))?,
            qkv_bias,
            o: r.linear(&b("attn_output"))?,
            gate: r.linear(&b("ffn_gate"))?,
            up: r.linear(&b("ffn_up"))?,
            down: r.linear(&b("ffn_down"))?,
            input_norm: norm(r.dense_f32(&b("attn_norm"))?)?,
            post_attn_norm: norm(r.dense_f32(&b("ffn_norm"))?)?,
        });
    }
    let final_norm = norm(r.dense_f32("output_norm.weight")?)?;
    let embed = Arc::new(r.qtensor("token_embd.weight")?);
    let lm_head = match cfg.tie_word_embeddings {
        true => Linear::Quantized(QuantLinear::new(QMatMul::from_arc(embed.clone())?)),
        false => r.linear("output.weight")?,
    };
    let max_pos = max_seq_len.min(cfg.max_position_embeddings);
    let rope = Rope::from_inv_freqs(
        inv_freqs(&cfg, rope_freqs.as_deref()),
        max_pos,
        DType::F32,
        device,
    )?
    .interleaved(interleaved_rope);
    // Quantized matmuls produce f32; keep KV at f16 on GPUs to hold the spec's 32 KB/token.
    let kv_dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };
    Ok(Llama::from_parts(LlamaParts {
        cfg,
        embed: Embedding::Quantized(embed),
        blocks,
        norm: final_norm,
        lm_head,
        rope,
        dtype: DType::F32,
        kv_dtype,
        device: device.clone(),
    }))
}

/// Uploads a CPU Q8_0 tensor to Metal through candle's public loader, keeping a handle on the
/// raw block buffer for the GEMV kernel. One copy of the weights serves both.
#[cfg(feature = "metal")]
pub fn upload_q8_0(cpu: QTensor, device: &Device) -> Result<(QTensor, crate::gemv::Q8Weight)> {
    use candle_core::quantized::GgmlDType;
    use candle_core::quantized::k_quants::BlockQ8_0;
    use candle_core::quantized::{QStorage, metal::load_quantized};
    anyhow::ensure!(
        cpu.dtype() == GgmlDType::Q8_0,
        "expected Q8_0, got {:?}",
        cpu.dtype()
    );
    let Device::Metal(metal) = device else {
        anyhow::bail!("upload_q8_0 needs a Metal device")
    };
    let (n, k) = cpu.shape().dims2()?;
    let bytes = cpu.data()?;
    let size = std::mem::size_of::<BlockQ8_0>();
    anyhow::ensure!(
        bytes.len() == n * k / 32 * size,
        "Q8_0 data has {} bytes for ({n}, {k})",
        bytes.len()
    );
    // Copy into properly aligned blocks (the byte buffer is only 1-aligned).
    let mut blocks: Vec<BlockQ8_0> = Vec::with_capacity(bytes.len() / size);
    // SAFETY: BlockQ8_0 is #[repr(C)] f16 + [i8; 32] (34 bytes, asserted by candle); every bit
    // pattern is valid, and the copy fills exactly the reserved capacity.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            blocks.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
        blocks.set_len(bytes.len() / size);
    }
    let storage = load_quantized(metal, &blocks)?;
    let QStorage::Metal(m) = &storage else {
        anyhow::bail!("expected Metal storage")
    };
    let weight = crate::gemv::Q8Weight {
        buffer: m.buffer().clone(),
        n,
        k,
    };
    Ok((QTensor::new(storage, (n, k))?, weight))
}
