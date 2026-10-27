//! Llama-architecture weights and forward pass on candle.

pub mod attention;
pub mod config;
pub mod kv;
pub mod layers;
pub mod llama;
pub mod paged;
pub mod paged_attention;
pub mod rope;

pub use config::LlamaConfig;
pub use kv::{ContiguousKvCache, KvBatch, KvStore, PerSeqKv};
pub use llama::Llama;
pub use paged::{AttentionImpl, PagedBatch, PagedKvCache, PagedSeq};

use candle_core::Device;

/// Picks a device by name: `cpu`, `metal`, or `cuda`.
pub fn device(name: &str) -> anyhow::Result<Device> {
    Ok(match name {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        "cuda" => Device::new_cuda(0)?,
        other => anyhow::bail!("unknown device {other:?}; expected cpu, metal or cuda"),
    })
}
