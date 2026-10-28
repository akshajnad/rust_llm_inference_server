//! Llama-architecture weights and forward pass on candle.

pub mod attention;
pub mod config;
pub mod gemv;
pub mod gguf;
pub mod kv;
pub mod layers;
pub mod llama;
pub mod paged;
pub mod paged_attention;
pub mod profile;
pub mod rope;
pub mod sampling;

pub use config::LlamaConfig;
pub use kv::{ContiguousKvCache, KvBatch, KvStore, PerSeqKv};
pub use llama::Llama;

/// Loads a GGUF file or a directory of HF safetensors; `dtype` applies to safetensors only.
pub fn load(
    path: &std::path::Path,
    dtype: candle_core::DType,
    device: &Device,
    max_seq_len: usize,
) -> anyhow::Result<Llama> {
    if path.extension().is_some_and(|e| e == "gguf") {
        gguf::load(path, device, max_seq_len)
    } else {
        Llama::load(path, dtype, device, max_seq_len)
    }
}
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

/// Runs `f` inside an Objective-C autorelease pool on Metal builds (a plain call otherwise).
///
/// Metal returns autoreleased objects (command encoders, label strings). A pool is drained
/// only when it ends, and a long-lived thread without one keeps every such object until the
/// thread exits, so loops that drive the GPU should wrap each iteration.
pub fn autorelease_pool<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "metal")]
    {
        objc2::rc::autoreleasepool(|_| f())
    }
    #[cfg(not(feature = "metal"))]
    {
        f()
    }
}
