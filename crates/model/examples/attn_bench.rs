//! The paged decode attention op alone, per layer of Llama 3.2 1B (32 query / 8 KV heads of
//! 64, f16 KV), across context lengths and batch sizes, with the K/V bandwidth it reaches.
//! `cargo run --release -p model --features metal --example attn_bench`
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use model::paged_attention::PagedAttention;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    let (n_heads, n_kv, hd, bs) = (32usize, 8usize, 64usize, 16usize);
    for &ctx in &[400usize, 2048, 8192] {
        for &b in &[1usize, 8] {
            let per = ctx.div_ceil(bs);
            let slots = b * per * bs;
            let k = Tensor::randn(0f32, 1.0, (slots, n_kv, hd), &dev)?.to_dtype(DType::F16)?;
            let v = Tensor::randn(0f32, 1.0, (slots, n_kv, hd), &dev)?.to_dtype(DType::F16)?;
            let q = Tensor::randn(0f32, 1.0, (b, n_heads, hd), &dev)?.to_dtype(DType::F16)?;
            let tables: Vec<u32> = (0..(b * per) as u32).collect();
            let tables = Tensor::from_vec(tables, (b, per), &dev)?;
            let lens = Tensor::new(vec![ctx as u32; b], &dev)?;
            let op = PagedAttention {
                k_cache: k,
                v_cache: v,
                block_size: bs,
                max_context: ctx,
                pipelines: Default::default(),
            };
            q.apply_op3_no_bwd(&tables, &lens, &op)?;
            dev.synchronize()?;
            let n = 200;
            let t = Instant::now();
            for _ in 0..n {
                q.apply_op3_no_bwd(&tables, &lens, &op)?;
            }
            dev.synchronize()?;
            let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            let bytes = 2.0 * (b * ctx * n_kv * hd * 2) as f64;
            println!(
                "ctx {ctx:5} batch {b}: {us:7.1} us/layer  x16 = {:5.2} ms  ({:5.1} GB/s of K/V)",
                us * 16.0 / 1e3,
                bytes / us / 1e3
            );
        }
    }
    Ok(())
}
