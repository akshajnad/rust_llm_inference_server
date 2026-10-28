//! Decode step time vs batch size and context length through the paged path, plus the
//! quantized matmul alone at the same row counts.
//! `cargo run --release -p model --features metal --example decode_profile [model] [contexts]
//! [batches]`, e.g. `... 2048,8192 1,2,4,8` (default: context 400, batches 1 to 64).
use std::time::Instant;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use model::{AttentionImpl, PagedBatch, PagedKvCache, PagedSeq};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    let path = std::env::args()
        .nth(1)
        .unwrap_or("models/Llama-3.2-1B-Instruct-Q8_0.gguf".into());
    let list = |i: usize, default: &str| -> Vec<usize> {
        std::env::args()
            .nth(i)
            .unwrap_or(default.into())
            .split(',')
            .map(|x| x.parse().expect("comma-separated numbers"))
            .collect()
    };
    let (contexts, batches) = (list(2, "400"), list(3, "1,2,4,8,16,32,64"));
    let max_ctx = *contexts.iter().max().unwrap();
    let max_b = *batches.iter().max().unwrap();
    let m = model::load(std::path::Path::new(&path), DType::F16, &dev, max_ctx + 16)?;
    let bs = 16;
    let per = (max_ctx + 8) / bs + 1;
    let mut cache = PagedKvCache::new(m.config(), max_b * per, bs, m.kv_dtype(), &dev)?;
    cache.set_attention(AttentionImpl::Kernel);
    let tables: Vec<Vec<u32>> = (0..max_b)
        .map(|i| ((i * per) as u32..((i + 1) * per) as u32).collect())
        .collect();
    for &ctx in &contexts {
        println!("{path}: decode step, context {ctx}");
        for &b in &batches {
            let tok = [1000u32];
            let toks: Vec<&[u32]> = (0..b).map(|_| &tok[..]).collect();
            let seqs: Vec<PagedSeq> = (0..b)
                .map(|i| PagedSeq {
                    block_table: &tables[i],
                    start_pos: ctx,
                    len: 1,
                })
                .collect();
            let run = || -> anyhow::Result<()> {
                m.forward_batch(&toks, &mut PagedBatch::new(&cache, &seqs)?)?
                    .argmax(1)?
                    .to_vec1::<u32>()?;
                Ok(())
            };
            run()?;
            let t = Instant::now();
            for _ in 0..10 {
                run()?;
            }
            let ms = t.elapsed().as_secs_f64() * 100.0;
            println!(
                "  batch {b:2}: {ms:6.1} ms/step  {:7.0} tok/s",
                b as f64 / ms * 1e3
            );
        }
    }
    if contexts != [400] {
        return Ok(());
    }
    println!("Q8_0 gate projection (2048 -> 8192), GEMV kernel vs candle per-row matrix-vector:");
    {
        let w = Tensor::randn(0f32, 0.02, (8192, 2048), &Device::Cpu)?;
        let (q, gw) = model::gguf::upload_q8_0(QTensor::quantize(&w, GgmlDType::Q8_0)?, &dev)?;
        let q = QMatMul::from_qtensor(q)?;
        for rows in [2usize, 3, 4, 5, 6, 7, 8] {
            let x = Tensor::randn(0f32, 1.0, (rows, 2048), &dev)?;
            let bench = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> anyhow::Result<f64> {
                f()?;
                dev.synchronize()?;
                let t = Instant::now();
                for _ in 0..200 {
                    f()?;
                }
                dev.synchronize()?;
                Ok(t.elapsed().as_secs_f64() * 1e3 / 200.0)
            };
            // Interleave the three paths and keep each one's best of 3, to damp contention.
            let (mut g, mut mv, mut mm) = (f64::MAX, f64::MAX, f64::MAX);
            for _ in 0..3 {
                g = g.min(bench(&|| model::gemv::gemv(&gw, &x))?);
                mv = mv.min(bench(&|| q.forward(&x.reshape((rows, 1, 2048))?))?);
                mm = mm.min(bench(&|| q.forward(&x))?);
            }
            println!(
                "  rows {rows}: gemv {g:6.3} ms   per-row mv {mv:6.3} ms   batched mm {mm:6.3} ms"
            );
        }
    }
    println!("Q8_0 gate projection (2048 -> 8192) alone:");
    let w = Tensor::randn(0f32, 0.02, (8192, 2048), &dev)?;
    let q = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
    for rows in [1usize, 2, 4, 8, 16, 32, 64] {
        let x = Tensor::randn(0f32, 1.0, (rows, 2048), &dev)?;
        q.forward(&x)?;
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..50 {
            q.forward(&x)?;
        }
        dev.synchronize()?;
        println!(
            "  rows {rows:2}: {:6.3} ms",
            t.elapsed().as_secs_f64() * 1e3 / 50.0
        );
    }
    Ok(())
}
