//! Where prefill time goes: one 1024-token forward pass vs its matmuls and attention alone.
//! `cargo run --release -p model --features metal --example prefill_profile`
use std::time::Instant;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use model::ContiguousKvCache;
use model::attention::{causal_mask, gqa_attention};

fn time<T>(dev: &Device, n: usize, mut f: impl FnMut() -> Result<T>) -> Result<f64> {
    f()?;
    dev.synchronize()?;
    let t = Instant::now();
    for _ in 0..n {
        f()?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / n as f64)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    let seq = 1024;
    // The 7 projections of one layer (in, out), times 16 layers.
    let shapes = [
        (2048, 2048),
        (2048, 512),
        (2048, 512),
        (2048, 2048),
        (2048, 8192),
        (2048, 8192),
        (8192, 2048),
    ];
    let flops: f64 = shapes
        .iter()
        .map(|(i, o)| 2.0 * (seq * i * o) as f64)
        .sum::<f64>()
        * 16.0;
    let _ = model::profile::take_report();
    for (name, dtype) in [("f16 dense", DType::F16), ("f32 dense", DType::F32)] {
        let mut ms = 0.0;
        for (i, o) in shapes {
            let w = Tensor::randn(0f32, 0.02, (o, i), &dev)?.to_dtype(dtype)?;
            let x = Tensor::randn(0f32, 1.0, (seq, i), &dev)?.to_dtype(dtype)?;
            ms += time(&dev, 5, || x.matmul(&w.t()?))? * 16.0;
        }
        println!(
            "{name:10} matmuls, 16 layers: {ms:7.1} ms ({:.1} TFLOPS)",
            flops / ms / 1e9
        );
    }
    let mut ms = 0.0;
    for (i, o) in shapes {
        let w = Tensor::randn(0f32, 0.02, (o, i), &dev)?;
        let q = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
        let x = Tensor::randn(0f32, 1.0, (seq, i), &dev)?;
        ms += time(&dev, 5, || q.forward(&x))? * 16.0;
    }
    println!(
        "Q8_0       matmuls, 16 layers: {ms:7.1} ms ({:.1} TFLOPS)",
        flops / ms / 1e9
    );

    for dtype in [DType::F16, DType::F32] {
        let q = Tensor::randn(0f32, 1.0, (1, 32, seq, 64), &dev)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1.0, (1, 8, seq, 64), &dev)?.to_dtype(dtype)?;
        let v = Tensor::randn(0f32, 1.0, (1, 8, seq, 64), &dev)?.to_dtype(dtype)?;
        let mask = causal_mask(seq, 0, &dev)?;
        let ms = time(&dev, 5, || gqa_attention(&q, &k, &v, Some(&mask)))? * 16.0;
        println!("{dtype:?} attention v1, 16 layers: {ms:7.1} ms");
    }

    for dtype in [DType::F16, DType::F32] {
        let q = Tensor::randn(0f32, 1.0, (1, 32, seq, 64), &dev)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1.0, (1, 8, seq, 64), &dev)?.to_dtype(dtype)?;
        let v = Tensor::randn(0f32, 1.0, (1, 8, seq, 64), &dev)?.to_dtype(dtype)?;
        let ms = time(&dev, 5, || {
            candle_nn::ops::sdpa(&q, &k, &v, None, true, 0.125, 1.0)
        })? * 16.0;
        println!("{dtype:?} fused sdpa, 16 layers: {ms:7.1} ms");
        let x = Tensor::randn(0f32, 1.0, (1, seq, 2048), &dev)?.to_dtype(dtype)?;
        let w = Tensor::ones(2048, dtype, &dev)?;
        let g = Tensor::randn(0f32, 1.0, (1, seq, 8192), &dev)?.to_dtype(dtype)?;
        let cs = Tensor::randn(0f32, 1.0, (seq, 32), &dev)?.to_dtype(dtype)?;
        let per_layer = [
            (
                "2x rms_norm",
                time(&dev, 10, || candle_nn::ops::rms_norm(&x, &w, 1e-5))? * 2.0,
            ),
            (
                "rope q+k",
                time(&dev, 10, || candle_nn::rotary_emb::rope_i(&q, &cs, &cs))? * 1.25,
            ),
            (
                "silu*up",
                time(&dev, 10, || candle_nn::ops::silu(&g)? * &g)?,
            ),
            ("2x residual add", time(&dev, 10, || &x + &x)? * 2.0),
            (
                "qkv/out transposes",
                time(&dev, 10, || {
                    x.reshape((1, seq, 32, 64))?.transpose(1, 2)?.contiguous()
                })? * 2.5,
            ),
        ];
        for (name, ms) in per_layer {
            println!("{dtype:?} {name:20} 16 layers: {:7.1} ms", ms * 16.0);
        }
    }

    for path in [
        "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "models/llama-3.2-1b-instruct",
    ] {
        let m = model::load(std::path::Path::new(path), DType::F16, &dev, 2048)?;
        let toks: Vec<u32> = (0..seq as u32).map(|i| 1000 + i % 5000).collect();
        let ms = time(&dev, 3, || {
            let mut kv = ContiguousKvCache::new(m.config(), seq, m.dtype(), &dev)?;
            m.forward(&toks, &mut kv)
        })?;
        println!(
            "full forward {seq} tokens ({path}): {ms:7.1} ms ({:.0} tok/s)",
            seq as f64 / ms * 1e3
        );
        if model::profile::enabled() {
            // Timings above ran with per-span syncs; this is the attribution, not the speed.
            let report = model::profile::take_report();
            let total: f64 = report.iter().map(|r| r.1.as_secs_f64()).sum();
            println!("  per-stage GPU time:");
            for (name, d, _) in report {
                println!("    {name:24} {:5.1}%", d.as_secs_f64() / total * 100.0);
            }
        }
    }
    Ok(())
}
