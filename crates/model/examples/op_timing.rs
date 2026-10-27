//! Times the matmul layouts a decode step uses, to find slow paths on a device.
//! `cargo run --release --features metal -p model --example op_timing`
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

fn time(name: &str, dev: &Device, f: impl Fn() -> Result<Tensor>) -> Result<()> {
    f()?;
    dev.synchronize()?;
    let n = 20;
    let start = Instant::now();
    for _ in 0..n {
        f()?;
    }
    dev.synchronize()?;
    println!(
        "{name:<44} {:>9.3} ms",
        start.elapsed().as_secs_f64() * 1e3 / n as f64
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = model::device(&std::env::args().nth(1).unwrap_or("metal".into())).unwrap();
    let dt = DType::F16;
    let w = Tensor::randn(0f32, 0.02, (8192, 2048), &dev)?.to_dtype(dt)?;
    let wt = w.t()?.contiguous()?;
    let x3 = Tensor::randn(0f32, 1.0, (1, 1, 2048), &dev)?.to_dtype(dt)?;
    let x2 = x3.reshape((1, 2048))?;
    time("x3.broadcast_matmul(w.t())", &dev, || {
        x3.broadcast_matmul(&w.t()?)
    })?;
    time("x2.matmul(w.t())", &dev, || x2.matmul(&w.t()?))?;
    time("x2.matmul(wt contiguous)", &dev, || x2.matmul(&wt))?;
    time("w.matmul(x2.t())", &dev, || w.matmul(&x2.t()?))?;
    time("w.t().contiguous()", &dev, || w.t()?.contiguous())?;
    let x64 = Tensor::randn(0f32, 1.0, (64, 2048), &dev)?.to_dtype(dt)?;
    time("x[64].matmul(w.t())", &dev, || x64.matmul(&w.t()?))?;
    Ok(())
}
