use std::sync::Arc;

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// A bias-free linear layer.
#[derive(Debug, Clone)]
pub enum Linear {
    /// Weight of shape `(out, in)`, as stored in HF checkpoints.
    Dense(Tensor),
    /// GGUF block-quantized weight; takes and returns f32 activations.
    Quantized(QMatMul),
}

impl Linear {
    pub fn load(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self::Dense(vb.get((out_dim, in_dim), "weight")?))
    }
}

impl Module for Linear {
    /// Flattens leading dims so the matmul is 2-D. `broadcast_matmul` on a 3-D input
    /// broadcasts the weight, and on Metal that materializes a transposed copy of the whole
    /// matrix every call (about 25x slower per layer at decode).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (lead, in_dim) = match x.dims() {
            [lead @ .., in_dim] => (lead.to_vec(), *in_dim),
            [] => candle_core::bail!("linear on a scalar"),
        };
        let rows = lead.iter().product::<usize>();
        let y = match self {
            Self::Dense(w) => x.reshape((rows, in_dim))?.matmul(&w.t()?)?,
            Self::Quantized(q) => {
                // candle 0.11's batched Metal quantized matmul ignores the input's start offset
                // (it rebuilds the layout from the shape), so a narrowed view would be read
                // from the start of its buffer. Hand it fresh storage instead (`copy` would
                // keep the offset; `force_contiguous` materializes a new buffer).
                let x = x.reshape((rows, in_dim))?;
                let x = match x.is_contiguous() && x.layout().start_offset() == 0 {
                    true => x,
                    false => x.force_contiguous()?,
                };
                q.forward(&x)?
            }
        };
        let mut out = lead;
        out.push(y.dim(1)?);
        y.reshape(out)
    }
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            eps: eps as f32,
        }
    }

    pub fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps: eps as f32,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps)
    }
}

/// SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.
#[derive(Debug, Clone)]
pub struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Self { gate, up, down }
    }

    pub fn load(vb: VarBuilder, hidden: usize, intermediate: usize) -> Result<Self> {
        Ok(Self {
            gate: Linear::load(vb.pp("gate_proj"), hidden, intermediate)?,
            up: Linear::load(vb.pp("up_proj"), hidden, intermediate)?,
            down: Linear::load(vb.pp("down_proj"), intermediate, hidden)?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(gate * self.up.forward(x)?)?)
    }
}

/// Token embedding table, dense or kept block-quantized (looked up row by row on the device,
/// so a quantized checkpoint never holds a dequantized copy of the 128k x hidden table).
#[derive(Debug, Clone)]
pub enum Embedding {
    Dense(Tensor),
    Quantized(Arc<QTensor>),
}

impl Embedding {
    /// Rows for `ids`, shape `(ids.len(), hidden)`; f32 when quantized.
    pub fn lookup(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(t) => t.index_select(ids, 0),
            Self::Quantized(q) => q.embedding(ids),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Device};

    /// Rows taken from the middle of a tensor (a view with a start offset) must give the
    /// same result as the same rows copied out.
    fn quantized_matmul_honors_views(dev: &Device) -> Result<()> {
        let w = Tensor::randn(0f32, 1.0, (96, 64), dev)?;
        let q = Linear::Quantized(QMatMul::from_qtensor(QTensor::quantize(
            &w,
            GgmlDType::Q8_0,
        )?)?);
        let x = Tensor::randn(0f32, 1.0, (40, 64), dev)?;
        let view = x.narrow(0, 13, 20)?;
        let got = q.forward(&view)?;
        // Rebuilt from host data, so it shares nothing with `x`'s buffer or layout.
        let fresh = Tensor::from_vec(view.flatten_all()?.to_vec1::<f32>()?, (20, 64), dev)?;
        let want = q.forward(&fresh)?;
        let diff = (got - want)?
            .abs()?
            .max_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        assert!(
            diff == 0.0,
            "offset view gave different rows: max diff {diff}"
        );
        Ok(())
    }

    #[test]
    fn quantized_matmul_honors_views_cpu() -> Result<()> {
        quantized_matmul_honors_views(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn quantized_matmul_honors_views_metal() -> Result<()> {
        quantized_matmul_honors_views(&Device::new_metal(0)?)
    }
}
