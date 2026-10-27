use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// A bias-free linear layer. M4 adds a `Quantized(QMatMul)` variant for GGUF weights.
#[derive(Debug, Clone)]
pub enum Linear {
    /// Weight of shape `(out, in)`, as stored in HF checkpoints.
    Dense(Tensor),
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
