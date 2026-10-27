use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::LlamaConfig;
use crate::kv::KvStore;
use crate::layers::Linear;
use crate::rope::Rope;

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn load(vb: VarBuilder, cfg: &LlamaConfig) -> Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim());
        let (nh, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        Ok(Self {
            q_proj: Linear::load(vb.pp("q_proj"), h, nh * hd)?,
            k_proj: Linear::load(vb.pp("k_proj"), h, nkv * hd)?,
            v_proj: Linear::load(vb.pp("v_proj"), h, nkv * hd)?,
            o_proj: Linear::load(vb.pp("o_proj"), nh * hd, h)?,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
        })
    }

    /// `x` is `(b, seq, hidden)`; `mask` is an additive `(seq, total)` causal mask, or `None`
    /// when `seq == 1`.
    pub fn forward(
        &self,
        x: &Tensor,
        layer: usize,
        start_pos: usize,
        rope: &Rope,
        kv: &mut dyn KvStore,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let split =
            |t: Tensor, heads: usize| t.reshape((b, seq, heads, self.head_dim))?.transpose(1, 2);
        let q = split(self.q_proj.forward(x)?, self.n_heads)?;
        let k = split(self.k_proj.forward(x)?, self.n_kv_heads)?;
        let v = split(self.v_proj.forward(x)?, self.n_kv_heads)?;
        let q = rope.apply(&q, start_pos)?;
        let k = rope.apply(&k, start_pos)?;
        let (k, v) = kv.append(layer, &k, &v)?;
        let out = gqa_attention(&q, &k, &v, mask, self.head_dim)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, seq, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

/// Grouped-query attention without materializing repeated K/V.
///
/// Query head `h` uses KV head `h / n_rep`, so `(b, n_heads, seq, d)` reshapes to
/// `(b, n_kv, n_rep * seq, d)` and each KV head is multiplied once against its whole group.
/// Softmax runs in f32, as in the reference implementation.
pub fn gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
) -> Result<Tensor> {
    let (b, n_heads, seq, d) = q.dims4()?;
    let (_, n_kv, total, _) = k.dims4()?;
    let n_rep = n_heads / n_kv;
    let dtype = q.dtype();
    let q = q.contiguous()?.reshape((b, n_kv, n_rep * seq, d))?;
    let scores = (q.matmul(&k.t()?)? * (1.0 / (head_dim as f64).sqrt()))?;
    let scores = scores.to_dtype(DType::F32)?;
    let scores = match mask {
        Some(m) => scores
            .reshape((b, n_kv, n_rep, seq, total))?
            .broadcast_add(m)?
            .reshape((b, n_kv, n_rep * seq, total))?,
        None => scores,
    };
    let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(dtype)?;
    probs
        .matmul(v)?
        .reshape((b, n_heads, seq, v.dim(D::Minus1)?))
}

/// Additive causal mask of shape `(seq, start_pos + seq)` in f32.
pub fn causal_mask(seq: usize, start_pos: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let total = start_pos + seq;
    let data: Vec<f32> = (0..seq)
        .flat_map(|i| {
            (0..total).map(move |j| {
                if j > start_pos + i {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_vec(data, (seq, total), dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        let (b, n_kv, t, d) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, n_kv, n_rep, t, d))?
            .reshape((b, n_kv * n_rep, t, d))
    }

    /// Straightforward attention with repeated K/V, the textbook formulation.
    fn naive(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let n_rep = q.dim(1)? / k.dim(1)?;
        let (k, v) = (repeat_kv(k, n_rep)?, repeat_kv(v, n_rep)?);
        let d = q.dim(D::Minus1)? as f64;
        let mut s = (q.matmul(&k.t()?)? / d.sqrt())?;
        if let Some(m) = mask {
            s = s.broadcast_add(m)?;
        }
        candle_nn::ops::softmax_last_dim(&s)?.matmul(&v)
    }

    #[test]
    fn grouped_matches_repeated_kv() -> Result<()> {
        let dev = Device::Cpu;
        let (b, nh, nkv, seq, start, d) = (1, 8, 2, 5, 3, 16);
        let q = Tensor::randn(0f32, 1.0, (b, nh, seq, d), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (b, nkv, start + seq, d), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (b, nkv, start + seq, d), &dev)?;
        let mask = causal_mask(seq, start, &dev)?;
        let got = gqa_attention(&q, &k, &v, Some(&mask), d)?;
        let want = naive(&q, &k, &v, Some(&mask))?;
        let diff = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max diff {diff}");
        Ok(())
    }

    #[test]
    fn repeat_kv_maps_heads_in_groups() -> Result<()> {
        let x = Tensor::arange(0f32, 4.0, &Device::Cpu)?.reshape((1, 2, 2, 1))?;
        let r = repeat_kv(&x, 3)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(r, [0., 1., 0., 1., 0., 1., 2., 3., 2., 3., 2., 3.]);
        Ok(())
    }

    #[test]
    fn causal_mask_offsets_by_start_pos() -> Result<()> {
        let m = causal_mask(2, 1, &Device::Cpu)?.to_vec2::<f32>()?;
        assert_eq!(m[0], [0.0, 0.0, f32::NEG_INFINITY]);
        assert_eq!(m[1], [0.0, 0.0, 0.0]);
        Ok(())
    }
}
