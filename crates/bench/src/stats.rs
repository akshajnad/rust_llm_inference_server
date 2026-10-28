use serde::Serialize;

/// Nearest-rank percentile of `xs` (any order); NaN when empty.
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * v.len() as f64).ceil().max(1.0) as usize;
    v[rank.min(v.len()) - 1]
}

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        f64::NAN
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// p50 / p90 / p99 / mean of one latency, in milliseconds.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Dist {
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub mean: f64,
}

impl Dist {
    pub fn of_secs(xs: &[f64]) -> Self {
        let ms: Vec<f64> = xs.iter().map(|x| x * 1e3).collect();
        Self {
            p50: percentile(&ms, 50.0),
            p90: percentile(&ms, 90.0),
            p99: percentile(&ms, 99.0),
            mean: mean(&ms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles() {
        let xs: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&xs, 50.0), 50.0);
        assert_eq!(percentile(&xs, 99.0), 99.0);
        assert_eq!(percentile(&xs, 100.0), 100.0);
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 50.0), 2.0);
        assert!(percentile(&[], 50.0).is_nan());
    }
}
