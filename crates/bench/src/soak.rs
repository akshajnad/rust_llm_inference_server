//! Long run at high concurrency with random client hang-ups. Passes when every KV block comes
//! back (utilization 0 once idle) and the server's memory footprint stops growing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};
use serde::Serialize;

use crate::load::Target;
use crate::prompts::Prompt;

#[derive(Debug, Serialize)]
pub struct Sample {
    pub t_s: f64,
    pub footprint_bytes: Option<u64>,
    pub kv_utilization: Option<f64>,
    pub running: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct SoakReport {
    pub duration_s: f64,
    pub concurrency: usize,
    pub requests: usize,
    pub hang_ups: usize,
    pub errors: usize,
    pub first_error: Option<String>,
    /// Mean footprint around the 1/6 mark (10 min of an hour) and over the last 1/6.
    pub footprint_early_bytes: Option<u64>,
    pub footprint_late_bytes: Option<u64>,
    pub footprint_growth_pct: Option<f64>,
    pub kv_utilization_after_idle: Option<f64>,
    pub passed: bool,
    pub samples: Vec<Sample>,
}

async fn health(t: &Target) -> Option<serde_json::Value> {
    t.get_json("/health").await.ok()
}

pub async fn run(
    t: &Target,
    prompts: &[Prompt],
    duration: Duration,
    concurrency: usize,
    hang_up_prob: f64,
    pid: Option<u32>,
) -> SoakReport {
    let start = Instant::now();
    let done = Arc::new(AtomicBool::new(false));
    let (requests, hang_ups, errors) = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let first_error = Arc::new(std::sync::Mutex::new(None::<String>));
    let prompts = Arc::new(prompts.to_vec());
    let workers: Vec<_> = (0..concurrency)
        .map(|w| {
            let (t, done, prompts) = (t.clone(), done.clone(), prompts.clone());
            let (requests, hang_ups, errors, first_error) = (
                requests.clone(),
                hang_ups.clone(),
                errors.clone(),
                first_error.clone(),
            );
            tokio::spawn(async move {
                let mut rng = ChaCha8Rng::seed_from_u64(w as u64);
                while !done.load(Ordering::Relaxed) {
                    let i = requests.fetch_add(1, Ordering::Relaxed);
                    let p = &prompts[i % prompts.len()];
                    let roll = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                    let hang = (roll < hang_up_prob)
                        .then(|| 1 + (rng.next_u64() as usize) % p.output_tokens.max(2) / 2);
                    let r = t.request(p, hang).await;
                    if r.disconnected {
                        hang_ups.fetch_add(1, Ordering::Relaxed);
                    }
                    if let Some(e) = r.error {
                        errors.fetch_add(1, Ordering::Relaxed);
                        first_error.lock().unwrap().get_or_insert(e);
                    }
                }
            })
        })
        .collect();

    let mut samples = Vec::new();
    let interval = (duration / 120).clamp(Duration::from_secs(1), Duration::from_secs(30));
    while start.elapsed() < duration {
        let h = health(t).await;
        samples.push(Sample {
            t_s: start.elapsed().as_secs_f64(),
            footprint_bytes: pid.and_then(crate::sys::footprint_bytes),
            kv_utilization: h.as_ref().and_then(|h| h["kv_cache_utilization"].as_f64()),
            running: h.as_ref().and_then(|h| h["running"].as_u64()),
        });
        tokio::time::sleep(interval).await;
    }
    done.store(true, Ordering::Relaxed);
    for w in workers {
        let _ = w.await;
    }
    // Idle: wait for the engine to drain, then every block must be free.
    let mut kv_after = None;
    for _ in 0..120 {
        if let Some(h) = health(t).await {
            kv_after = h["kv_cache_utilization"].as_f64();
            if h["running"].as_u64() == Some(0) && h["waiting"].as_u64() == Some(0) {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let total = duration.as_secs_f64();
    let window_mean = |lo: f64, hi: f64| {
        let v: Vec<u64> = samples
            .iter()
            .filter(|s| s.t_s >= lo && s.t_s <= hi)
            .filter_map(|s| s.footprint_bytes)
            .collect();
        (!v.is_empty()).then(|| v.iter().sum::<u64>() / v.len() as u64)
    };
    let early = window_mean(total * 0.15, total * 0.2);
    let late = window_mean(total * 5.0 / 6.0, total);
    let growth = early
        .zip(late)
        .map(|(e, l)| (l as f64 - e as f64) / e as f64 * 100.0);
    let errors = errors.load(Ordering::Relaxed);
    SoakReport {
        duration_s: start.elapsed().as_secs_f64(),
        concurrency,
        requests: requests.load(Ordering::Relaxed),
        hang_ups: hang_ups.load(Ordering::Relaxed),
        errors,
        first_error: first_error.lock().unwrap().clone(),
        footprint_early_bytes: early,
        footprint_late_bytes: late,
        footprint_growth_pct: growth,
        kv_utilization_after_idle: kv_after,
        passed: errors == 0 && kv_after == Some(0.0) && growth.is_none_or(|g| g < 5.0),
        samples,
    }
}
