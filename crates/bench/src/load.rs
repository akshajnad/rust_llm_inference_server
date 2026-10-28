//! Streaming load generation against an OpenAI-compatible `/v1/completions` endpoint.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::prompts::Prompt;
use crate::stats::Dist;

#[derive(Debug, Clone, Default)]
pub struct RequestResult {
    pub error: Option<String>,
    /// Send to first token, seconds.
    pub ttft: f64,
    /// Send to last token, seconds.
    pub e2e: f64,
    pub tokens: usize,
    /// Gaps between successive token-bearing chunks, seconds.
    pub gaps: Vec<f64>,
    pub disconnected: bool,
}

impl RequestResult {
    /// Time per output token after the first.
    pub fn tpot(&self) -> Option<f64> {
        (self.tokens > 1).then(|| (self.e2e - self.ttft) / (self.tokens - 1) as f64)
    }
}

/// An OpenAI-compatible server. Every request uses its own connection (`Connection: close`),
/// as vLLM's benchmark does, so connection reuse can't skew either engine.
#[derive(Clone)]
pub struct Target {
    pub url: String,
    pub model: String,
}

impl Target {
    pub fn new(url: &str, model: &str) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    pub async fn get_json(&self, path: &str) -> Result<Value> {
        let resp = crate::http::request(&self.url, "GET", path, None).await?;
        Ok(serde_json::from_str(&resp.text().await?)?)
    }

    /// Streams one completion. With `hang_up_after`, drops the connection after that many
    /// token chunks (soak testing).
    pub async fn request(&self, p: &Prompt, hang_up_after: Option<usize>) -> RequestResult {
        match self.request_inner(p, hang_up_after).await {
            Ok(r) => r,
            Err(e) => RequestResult {
                error: Some(format!("{e:#}")),
                ..Default::default()
            },
        }
    }

    async fn request_inner(
        &self,
        p: &Prompt,
        hang_up_after: Option<usize>,
    ) -> Result<RequestResult> {
        let body = json!({
            "model": self.model,
            "prompt": p.prompt,
            "max_tokens": p.output_tokens,
            "temperature": 0,
            "stream": true,
            "ignore_eos": true,
            "stream_options": {"include_usage": true},
        });
        let start = Instant::now();
        let body = serde_json::to_vec(&body)?;
        let mut resp =
            crate::http::request(&self.url, "POST", "/v1/completions", Some(&body)).await?;
        if resp.status != 200 {
            let status = resp.status;
            anyhow::bail!("HTTP {status}: {}", resp.text().await.unwrap_or_default());
        }
        let mut r = RequestResult::default();
        let (mut chunks, mut last, mut usage_tokens, mut buf) =
            (0usize, None::<Instant>, None, Vec::new());
        'read: while let Some(bytes) = resp.next().await? {
            buf.extend_from_slice(&bytes);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                if data == "[DONE]" {
                    break 'read;
                }
                let v: Value = serde_json::from_str(data)?;
                if let Some(e) = v.get("error") {
                    anyhow::bail!("stream error: {e}");
                }
                if let Some(n) = v["usage"]["completion_tokens"].as_u64() {
                    usage_tokens = Some(n as usize);
                }
                let text = v["choices"][0]["text"].as_str().unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                let now = Instant::now();
                match last {
                    None => r.ttft = (now - start).as_secs_f64(),
                    Some(prev) => r.gaps.push((now - prev).as_secs_f64()),
                }
                last = Some(now);
                chunks += 1;
                if hang_up_after.is_some_and(|n| chunks >= n) {
                    r.disconnected = true;
                    break 'read;
                }
            }
        }
        r.tokens = usage_tokens.unwrap_or(chunks);
        match last {
            Some(t) => r.e2e = (t - start).as_secs_f64(),
            // Every token decoded to empty text (e.g. special tokens after an ignored EOS):
            // no chunk carried text, but the usage report still counts the tokens.
            None if r.tokens > 0 => {
                r.e2e = start.elapsed().as_secs_f64();
                r.ttft = r.e2e;
            }
            None if r.disconnected => {}
            None => anyhow::bail!("no tokens received"),
        }
        Ok(r)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelResult {
    /// "concurrency" (closed loop) or "rate" (open loop, requests/s).
    pub mode: String,
    pub value: f64,
    pub requests: usize,
    pub errors: usize,
    pub first_error: Option<String>,
    pub duration_s: f64,
    pub output_tokens: usize,
    pub throughput_tok_s: f64,
    pub request_throughput: f64,
    pub ttft_ms: Dist,
    /// Gaps between streamed tokens.
    pub itl_ms: Dist,
    /// Per-request mean time per output token.
    pub tpot_ms: Dist,
    pub e2e_ms: Dist,
    pub footprint_peak_bytes: Option<u64>,
    pub footprint_mean_bytes: Option<u64>,
}

/// Samples `pid`'s footprint every 200 ms until `stop` is set; returns (peak, mean).
pub fn spawn_footprint_sampler(
    pid: Option<u32>,
    stop: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<Option<(u64, u64)>> {
    tokio::spawn(async move {
        let pid = pid?;
        let mut samples = Vec::new();
        while !stop.load(Ordering::Relaxed) {
            samples.extend(crate::sys::footprint_bytes(pid));
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let peak = *samples.iter().max()?;
        Some((peak, samples.iter().sum::<u64>() / samples.len() as u64))
    })
}

fn summarize(
    mode: &str,
    value: f64,
    results: Vec<RequestResult>,
    duration: Duration,
    mem: Option<(u64, u64)>,
) -> LevelResult {
    let ok: Vec<&RequestResult> = results.iter().filter(|r| r.error.is_none()).collect();
    let tokens: usize = ok.iter().map(|r| r.tokens).sum();
    let secs = duration.as_secs_f64();
    let collect =
        |f: &dyn Fn(&RequestResult) -> Vec<f64>| ok.iter().flat_map(|r| f(r)).collect::<Vec<f64>>();
    LevelResult {
        mode: mode.into(),
        value,
        requests: results.len(),
        errors: results.len() - ok.len(),
        first_error: results.iter().find_map(|r| r.error.clone()),
        duration_s: secs,
        output_tokens: tokens,
        throughput_tok_s: tokens as f64 / secs,
        request_throughput: ok.len() as f64 / secs,
        ttft_ms: Dist::of_secs(&collect(&|r| vec![r.ttft])),
        itl_ms: Dist::of_secs(&collect(&|r| r.gaps.clone())),
        tpot_ms: Dist::of_secs(&collect(&|r| r.tpot().into_iter().collect())),
        e2e_ms: Dist::of_secs(&collect(&|r| vec![r.e2e])),
        footprint_peak_bytes: mem.map(|m| m.0),
        footprint_mean_bytes: mem.map(|m| m.1),
    }
}

/// Closed loop: `concurrency` clients each send the next prompt as soon as their previous
/// request finishes, until `requests` have been sent. Prompts are taken in order, so every
/// engine sees the same requests.
pub async fn closed_loop(
    t: &Target,
    prompts: &[Prompt],
    concurrency: usize,
    requests: usize,
    pid: Option<u32>,
) -> LevelResult {
    let next = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = spawn_footprint_sampler(pid, stop.clone());
    let prompts = Arc::new(prompts.to_vec());
    let start = Instant::now();
    let workers: Vec<_> = (0..concurrency)
        .map(|_| {
            let (t, next, prompts) = (t.clone(), next.clone(), prompts.clone());
            tokio::spawn(async move {
                let mut out = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= requests {
                        return out;
                    }
                    out.push(t.request(&prompts[i % prompts.len()], None).await);
                }
            })
        })
        .collect();
    let mut results = Vec::with_capacity(requests);
    for w in workers {
        results.extend(w.await.expect("worker panicked"));
    }
    let duration = start.elapsed();
    stop.store(true, Ordering::Relaxed);
    summarize(
        "concurrency",
        concurrency as f64,
        results,
        duration,
        sampler.await.ok().flatten(),
    )
}

/// Open loop: Poisson arrivals at `rate` requests/s (seeded), regardless of completions.
pub async fn open_loop(
    t: &Target,
    prompts: &[Prompt],
    rate: f64,
    requests: usize,
    seed: u64,
    pid: Option<u32>,
) -> LevelResult {
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = spawn_footprint_sampler(pid, stop.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let start = Instant::now();
    let mut tasks = Vec::with_capacity(requests);
    let mut at = 0.0;
    for i in 0..requests {
        let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        at += -u.ln() / rate;
        tokio::time::sleep_until((start + Duration::from_secs_f64(at)).into()).await;
        let (t, p) = (t.clone(), prompts[i % prompts.len()].clone());
        tasks.push(tokio::spawn(async move { t.request(&p, None).await }));
    }
    let mut results = Vec::with_capacity(requests);
    for task in tasks {
        results.push(task.await.expect("request task panicked"));
    }
    let duration = start.elapsed();
    stop.store(true, Ordering::Relaxed);
    summarize(
        "rate",
        rate,
        results,
        duration,
        sampler.await.ok().flatten(),
    )
}
