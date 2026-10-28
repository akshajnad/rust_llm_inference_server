//! What N completions of one prompt cost relative to one: N independent requests, or one
//! request with `n = N` (which rlis forks from a single prefill), against any
//! OpenAI-compatible server. Every run gets a prompt nobody has seen, so no run is helped by
//! an earlier run's cache, and every completion runs to exactly `max_tokens` (`ignore_eos`).

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::load::{Target, spawn_footprint_sampler};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// N concurrent requests with n = 1.
    Independent,
    /// One request with n = N.
    N,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkRun {
    /// Requested prompt length (the sweep value); `prompt_tokens` is what the server counted.
    pub nominal_prompt_tokens: usize,
    pub prompt_tokens: usize,
    pub n: usize,
    pub rep: usize,
    /// First request sent to last completion finished.
    pub wall_s: f64,
    /// Over the N completions.
    pub ttft_mean_s: f64,
    pub ttft_max_s: f64,
    pub output_tokens: usize,
    /// Output tokens over wall time.
    pub tok_s: f64,
    pub footprint_peak_bytes: Option<u64>,
    /// Peak KV blocks in use, from the server's `/health` (rlis only).
    pub kv_blocks_peak: Option<usize>,
    pub kv_block_size: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForkFile {
    pub provenance: crate::sys::Provenance,
    pub mode: Mode,
    pub max_tokens: usize,
    pub temperature: f32,
    pub runs: Vec<ForkRun>,
}

/// Prompts of a given token length cut from a corpus, each starting with a unique tag.
pub struct PromptMaker {
    tok: tokenizers::Tokenizer,
    ids: Vec<u32>,
    nonce: u64,
}

impl PromptMaker {
    pub fn new(corpus: &Path, tokenizer_dir: &Path) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(tokenizer_dir.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)?;
        let text = std::fs::read_to_string(corpus)
            .with_context(|| format!("reading {}", corpus.display()))?;
        let ids = tok
            .encode(text.as_str(), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64;
        Ok(Self { tok, ids, nonce })
    }

    /// Roughly `len` tokens: a unique tag (so no cache holds any of it) and a corpus slice.
    pub fn make(&mut self, len: usize) -> Result<String> {
        const TAG_TOKENS: usize = 12;
        anyhow::ensure!(
            len > TAG_TOKENS && self.ids.len() > 2 * len,
            "corpus too short"
        );
        self.nonce = self.nonce.wrapping_mul(6364136223846793005).wrapping_add(1);
        let start = (self.nonce >> 16) as usize % (self.ids.len() - len);
        let body = self
            .tok
            .decode(&self.ids[start..start + len - TAG_TOKENS], true)
            .map_err(anyhow::Error::msg)?;
        Ok(format!("Document {:016x}.\n{body}", self.nonce))
    }
}

/// When each choice of one streamed request produced its first text, and how many tokens
/// the server reports for the whole request.
struct StreamResult {
    first: Vec<Option<Instant>>,
    completion_tokens: usize,
    prompt_tokens: usize,
}

async fn stream(
    t: &Target,
    prompt: &str,
    n: usize,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
) -> Result<StreamResult> {
    let body = json!({
        "model": t.model,
        "prompt": prompt,
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": true,
        "ignore_eos": true,
        "stream_options": {"include_usage": true},
    });
    let body = serde_json::to_vec(&body)?;
    let mut resp = crate::http::request(&t.url, "POST", "/v1/completions", Some(&body)).await?;
    if resp.status != 200 {
        let status = resp.status;
        anyhow::bail!("HTTP {status}: {}", resp.text().await.unwrap_or_default());
    }
    let mut r = StreamResult {
        first: vec![None; n],
        completion_tokens: 0,
        prompt_tokens: 0,
    };
    let (mut per_choice, mut total) = (vec![None::<usize>; n], None);
    let mut buf = Vec::new();
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
            // rlis reports usage once for the whole request; llama.cpp once per choice, in a
            // chunk that also carries that choice.
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                let tokens = u["completion_tokens"].as_u64().unwrap_or(0) as usize;
                match v["choices"][0]["index"].as_u64() {
                    Some(i) if (i as usize) < n => per_choice[i as usize] = Some(tokens),
                    _ => total = Some(tokens),
                }
                r.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as usize;
            }
            for c in v["choices"].as_array().into_iter().flatten() {
                let i = c["index"].as_u64().unwrap_or(0) as usize;
                if !c["text"].as_str().unwrap_or_default().is_empty() && i < n {
                    r.first[i].get_or_insert_with(Instant::now);
                }
            }
        }
    }
    r.completion_tokens = total.unwrap_or_else(|| per_choice.iter().flatten().sum());
    Ok(r)
}

/// Polls rlis's `/health` for KV blocks in use until `stop`; returns (peak, block size), or
/// `None` for servers that don't report them.
fn spawn_kv_sampler(
    t: &Target,
    stop: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<Option<(usize, usize)>> {
    let t = t.clone();
    tokio::spawn(async move {
        let mut peak: Option<(usize, usize)> = None;
        while !stop.load(Ordering::Relaxed) {
            if let Ok(h) = t.get_json("/health").await
                && let (Some(used), Some(bs)) =
                    (h["kv_blocks_used"].as_u64(), h["kv_block_size"].as_u64())
            {
                let p = peak.get_or_insert((0, bs as usize));
                p.0 = p.0.max(used as usize);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        peak
    })
}

/// One measurement: N completions of `prompt`, as N requests or one request with n = N.
#[allow(clippy::too_many_arguments)]
pub async fn run_once(
    t: &Target,
    mode: Mode,
    prompt: &str,
    nominal: usize,
    n: usize,
    rep: usize,
    max_tokens: usize,
    temperature: f32,
    pid: Option<u32>,
) -> Result<ForkRun> {
    let stop = Arc::new(AtomicBool::new(false));
    let mem = spawn_footprint_sampler(pid, stop.clone());
    let kv = spawn_kv_sampler(t, stop.clone());
    let start = Instant::now();
    let results: Vec<StreamResult> = match mode {
        Mode::N => vec![stream(t, prompt, n, max_tokens, temperature, 0).await?],
        Mode::Independent => {
            let tasks: Vec<_> = (0..n)
                .map(|i| {
                    let (t, p) = (t.clone(), prompt.to_string());
                    tokio::spawn(async move {
                        stream(&t, &p, 1, max_tokens, temperature, i as u64).await
                    })
                })
                .collect();
            let mut out = Vec::new();
            for task in tasks {
                out.push(task.await??);
            }
            out
        }
    };
    let wall = start.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let ttfts: Vec<f64> = results
        .iter()
        .flat_map(|r| &r.first)
        .map(|f| f.map_or(wall, |f| (f - start).as_secs_f64()))
        .collect();
    let output_tokens: usize = results.iter().map(|r| r.completion_tokens).sum();
    anyhow::ensure!(
        output_tokens == n * max_tokens,
        "expected {} output tokens, the server reported {output_tokens}",
        n * max_tokens
    );
    let kv = kv.await.ok().flatten();
    Ok(ForkRun {
        nominal_prompt_tokens: nominal,
        prompt_tokens: results[0].prompt_tokens,
        n,
        rep,
        wall_s: wall,
        ttft_mean_s: ttfts.iter().sum::<f64>() / ttfts.len() as f64,
        ttft_max_s: ttfts.iter().copied().fold(0.0, f64::max),
        output_tokens,
        tok_s: output_tokens as f64 / wall,
        footprint_peak_bytes: mem.await.ok().flatten().map(|m| m.0),
        kv_blocks_peak: kv.map(|k| k.0),
        kv_block_size: kv.map(|k| k.1),
    })
}

/// Median over repetitions of one (prompt length, N) cell.
pub struct Cell {
    pub nominal: usize,
    pub n: usize,
    pub wall_s: f64,
    pub ttft_mean_s: f64,
    pub tok_s: f64,
    pub footprint_peak_bytes: Option<u64>,
    pub kv_blocks_peak: Option<usize>,
    /// wall(N) / wall(1) at the same prompt length.
    pub ratio: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

pub fn cells(f: &ForkFile) -> Vec<Cell> {
    let mut keys: Vec<(usize, usize)> = f
        .runs
        .iter()
        .map(|r| (r.nominal_prompt_tokens, r.n))
        .collect();
    keys.sort();
    keys.dedup();
    let cell = |nominal: usize, n: usize| {
        let runs: Vec<&ForkRun> = f
            .runs
            .iter()
            .filter(|r| (r.nominal_prompt_tokens, r.n) == (nominal, n))
            .collect();
        let m = |g: fn(&ForkRun) -> f64| median(runs.iter().map(|r| g(r)).collect());
        Cell {
            nominal,
            n,
            wall_s: m(|r| r.wall_s),
            ttft_mean_s: m(|r| r.ttft_mean_s),
            tok_s: m(|r| r.tok_s),
            footprint_peak_bytes: runs.iter().filter_map(|r| r.footprint_peak_bytes).max(),
            kv_blocks_peak: runs.iter().filter_map(|r| r.kv_blocks_peak).max(),
            ratio: f64::NAN,
        }
    };
    let mut out: Vec<Cell> = keys.iter().map(|&(p, n)| cell(p, n)).collect();
    for i in 0..out.len() {
        if let Some(base) = out.iter().find(|c| c.nominal == out[i].nominal && c.n == 1) {
            out[i].ratio = out[i].wall_s / base.wall_s;
        }
    }
    out
}
