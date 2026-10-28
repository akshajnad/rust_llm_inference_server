//! Benchmark tooling: a fixed prompt set, a streaming load generator for any OpenAI-compatible
//! server (ours or llama.cpp's), SVG charts, and a soak test.

mod chart;
mod http;
mod load;
mod prompts;
mod soak;
mod stats;
mod sys;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(
    name = "rlis-bench",
    about = "Benchmarks for OpenAI-compatible LLM servers"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate the fixed prompt set from a text corpus.
    Prompts {
        #[arg(long, default_value = "models/wikitext-2-raw/wiki.test.raw")]
        corpus: PathBuf,
        #[arg(long, default_value = "models/llama-3.2-1b-instruct")]
        tokenizer: PathBuf,
        #[arg(long, default_value_t = 256)]
        n: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "bench/prompts.json")]
        out: PathBuf,
    },
    /// Run a concurrency sweep (closed loop) or a request-rate sweep (open loop).
    Run(RunArgs),
    /// Print result files as one Markdown table.
    Table {
        #[arg(long, required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,
    },
    /// Draw an SVG chart from result files.
    Chart(ChartArgs),
    /// Long run with random client hang-ups; checks KV blocks and memory return to baseline.
    Soak {
        #[command(flatten)]
        target: TargetArgs,
        #[arg(long, default_value_t = 3600)]
        duration_s: u64,
        #[arg(long, default_value_t = 32)]
        concurrency: usize,
        #[arg(long, default_value_t = 0.25)]
        hang_up_prob: f64,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(clap::Args)]
struct TargetArgs {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,
    #[arg(long, default_value = "bench/prompts.json")]
    prompts: PathBuf,
    /// Server process id, to sample its memory footprint.
    #[arg(long)]
    pid: Option<u32>,
}

#[derive(clap::Args)]
struct RunArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// Series name in charts, e.g. "rlis Q8_0".
    #[arg(long)]
    label: String,
    #[arg(long, default_value = "rlis")]
    engine: String,
    #[arg(long, default_value = "Llama-3.2-1B-Instruct")]
    model: String,
    #[arg(long, default_value = "Q8_0")]
    quantization: String,
    /// Closed-loop concurrency levels.
    #[arg(long, value_delimiter = ',', default_value = "1,4,16,32,64")]
    concurrency: Vec<usize>,
    /// Open-loop request rates (req/s); replaces the concurrency sweep when given.
    #[arg(long, value_delimiter = ',')]
    rate: Vec<f64>,
    /// Requests per level; default max(16, 4 x concurrency) or 64 per rate.
    #[arg(long)]
    requests: Option<usize>,
    #[arg(long, default_value_t = 8)]
    warmup: usize,
    /// Text file prepended to every prompt (a shared system prompt), for prefix-caching runs.
    #[arg(long)]
    shared_prefix: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
}

#[derive(clap::Args)]
struct ChartArgs {
    #[arg(long, required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "")]
    title: String,
    /// x: concurrency | throughput_tok_s; y: throughput_tok_s | ttft_p99 | itl_p99 | tpot_p99 | tpot_p50 | footprint_peak_gb
    #[arg(long, default_value = "concurrency")]
    x: String,
    #[arg(long, default_value = "throughput_tok_s")]
    y: String,
}

#[derive(Serialize, Deserialize)]
struct RunFile {
    provenance: sys::Provenance,
    prompts: String,
    prompt_count: usize,
    levels: Vec<load::LevelResult>,
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Prompts {
            corpus,
            tokenizer,
            n,
            seed,
            out,
        } => {
            let input = prompts::LengthDist {
                median: 256.0,
                sigma: 0.6,
                min: 32,
                max: 1024,
            };
            let output = prompts::LengthDist {
                median: 128.0,
                sigma: 0.6,
                min: 16,
                max: 512,
            };
            let set = prompts::generate(&corpus, &tokenizer, n, seed, input, output)?;
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&set)?)?;
            let mean = |f: fn(&prompts::Prompt) -> usize| {
                set.iter().map(f).sum::<usize>() as f64 / set.len() as f64
            };
            eprintln!(
                "wrote {} prompts to {} (mean {:.0} in / {:.0} out tokens)",
                set.len(),
                out.display(),
                mean(|p| p.input_tokens),
                mean(|p| p.output_tokens)
            );
        }
        Command::Run(a) => run(a).await?,
        Command::Chart(a) => chart(a)?,
        Command::Table { inputs } => table(&inputs)?,
        Command::Soak {
            target,
            duration_s,
            concurrency,
            hang_up_prob,
            out,
        } => {
            let prompts = prompts::load(&target.prompts)?;
            let t = load::Target::new(&target.url, "default");
            eprintln!(
                "soak: {duration_s}s at concurrency {concurrency}, hang-up probability {hang_up_prob}"
            );
            let report = soak::run(
                &t,
                &prompts,
                Duration::from_secs(duration_s),
                concurrency,
                hang_up_prob,
                target.pid,
            )
            .await;
            std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
            eprintln!(
                "soak {}: {} requests, {} hang-ups, {} errors; footprint {:?} -> {:?} ({:+.2?}%); kv after idle {:?}",
                if report.passed { "PASSED" } else { "FAILED" },
                report.requests,
                report.hang_ups,
                report.errors,
                report.footprint_early_bytes.map(|b| b / (1 << 20)),
                report.footprint_late_bytes.map(|b| b / (1 << 20)),
                report.footprint_growth_pct,
                report.kv_utilization_after_idle
            );
        }
    }
    Ok(())
}

async fn run(a: RunArgs) -> Result<()> {
    let mut prompts = prompts::load(&a.target.prompts)
        .with_context(|| format!("reading {}", a.target.prompts.display()))?;
    if let Some(path) = &a.shared_prefix {
        let prefix =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        for p in &mut prompts {
            p.prompt = format!("{prefix}{}", p.prompt);
        }
    }
    let t = load::Target::new(&a.target.url, &a.model);
    if a.warmup > 0 {
        eprintln!("warmup: {} requests", a.warmup);
        let w = load::closed_loop(
            &t,
            &prompts[prompts.len() - a.warmup.min(prompts.len())..],
            4,
            a.warmup,
            None,
        )
        .await;
        anyhow::ensure!(w.errors == 0, "warmup failed: {:?}", w.first_error);
    }
    let (gpu, gpu_driver) = sys::gpu();
    let provenance = sys::Provenance {
        label: a.label.clone(),
        engine: a.engine,
        model: a.model,
        quantization: a.quantization,
        gpu,
        gpu_driver,
        os: sys::os(),
        commit: sys::commit(),
        timestamp: sys::timestamp(),
    };
    let mut levels = Vec::new();
    if a.rate.is_empty() {
        for &c in &a.concurrency {
            let n = a.requests.unwrap_or((4 * c).max(16));
            let l = load::closed_loop(&t, &prompts, c, n, a.target.pid).await;
            report(&l);
            levels.push(l);
        }
    } else {
        for (i, &r) in a.rate.iter().enumerate() {
            let l = load::open_loop(
                &t,
                &prompts,
                r,
                a.requests.unwrap_or(64),
                i as u64,
                a.target.pid,
            )
            .await;
            report(&l);
            levels.push(l);
        }
    }
    let file = RunFile {
        provenance,
        prompts: a.target.prompts.display().to_string(),
        prompt_count: prompts.len(),
        levels,
    };
    if let Some(dir) = a.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&a.out, serde_json::to_string_pretty(&file)?)?;
    eprintln!("wrote {}", a.out.display());
    Ok(())
}

fn report(l: &load::LevelResult) {
    eprintln!(
        "{} {:>5}: {:>4} req {:>2} err | {:7.1} tok/s | ttft p50 {:7.1} p99 {:7.1} ms | itl p50 {:5.1} p99 {:6.1} ms | tpot p99 {:5.1} ms | peak {}",
        l.mode,
        l.value,
        l.requests,
        l.errors,
        l.throughput_tok_s,
        l.ttft_ms.p50,
        l.ttft_ms.p99,
        l.itl_ms.p50,
        l.itl_ms.p99,
        l.tpot_ms.p99,
        l.footprint_peak_bytes
            .map_or("-".into(), |b| format!("{:.2} GB", b as f64 / 1e9)),
    );
    if let Some(e) = &l.first_error {
        eprintln!("  first error: {e}");
    }
}

fn field(l: &load::LevelResult, name: &str) -> Result<f64> {
    Ok(match name {
        "concurrency" | "rate" => l.value,
        "throughput_tok_s" => l.throughput_tok_s,
        "ttft_p50" => l.ttft_ms.p50,
        "ttft_p99" => l.ttft_ms.p99,
        "itl_p50" => l.itl_ms.p50,
        "itl_p99" => l.itl_ms.p99,
        "tpot_p50" => l.tpot_ms.p50,
        "tpot_p99" => l.tpot_ms.p99,
        "footprint_peak_gb" => l.footprint_peak_bytes.unwrap_or(0) as f64 / 1e9,
        other => anyhow::bail!("unknown field {other}"),
    })
}

fn axis_label(name: &str) -> &str {
    match name {
        "concurrency" => "concurrent requests",
        "rate" => "request rate (req/s)",
        "throughput_tok_s" => "output throughput (tok/s)",
        "ttft_p50" => "time to first token p50 (ms)",
        "ttft_p99" => "time to first token p99 (ms)",
        "itl_p50" => "inter-token latency p50 (ms)",
        "itl_p99" => "inter-token latency p99 (ms)",
        "tpot_p50" => "time per output token p50 (ms)",
        "tpot_p99" => "time per output token p99 (ms)",
        "footprint_peak_gb" => "peak memory footprint (GB)",
        other => other,
    }
}

fn chart(a: ChartArgs) -> Result<()> {
    let mut series = Vec::new();
    for path in &a.inputs {
        let f: RunFile = serde_json::from_str(
            &std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
        )?;
        // Points stay in load order: a latency-vs-throughput curve can double back when
        // throughput drops at the highest load, and sorting by x would zig-zag it.
        let points = f
            .levels
            .iter()
            .map(|l| Ok((field(l, &a.x)?, field(l, &a.y)?)))
            .collect::<Result<Vec<_>>>()?;
        // Label points with their load level when the x axis isn't the load itself.
        let point_labels = match a.x.as_str() {
            "concurrency" | "rate" => Vec::new(),
            _ => f.levels.iter().map(|l| format!("{}", l.value)).collect(),
        };
        series.push(chart::Series {
            label: f.provenance.label,
            points,
            point_labels,
        });
    }
    let c = chart::Chart {
        title: a.title,
        x_label: axis_label(&a.x).into(),
        y_label: axis_label(&a.y).into(),
        log2_x: a.x == "concurrency",
        series,
    };
    if let Some(dir) = a.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&a.out, c.svg())?;
    eprintln!("wrote {}", a.out.display());
    Ok(())
}

fn table(inputs: &[PathBuf]) -> Result<()> {
    println!(
        "| configuration | concurrency | tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p99 (ms) | peak memory |"
    );
    println!("|---|---|---|---|---|---|---|");
    for path in inputs {
        let f: RunFile = serde_json::from_str(
            &std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
        )?;
        for l in &f.levels {
            let errors = if l.errors > 0 {
                format!(" ({} errors)", l.errors)
            } else {
                String::new()
            };
            println!(
                "| {} | {} | {:.0}{errors} | {:.0} / {:.0} | {:.1} / {:.1} | {:.0} | {} |",
                f.provenance.label,
                l.value,
                l.throughput_tok_s,
                l.ttft_ms.p50,
                l.ttft_ms.p99,
                l.tpot_ms.p50,
                l.tpot_ms.p99,
                l.itl_ms.p99,
                l.footprint_peak_bytes
                    .map_or("-".into(), |b| format!("{:.2} GB", b as f64 / 1e9)),
            );
        }
    }
    Ok(())
}
