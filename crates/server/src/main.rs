use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::DType;
use clap::{Parser, Subcommand, ValueEnum};
use engine::{
    AttentionImpl, ChatTemplate, EngineConfig, Event, Policy, Request, SamplingParams, Tokenizer,
};
use model::Llama;
use server::AppState;

#[derive(Parser)]
#[command(name = "rlis", about = "Rust LLM inference server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a completion for one prompt and stream it to stdout.
    Generate(GenerateArgs),
    /// Serve the OpenAI-compatible HTTP API.
    Serve(ServeArgs),
    /// Perplexity of a text file (WikiText-2: `make wikitext`), llama.cpp-compatible method.
    Perplexity(PerplexityArgs),
}

#[derive(clap::Args)]
struct PerplexityArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, default_value = "models/wikitext-2-raw/wiki.test.raw")]
    file: PathBuf,
    /// Window length; the second half of each window is scored.
    #[arg(long, default_value_t = 512)]
    ctx: usize,
    /// Score only the first N windows (0 = all).
    #[arg(long, default_value_t = 0)]
    max_windows: usize,
}

#[derive(clap::Args)]
struct ModelArgs {
    /// A `.gguf` file or a directory of HF safetensors.
    #[arg(long, default_value = "models/Llama-3.2-1B-Instruct-Q8_0.gguf")]
    model: PathBuf,
    /// HF directory with tokenizer.json, tokenizer_config.json and generation_config.json.
    #[arg(long, default_value = "models/llama-3.2-1b-instruct")]
    tokenizer: PathBuf,
    /// Weight and activation dtype for safetensors (GGUF runs quantized with f16 KV).
    #[arg(long, value_enum, default_value_t = Dtype::F16)]
    dtype: Dtype,
    #[arg(long, default_value = default_device())]
    device: String,
    /// Longest prompt plus completion, in tokens.
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
}

#[derive(clap::Args)]
struct GenerateArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    engine: EngineArgs,
    #[arg(long)]
    prompt: String,
    /// Wrap the prompt as one user message in the model's chat template.
    #[arg(long)]
    chat: bool,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(clap::Args)]
struct ServeArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Name reported in responses and `/v1/models`.
    #[arg(long, default_value = "llama-3.2-1b-instruct")]
    served_model_name: String,
    #[command(flatten)]
    engine: EngineArgs,
    /// Waiting requests allowed before new ones get 503.
    #[arg(long, default_value_t = 256)]
    queue_size: usize,
}

#[derive(clap::Args)]
struct EngineArgs {
    /// KV cache pool size, allocated at startup.
    #[arg(long, default_value_t = 1.0)]
    kv_cache_gb: f64,
    /// Tokens per KV block.
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    /// Most sequences in one step.
    #[arg(long, default_value_t = 64)]
    max_num_seqs: usize,
    /// Most tokens in one step; longer prompts are prefilled in chunks.
    #[arg(long, default_value_t = 2048)]
    max_batched_tokens: usize,
    #[arg(long, value_enum, default_value_t = SchedulerArg::Continuous)]
    scheduler: SchedulerArg,
    #[arg(long, value_enum, default_value_t = AttentionArg::V2)]
    attention: AttentionArg,
}

#[derive(Clone, Copy, ValueEnum)]
enum AttentionArg {
    /// Gather each sequence's KV blocks, then standard attention.
    V1,
    /// Paged-attention Metal kernel for decode (falls back to v1 elsewhere).
    V2,
}

#[derive(Clone, Copy, ValueEnum)]
enum SchedulerArg {
    /// Sequences join and leave the batch every step.
    Continuous,
    /// A batch runs until all of it finishes before new requests start (baseline).
    Static,
}

impl EngineArgs {
    fn config(&self, queue_size: usize) -> EngineConfig {
        EngineConfig {
            max_num_seqs: self.max_num_seqs,
            max_batched_tokens: self.max_batched_tokens,
            queue_size,
            kv_cache_bytes: engine::gib(self.kv_cache_gb),
            block_size: self.block_size,
            policy: match self.scheduler {
                SchedulerArg::Continuous => Policy::Continuous,
                SchedulerArg::Static => Policy::Static,
            },
            attention: match self.attention {
                AttentionArg::V1 => AttentionImpl::Gather,
                AttentionArg::V2 => AttentionImpl::Kernel,
            },
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

impl From<Dtype> for DType {
    fn from(d: Dtype) -> Self {
        match d {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::Bf16 => DType::BF16,
        }
    }
}

const fn default_device() -> &'static str {
    if cfg!(feature = "metal") {
        "metal"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "cpu"
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().command {
        Command::Generate(args) => run_generate(args),
        Command::Serve(args) => run_serve(args),
        Command::Perplexity(args) => run_perplexity(args),
    }
}

fn load(args: &ModelArgs) -> Result<(Llama, Arc<Tokenizer>)> {
    let device = model::device(&args.device)?;
    let start = std::time::Instant::now();
    let llama = model::load(&args.model, args.dtype.into(), &device, args.max_seq_len)?;
    let tokenizer = Arc::new(Tokenizer::load(&args.tokenizer)?);
    tracing::info!(model = %args.model.display(), device = %args.device, "loaded in {:.2?}", start.elapsed());
    Ok((llama, tokenizer))
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let (llama, tokenizer) = load(&args.model)?;
    let prompt = if args.chat {
        let template = ChatTemplate::load(&args.model.tokenizer)?;
        let msgs = [engine::ChatMessage {
            role: "user".into(),
            content: args.prompt.clone(),
        }];
        tokenizer.encode_raw(&template.render(&msgs, true)?)?
    } else {
        tokenizer.encode(&args.prompt)?
    };
    let engine = engine::spawn(llama, tokenizer, args.engine.config(1))?;
    let params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        max_tokens: args.max_tokens,
        stop: Vec::new(),
    };
    let (req, mut rx) = Request::new(prompt, params);
    engine.submit(req)?;
    let mut stdout = std::io::stdout().lock();
    loop {
        match rx.blocking_recv().context("engine stopped")? {
            Event::Text(t) => {
                stdout.write_all(t.as_bytes())?;
                stdout.flush()?;
            }
            Event::Finished(f) => {
                eprintln!(
                    "\n\nprompt {} tok | output {} tok ({:?}) | ttft {:.1?} | decode {:.1} tok/s",
                    f.prompt_tokens,
                    f.completion_tokens,
                    f.reason,
                    f.ttft,
                    f.decode_tokens_per_sec(),
                );
                return Ok(());
            }
            Event::Error(e) => anyhow::bail!(e),
        }
    }
}

fn run_perplexity(args: PerplexityArgs) -> Result<()> {
    let (llama, tokenizer) = load(&args.model)?;
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading {}", args.file.display()))?;
    let tokens = tokenizer.encode(&text)?;
    let bos = tokens[0];
    eprintln!("{} tokens from {}", tokens.len(), args.file.display());
    let report = engine::perplexity::perplexity(
        &llama,
        &tokens,
        bos,
        args.ctx,
        args.max_windows,
        |done, total, ppl| {
            if done % 50 == 0 || done == total {
                eprintln!("[{done}/{total}] ppl {ppl:.4}");
            }
        },
    )?;
    eprintln!(
        "PPL = {:.4} +/- {:.4} over {} tokens ({} windows of {}) in {:.1}s",
        report.ppl,
        report.ppl_stderr,
        report.tokens_scored,
        report.windows,
        report.ctx,
        report.seconds
    );
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn run_serve(args: ServeArgs) -> Result<()> {
    let (llama, tokenizer) = load(&args.model)?;
    let chat_template = match ChatTemplate::load(&args.model.tokenizer) {
        Ok(t) => Some(Arc::new(t)),
        Err(e) => {
            tracing::warn!("chat endpoint disabled: {e:#}");
            None
        }
    };
    let engine = engine::spawn(
        llama,
        tokenizer.clone(),
        args.engine.config(args.queue_size),
    )?;
    let metrics = Some(server::install_prometheus()?);
    let state = AppState {
        metrics,
        engine,
        tokenizer,
        chat_template,
        model_name: args.served_model_name,
    };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let addr = format!("{}:{}", args.host, args.port);
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .with_context(|| format!("binding {addr}"))?;
            tracing::info!("listening on http://{addr}");
            axum::serve(listener, server::router(state))
                .with_graceful_shutdown(shutdown_signal())
                .await?;
            Ok(())
        })
}

/// Resolves on Ctrl-C or SIGTERM so the server drains and exits cleanly either way.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
    tracing::info!("shutting down");
}
