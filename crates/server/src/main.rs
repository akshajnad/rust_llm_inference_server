use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::DType;
use clap::{Parser, Subcommand, ValueEnum};
use engine::{ChatTemplate, EngineConfig, Event, Request, SamplingParams, Tokenizer};
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
}

#[derive(clap::Args)]
struct ModelArgs {
    #[arg(long, default_value = "models/llama-3.2-1b-instruct")]
    model: PathBuf,
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
    #[arg(long, default_value_t = 32)]
    max_batch_size: usize,
    /// Waiting requests allowed before new ones get 503.
    #[arg(long, default_value_t = 256)]
    queue_size: usize,
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
    }
}

fn load(args: &ModelArgs) -> Result<(Llama, Arc<Tokenizer>)> {
    let device = model::device(&args.device)?;
    let start = std::time::Instant::now();
    let llama = Llama::load(&args.model, args.dtype.into(), &device, args.max_seq_len)?;
    let tokenizer = Arc::new(Tokenizer::load(&args.model)?);
    tracing::info!(model = %args.model.display(), device = %args.device, "loaded in {:.2?}", start.elapsed());
    Ok((llama, tokenizer))
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let (llama, tokenizer) = load(&args.model)?;
    let prompt = if args.chat {
        let template = ChatTemplate::load(&args.model.model)?;
        let msgs = [engine::ChatMessage {
            role: "user".into(),
            content: args.prompt.clone(),
        }];
        tokenizer.encode_raw(&template.render(&msgs, true)?)?
    } else {
        tokenizer.encode(&args.prompt)?
    };
    let engine = engine::spawn(
        llama,
        tokenizer,
        EngineConfig {
            max_batch_size: 1,
            queue_size: 1,
        },
    )?;
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

fn run_serve(args: ServeArgs) -> Result<()> {
    let (llama, tokenizer) = load(&args.model)?;
    let chat_template = match ChatTemplate::load(&args.model.model) {
        Ok(t) => Some(Arc::new(t)),
        Err(e) => {
            tracing::warn!("chat endpoint disabled: {e:#}");
            None
        }
    };
    let cfg = EngineConfig {
        max_batch_size: args.max_batch_size,
        queue_size: args.queue_size,
    };
    let engine = engine::spawn(llama, tokenizer.clone(), cfg)?;
    let state = AppState {
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
