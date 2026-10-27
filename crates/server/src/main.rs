use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::DType;
use clap::{Parser, Subcommand, ValueEnum};
use engine::{Sampler, Tokenizer};
use model::Llama;

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
}

#[derive(clap::Args)]
struct GenerateArgs {
    #[arg(long, default_value = "models/llama-3.2-1b-instruct")]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, value_enum, default_value_t = Dtype::F16)]
    dtype: Dtype,
    #[arg(long, default_value = default_device())]
    device: String,
    /// Positions covered by the RoPE tables; bounds prompt plus output length.
    #[arg(long, default_value_t = 8192)]
    max_seq_len: usize,
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
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().command {
        Command::Generate(args) => run_generate(args),
    }
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let device = model::device(&args.device)?;
    let load_start = std::time::Instant::now();
    let llama = Llama::load(&args.model, args.dtype.into(), &device, args.max_seq_len)?;
    let tokenizer = Tokenizer::load(&args.model)?;
    eprintln!(
        "loaded {} in {:.2?}",
        args.model.display(),
        load_start.elapsed()
    );

    let prompt = tokenizer.encode(&args.prompt)?;
    let mut stdout = std::io::stdout().lock();
    let generation = engine::generate(
        &llama,
        &tokenizer,
        &mut Sampler::greedy(),
        &prompt,
        args.max_tokens,
        |text| {
            let _ = stdout.write_all(text.as_bytes());
            let _ = stdout.flush();
        },
    )?;
    eprintln!(
        "\n\nprompt {} tok | output {} tok ({:?}) | ttft {:.1?} | decode {:.1} tok/s",
        generation.prompt_tokens,
        generation.tokens.len(),
        generation.finish,
        generation.ttft,
        generation.decode_tokens_per_sec(),
    );
    Ok(())
}
