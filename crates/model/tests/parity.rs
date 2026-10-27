//! Parity against Hugging Face transformers (float32, CPU), using the fixture written by
//! `make parity-ref`. Needs the downloaded model, so it is ignored by default:
//! `make parity` (or `PARITY_DEVICE=cpu|metal|cuda PARITY_DTYPE=f32|f16 cargo test ...`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use model::{ContiguousKvCache, Llama};

const STEPS: usize = 128;

/// Max abs difference allowed on each of the reference top-64 logits and on logsumexp.
/// Calibrated from measured diffs (see `make parity` output) with headroom, then frozen.
fn logit_tolerance(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 5e-4,
        _ => 0.15,
    }
}

/// Reference top-2 margins below this count as ties where reduced precision may legitimately
/// pick the other token. f32 must match exactly; f16 logit noise (measured up to ~0.06 on the
/// Llama 3.2 1B fixture) cannot resolve near-ties, so a flip there ends that prompt's
/// comparison without failing. Any flip with a larger margin is a bug.
fn tie_margin(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 0.0,
        _ => 0.02,
    }
}

struct Case {
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    topk_values: Vec<Vec<f32>>,
    topk_indices: Vec<Vec<u32>>,
    logsumexp: Vec<f32>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_cases() -> Vec<Case> {
    let path = root().join("tests/fixtures/parity/reference.safetensors");
    let t: HashMap<String, Tensor> = candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("{}: {e}; run `make parity-ref`", path.display()));
    let get = |k: String| t.get(&k).unwrap_or_else(|| panic!("missing {k}")).clone();
    (0..)
        .take_while(|i| t.contains_key(&format!("p{i}.prompt_ids")))
        .map(|i| Case {
            prompt_ids: get(format!("p{i}.prompt_ids")).to_vec1().unwrap(),
            greedy_ids: get(format!("p{i}.greedy_ids")).to_vec1().unwrap(),
            topk_values: get(format!("p{i}.topk_values")).to_vec2().unwrap(),
            topk_indices: get(format!("p{i}.topk_indices")).to_vec2().unwrap(),
            logsumexp: get(format!("p{i}.logsumexp")).to_vec1().unwrap(),
        })
        .collect()
}

fn setup() -> (Llama, Device, DType) {
    let (dev_name, device) = device();
    let dtype = match std::env::var("PARITY_DTYPE").as_deref() {
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let dir = root().join("models/llama-3.2-1b-instruct");
    let model = Llama::load(&dir, dtype, &device, 1024).unwrap();
    eprintln!("parity: device={dev_name} dtype={dtype:?}");
    (model, device, dtype)
}

fn device() -> (String, Device) {
    let dev_name = std::env::var("PARITY_DEVICE").unwrap_or_else(|_| {
        let name = if cfg!(feature = "metal") {
            "metal"
        } else if cfg!(feature = "cuda") {
            "cuda"
        } else {
            "cpu"
        };
        name.to_string()
    });
    let device = model::device(&dev_name).unwrap();
    (dev_name, device)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn logsumexp(v: &[f32]) -> f32 {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    m + v.iter().map(|x| ((x - m) as f64).exp()).sum::<f64>().ln() as f32
}

#[test]
#[ignore = "needs models/ (make models) and a GPU"]
fn tokenizer_matches_reference() {
    let tok = tokenizers::Tokenizer::from_file(
        root().join("models/llama-3.2-1b-instruct/tokenizer.json"),
    )
    .unwrap();
    let prompts: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(root().join("tests/fixtures/parity/prompts.json")).unwrap(),
    )
    .unwrap();
    for (prompt, case) in prompts.iter().zip(load_cases()) {
        let ids = tok
            .encode(prompt.as_str(), true)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(ids, case.prompt_ids, "tokenization differs for {prompt:?}");
    }
}

/// Feeds the reference greedy tokens through the KV cache and compares logits at every step.
#[test]
#[ignore = "needs models/ (make models) and a GPU"]
fn teacher_forced_logits_match_reference() {
    let (model, device, dtype) = setup();
    let tol = logit_tolerance(dtype);
    let mut worst = 0f32;
    for (p, case) in load_cases().iter().enumerate() {
        let mut kv = ContiguousKvCache::new(
            model.config(),
            case.prompt_ids.len() + STEPS,
            dtype,
            &device,
        )
        .unwrap();
        let mut logits = model.forward(&case.prompt_ids, &mut kv).unwrap();
        let mut prompt_worst = 0f32;
        for step in 0..STEPS {
            let v = logits.to_vec1::<f32>().unwrap();
            for (&ix, &want) in case.topk_indices[step].iter().zip(&case.topk_values[step]) {
                prompt_worst = prompt_worst.max((v[ix as usize] - want).abs());
            }
            prompt_worst = prompt_worst.max((logsumexp(&v) - case.logsumexp[step]).abs());
            assert!(
                prompt_worst <= tol,
                "prompt {p} step {step}: logit diff {prompt_worst} > tolerance {tol}"
            );
            if step + 1 < STEPS {
                logits = model.forward(&[case.greedy_ids[step]], &mut kv).unwrap();
            }
        }
        eprintln!("prompt {p:>2}: max |logit diff| {prompt_worst:.2e}");
        worst = worst.max(prompt_worst);
    }
    eprintln!("teacher-forced: worst max |logit diff| {worst:.2e} (tolerance {tol:.1e})");
}

/// Free-running greedy decoding must match the reference token for token.
#[test]
#[ignore = "needs models/ (make models) and a GPU"]
fn greedy_tokens_match_reference() {
    let (model, device, dtype) = setup();
    let mut failures = Vec::new();
    let mut ties = Vec::new();
    for (p, case) in load_cases().iter().enumerate() {
        let mut kv = ContiguousKvCache::new(
            model.config(),
            case.prompt_ids.len() + STEPS,
            dtype,
            &device,
        )
        .unwrap();
        let mut logits = model.forward(&case.prompt_ids, &mut kv).unwrap();
        for step in 0..STEPS {
            let got = argmax(&logits.to_vec1::<f32>().unwrap());
            let want = case.greedy_ids[step];
            if got != want {
                // A near-tie in the reference distinguishes numerical noise from a real bug.
                let margin = case.topk_values[step][0] - case.topk_values[step][1];
                let msg = format!(
                    "prompt {p} step {step}: got {got} want {want} (reference top-2 margin {margin:.4})"
                );
                if margin < tie_margin(dtype) {
                    ties.push(msg);
                } else {
                    failures.push(msg);
                }
                break;
            }
            logits = model.forward(&[got], &mut kv).unwrap();
        }
    }
    let n = load_cases().len();
    let exact = n - failures.len() - ties.len();
    eprintln!(
        "greedy: {exact}/{n} prompts match for {STEPS} tokens, {} near-tie flips",
        ties.len()
    );
    for t in &ties {
        eprintln!("  near-tie {t}");
    }
    assert!(
        failures.is_empty(),
        "greedy divergence:\n{}",
        failures.join("\n")
    );
}

/// Batching invariance on the paged path: 32 sequences (the fixture prompts, cycled) prefilled
/// together in one step and decoded together for 128 steps through the paged KV pool must
/// each produce their reference greedy tokens, exactly as when run alone.
#[test]
#[ignore = "needs models/ (make models) and a GPU"]
fn paged_batch_of_32_matches_reference() {
    use model::{AttentionImpl, PagedBatch, PagedKvCache, PagedSeq};
    const N: usize = 32;
    const BS: usize = 16;
    let (model, device, dtype) = setup();
    let cases = load_cases();
    let batch: Vec<&Case> = (0..N).map(|i| &cases[i % cases.len()]).collect();
    let longest = batch.iter().map(|c| c.prompt_ids.len()).max().unwrap() + STEPS;
    let per_seq = longest.div_ceil(BS);
    let mut cache = PagedKvCache::new(model.config(), N * per_seq, BS, dtype, &device).unwrap();
    let attention = match std::env::var("PARITY_ATTENTION").as_deref() {
        Ok("v1") => AttentionImpl::Gather,
        _ => AttentionImpl::Kernel,
    };
    let attention = cache.set_attention(attention);
    // Interleave physical blocks across sequences so no table is contiguous.
    let tables: Vec<Vec<u32>> = (0..N)
        .map(|i| (0..per_seq).map(|j| (j * N + (N - 1 - i)) as u32).collect())
        .collect();

    let mut seqs: Vec<Vec<u32>> = batch.iter().map(|c| c.prompt_ids.clone()).collect();
    let mut computed = vec![0usize; N];
    let mut diverged: Vec<Option<String>> = vec![None; N];
    for step in 0..STEPS {
        let toks: Vec<&[u32]> = (0..N).map(|i| &seqs[i][computed[i]..]).collect();
        let paged: Vec<PagedSeq> = (0..N)
            .map(|i| PagedSeq {
                block_table: &tables[i],
                start_pos: computed[i],
                len: seqs[i].len() - computed[i],
            })
            .collect();
        let logits = model
            .forward_batch(&toks, &mut PagedBatch::new(&cache, &paged).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for i in 0..N {
            computed[i] = seqs[i].len();
            let got = argmax(&logits[i]);
            let want = batch[i].greedy_ids[step];
            if got != want && diverged[i].is_none() {
                let margin = batch[i].topk_values[step][0] - batch[i].topk_values[step][1];
                diverged[i] = Some(format!(
                    "seq {i} step {step}: got {got} want {want} (margin {margin:.4})"
                ));
                assert!(
                    margin < tie_margin(dtype),
                    "seq {i} (prompt {}) step {step}: got {got} want {want}, reference margin {margin:.4}",
                    i % cases.len()
                );
            }
            seqs[i].push(got);
        }
    }
    let flips: Vec<String> = diverged.into_iter().flatten().collect();
    eprintln!(
        "paged batch of {N} ({attention:?}): {}/{N} exact for {STEPS} tokens, {} near-tie flips",
        N - flips.len(),
        flips.len()
    );
    for f in &flips {
        eprintln!("  near-tie {f}");
    }
}

/// GGUF checkpoints against the f32 HF reference. Quantization changes the logits, so this
/// measures rather than demands equality: top-1 agreement under teacher forcing (the model is
/// fed the reference's tokens, so one disagreement doesn't cascade) and the logit error.
#[test]
#[ignore = "needs models/ (make models) and a GPU"]
fn gguf_tracks_reference() {
    use candle_core::quantized::gguf_file;
    let (_, device) = device();
    let hf = model::LlamaConfig::from_file(root().join("models/llama-3.2-1b-instruct/config.json"))
        .unwrap();
    let cases = load_cases();
    for (file, min_agree) in [
        ("Llama-3.2-1B-Instruct-Q8_0.gguf", 0.98),
        ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", 0.92),
    ] {
        let path = root().join("models").join(file);
        // RoPE from rope_freqs must reproduce the HF llama3-scaled frequencies.
        let mut f = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut f).unwrap();
        let freqs = content
            .tensor(&mut f, "rope_freqs.weight", &Device::Cpu)
            .unwrap();
        let freqs = freqs
            .dequantize(&Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cfg = model::gguf::config(&content).unwrap();
        let from_gguf = model::gguf::inv_freqs(&cfg, Some(&freqs));
        for (i, (a, b)) in from_gguf
            .iter()
            .zip(model::rope::inv_freqs(&hf))
            .enumerate()
        {
            assert!(
                ((a - b) / b).abs() < 1e-5,
                "{file}: inv_freq[{i}] {a} vs HF {b}"
            );
        }

        let model = model::load(&path, DType::F32, &device, 1024).unwrap();
        let (mut agree, mut total, mut worst, mut sum_err) = (0usize, 0usize, 0f32, 0f64);
        for case in &cases {
            let mut kv = ContiguousKvCache::new(
                model.config(),
                case.prompt_ids.len() + STEPS,
                model.dtype(),
                &device,
            )
            .unwrap();
            let mut logits = model.forward(&case.prompt_ids, &mut kv).unwrap();
            for step in 0..STEPS {
                let v = logits.to_vec1::<f32>().unwrap();
                agree += usize::from(argmax(&v) == case.topk_indices[step][0]);
                total += 1;
                let err =
                    (v[case.topk_indices[step][0] as usize] - case.topk_values[step][0]).abs();
                worst = worst.max(err);
                sum_err += err as f64;
                if step + 1 < STEPS {
                    logits = model.forward(&[case.greedy_ids[step]], &mut kv).unwrap();
                }
            }
        }
        let rate = agree as f64 / total as f64;
        eprintln!(
            "{file}: top-1 agreement {:.2}% over {total} steps, top logit error mean {:.3} max {:.3}",
            rate * 100.0,
            sum_err / total as f64,
            worst
        );
        assert!(
            rate >= min_agree,
            "{file}: top-1 agreement {rate:.4} below {min_agree}"
        );
    }
}
