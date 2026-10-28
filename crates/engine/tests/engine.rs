//! End-to-end engine tests on a tiny random model (no weights needed).

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::testing::{byte_tokenizer, tiny_model};
use engine::{EngineConfig, EngineHandle, Event, FinishReason, Policy, Request, SamplingParams};
use tokio::sync::mpsc::UnboundedReceiver;

fn start_with(cfg: EngineConfig) -> (EngineHandle, Arc<engine::Tokenizer>) {
    let tok = Arc::new(byte_tokenizer());
    (
        engine::spawn(tiny_model(256), tok.clone(), cfg).unwrap(),
        tok,
    )
}

/// The tiny model's KV is 512 bytes per token, so `kv_tokens` sizes the pool directly.
fn config(max_num_seqs: usize, kv_tokens: usize) -> EngineConfig {
    EngineConfig {
        max_num_seqs,
        max_batched_tokens: 64,
        queue_size: 64,
        kv_cache_bytes: kv_tokens * 512,
        block_size: 4,
        policy: Policy::Continuous,
        attention: engine::AttentionImpl::Kernel,
        prefix_caching: true,
    }
}

fn start(max_num_seqs: usize) -> (EngineHandle, Arc<engine::Tokenizer>) {
    start_with(config(max_num_seqs, 4096))
}

fn collect(mut rx: UnboundedReceiver<Event>) -> (String, engine::Finish) {
    let mut text = String::new();
    loop {
        match rx.blocking_recv().expect("engine dropped request") {
            Event::Text(t) => text += &t,
            Event::Finished(f) => return (text, f),
            Event::Error(e) => panic!("engine error: {e}"),
        }
    }
}

fn submit(
    h: &EngineHandle,
    tok: &engine::Tokenizer,
    prompt: &str,
    params: SamplingParams,
) -> UnboundedReceiver<Event> {
    let (req, rx) = Request::new(tok.encode(prompt).unwrap(), params);
    h.submit(req).unwrap();
    rx
}

#[test]
fn greedy_runs_to_max_tokens() {
    let (h, tok) = start(8);
    let (text, fin) = collect(submit(&h, &tok, "hello", SamplingParams::greedy(12)));
    assert_eq!(fin.reason, FinishReason::Length);
    assert_eq!(fin.completion_tokens, 12);
    assert_eq!(fin.prompt_tokens, 6);
    assert!(!text.is_empty());
}

/// Batched requests produce exactly what each produces alone (static batching invariance).
#[test]
fn batched_requests_match_solo_runs() {
    let (h, tok) = start(8);
    let prompts = ["a", "the quick brown fox", "zz", "é😀"];
    let solo: Vec<String> = prompts
        .iter()
        .map(|p| collect(submit(&h, &tok, p, SamplingParams::greedy(10))).0)
        .collect();
    let rxs: Vec<_> = prompts
        .iter()
        .map(|p| submit(&h, &tok, p, SamplingParams::greedy(10)))
        .collect();
    let batched: Vec<String> = rxs.into_iter().map(|rx| collect(rx).0).collect();
    assert_eq!(solo, batched);
}

#[test]
fn seeded_sampling_is_reproducible() {
    let (h, tok) = start(8);
    let params = |seed| SamplingParams {
        temperature: 1.0,
        top_p: 0.95,
        seed: Some(seed),
        ..SamplingParams::greedy(20)
    };
    let a = collect(submit(&h, &tok, "seed", params(1))).0;
    let b = collect(submit(&h, &tok, "seed", params(1))).0;
    let c = collect(submit(&h, &tok, "seed", params(2))).0;
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn stop_string_ends_generation_without_emitting_it() {
    let (h, tok) = start(8);
    let (full, _) = collect(submit(&h, &tok, "stop test", SamplingParams::greedy(30)));
    // Stop on a string the greedy output is known to contain.
    let chars: Vec<char> = full.chars().collect();
    let stop: String = chars[5..7].iter().collect();
    let params = SamplingParams {
        stop: vec![stop.clone()],
        ..SamplingParams::greedy(30)
    };
    let (text, fin) = collect(submit(&h, &tok, "stop test", params));
    assert_eq!(fin.reason, FinishReason::Stop);
    assert_eq!(text, &full[..full.find(&stop).unwrap()]);
}

#[test]
fn dropping_the_receiver_cancels() {
    let (h, tok) = start(8);
    let rx = submit(&h, &tok, "cancel me", SamplingParams::greedy(200));
    let deadline = Instant::now() + Duration::from_secs(5);
    while h.running() == 0 {
        assert!(Instant::now() < deadline, "request never started");
        std::thread::yield_now();
    }
    drop(rx);
    while h.running() != 0 {
        assert!(Instant::now() < deadline, "cancelled request kept running");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn too_long_requests_get_an_error() {
    let (h, tok) = start(8);
    let (req, mut rx) = Request::new(tok.encode("x").unwrap(), SamplingParams::greedy(1000));
    h.submit(req).unwrap();
    assert!(matches!(rx.blocking_recv(), Some(Event::Error(_))));
}

#[test]
fn full_queue_is_refused() {
    let cfg = EngineConfig {
        queue_size: 1,
        ..EngineConfig::default()
    };
    let (h, _rx) = EngineHandle::channel(&cfg, 256);
    let req = || Request::new(vec![1], SamplingParams::greedy(1)).0;
    h.submit(req()).unwrap();
    assert!(matches!(
        h.submit(req()),
        Err(engine::SubmitError::QueueFull)
    ));
    assert_eq!(h.queue_depth(), 1);
}

/// Under a pool too small for everyone, sequences get preempted and recomputed; greedy output
/// must be exactly what each prompt produces alone with plenty of memory.
#[test]
fn preemption_and_recompute_do_not_change_output() {
    let prompts: Vec<String> = (0..12).map(|i| format!("prompt number {i} says")).collect();
    let (roomy, tok) = start(16);
    let want: Vec<String> = prompts
        .iter()
        .map(|p| collect(submit(&roomy, &tok, p, SamplingParams::greedy(40))).0)
        .collect();

    // 160 slots for 12 sequences of ~19 + 40 tokens each.
    let (tight, tok) = start_with(config(16, 160));
    let rxs: Vec<_> = prompts
        .iter()
        .map(|p| submit(&tight, &tok, p, SamplingParams::greedy(40)))
        .collect();
    let got: Vec<String> = rxs.into_iter().map(|rx| collect(rx).0).collect();
    assert_eq!(got, want);
    assert!(
        tight
            .stats()
            .preemptions
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0,
        "pool was meant to be too small"
    );
    // Stats publish after the step that finished the last request; give it a moment.
    let deadline = Instant::now() + Duration::from_secs(5);
    while tight.kv_utilization() != 0.0 {
        assert!(
            Instant::now() < deadline,
            "blocks still in use: {}",
            tight.kv_utilization()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A prompt longer than the token budget is prefilled over several steps.
#[test]
fn chunked_prefill_matches_single_shot() {
    let prompt = "a fairly long prompt that will not fit in one step of sixteen tokens";
    let (one_shot, tok) = start(4);
    let want = collect(submit(&one_shot, &tok, prompt, SamplingParams::greedy(10))).0;
    let (chunked, tok) = start_with(EngineConfig {
        max_batched_tokens: 16,
        ..config(4, 4096)
    });
    assert_eq!(
        collect(submit(&chunked, &tok, prompt, SamplingParams::greedy(10))).0,
        want
    );
}

#[test]
fn static_policy_produces_the_same_text() {
    let (cont, tok) = start(8);
    let (stat, _) = start_with(EngineConfig {
        policy: Policy::Static,
        ..config(8, 4096)
    });
    let prompts = ["x", "yy", "zzz"];
    let a: Vec<_> = prompts
        .iter()
        .map(|p| submit(&cont, &tok, p, SamplingParams::greedy(8)))
        .collect();
    let b: Vec<_> = prompts
        .iter()
        .map(|p| submit(&stat, &tok, p, SamplingParams::greedy(8)))
        .collect();
    let a: Vec<String> = a.into_iter().map(|rx| collect(rx).0).collect();
    let b: Vec<String> = b.into_iter().map(|rx| collect(rx).0).collect();
    assert_eq!(a, b);
}

/// Temperature-only sampling runs on the device with noise keyed by (seed, position): a seeded
/// request produces the same text alone and surrounded by other requests.
#[test]
fn seeded_temperature_sampling_ignores_batch_neighbours() {
    let (h, tok) = start(8);
    let params = |seed| SamplingParams {
        temperature: 0.9,
        seed: Some(seed),
        ..SamplingParams::greedy(24)
    };
    let alone = collect(submit(&h, &tok, "neighbours", params(5))).0;
    let other_seed = collect(submit(&h, &tok, "neighbours", params(6))).0;
    assert_ne!(
        other_seed, alone,
        "a different seed should give different text"
    );
    let rxs: Vec<_> = (0..4)
        .map(|i| match i {
            1 => submit(&h, &tok, "neighbours", params(5)),
            _ => submit(&h, &tok, "noise", params(99 + i)),
        })
        .collect();
    let batched: Vec<String> = rxs.into_iter().map(|rx| collect(rx).0).collect();
    assert_eq!(batched[1], alone);
}

#[test]
fn ignore_eos_generates_exactly_max_tokens() {
    // Every token is EOS, so normally generation stops at once.
    let tok = Arc::new(engine::testing::byte_tokenizer_with_eos((0..258).collect()));
    let h = engine::spawn(tiny_model(256), tok.clone(), config(4, 4096)).unwrap();
    let (_, fin) = collect(submit(&h, &tok, "eos", SamplingParams::greedy(10)));
    assert_eq!((fin.reason, fin.completion_tokens), (FinishReason::Stop, 0));
    let params = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::greedy(10)
    };
    let (_, fin) = collect(submit(&h, &tok, "eos", params));
    assert_eq!(
        (fin.reason, fin.completion_tokens),
        (FinishReason::Length, 10)
    );
}

/// Reused prefix blocks must give exactly the output of computing them fresh: a repeated
/// prompt and prompts sharing a long prefix, with caching on and off.
#[test]
fn prefix_caching_changes_nothing_but_speed() {
    let shared = "a long shared system prompt that several requests start with, ".repeat(2);
    let prompts: Vec<String> = ["first question", "second question", "first question"]
        .iter()
        .map(|q| format!("{shared}{q}"))
        .collect();
    let run = |caching: bool| {
        let (h, tok) = start_with(EngineConfig {
            prefix_caching: caching,
            ..config(4, 4096)
        });
        // One at a time, so later requests can reuse blocks earlier ones computed.
        let texts: Vec<String> = prompts
            .iter()
            .map(|p| collect(submit(&h, &tok, p, SamplingParams::greedy(12))).0)
            .collect();
        (
            texts,
            h.stats()
                .cached_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    };
    let (with, hits) = run(true);
    let (without, no_hits) = run(false);
    assert_eq!(with, without);
    assert_eq!(no_hits, 0);
    let prefix_tokens = byte_tokenizer().encode(&shared).unwrap().len() as u64;
    assert!(
        hits >= prefix_tokens,
        "expected the shared prefix to be reused, got {hits} cached tokens"
    );
}

fn submit_forked(
    h: &EngineHandle,
    tok: &engine::Tokenizer,
    prompt: &str,
    params: Vec<SamplingParams>,
) -> Vec<UnboundedReceiver<Event>> {
    let (req, rxs) = Request::with_forks(tok.encode(prompt).unwrap(), params);
    h.submit(req).unwrap();
    rxs
}

/// Waits for the engine to publish an empty KV pool (stats lag the last step slightly).
fn assert_pool_drains(h: &EngineHandle) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while h.kv_utilization() > 0.0 {
        assert!(Instant::now() < deadline, "KV blocks still in use");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Forked completions share one prefill and copy the prompt's last block on write; each must
/// produce exactly what a separate request with the same parameters does. Prompt lengths
/// cover a last block that is full (no copy needed) and partial (copied by all but one).
#[test]
fn forked_completions_match_separate_requests() {
    let (h, tok) = start(8);
    let params = |seed| SamplingParams {
        temperature: 0.9,
        seed: Some(seed),
        ..SamplingParams::greedy(20)
    };
    // "<s>" plus 11 or 13 bytes: 12 and 14 tokens with 4-token blocks.
    for prompt in ["fork prompt", "a fork prompt"] {
        let solo: Vec<String> = (0..4)
            .map(|i| collect(submit(&h, &tok, prompt, params(i))).0)
            .collect();
        assert_ne!(solo[0], solo[1], "seeds should give different text");
        let forked: Vec<String> = submit_forked(&h, &tok, prompt, (0..4).map(params).collect())
            .into_iter()
            .map(|rx| collect(rx).0)
            .collect();
        assert_eq!(forked, solo, "{prompt:?}");
        let greedy = collect(submit(&h, &tok, prompt, SamplingParams::greedy(20))).0;
        for rx in submit_forked(&h, &tok, prompt, vec![SamplingParams::greedy(20); 3]) {
            assert_eq!(collect(rx).0, greedy);
        }
    }
    assert_pool_drains(&h);
}

/// The parent's client hanging up doesn't cancel its children; everyone hanging up frees all.
#[test]
fn children_outlive_their_parents_client() {
    let (h, tok) = start(8);
    let expected = collect(submit(&h, &tok, "orphans", SamplingParams::greedy(30))).0;
    let mut rxs = submit_forked(&h, &tok, "orphans", vec![SamplingParams::greedy(30); 3]);
    drop(rxs.remove(0));
    for rx in rxs {
        let (text, fin) = collect(rx);
        assert_eq!((text, fin.completion_tokens), (expected.clone(), 30));
    }
    let rxs = submit_forked(&h, &tok, "orphans", vec![SamplingParams::greedy(200); 4]);
    drop(rxs);
    assert_pool_drains(&h);
    // The engine is still healthy afterwards.
    assert_eq!(
        collect(submit(&h, &tok, "orphans", SamplingParams::greedy(30))).0,
        expected
    );
}

#[test]
fn more_forks_than_seats_is_an_error_for_every_receiver() {
    let (h, tok) = start(4);
    for mut rx in submit_forked(&h, &tok, "x", vec![SamplingParams::greedy(4); 5]) {
        match rx.blocking_recv() {
            Some(Event::Error(e)) => assert!(e.contains("max_num_seqs"), "{e}"),
            other => panic!("expected an error, got {other:?}"),
        }
    }
}
