//! Arbitrary bytes through request parsing and validation must never panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use server::api::{ChatRequest, CompletionRequest};

fuzz_target!(|data: &[u8]| {
    if let Ok(req) = serde_json::from_slice::<CompletionRequest>(data) {
        let _ = req.prompt_text();
        let _ = req.sampling.to_params(req.sampling.max_tokens, 16);
    }
    if let Ok(req) = serde_json::from_slice::<ChatRequest>(data) {
        let _ = req.messages();
        let _ = req.sampling.to_params(req.max_tokens(), 1024);
    }
});
