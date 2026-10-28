//! Chat templates rendered by minijinja must match transformers' `apply_chat_template`
//! exactly: Llama 3.2 (fixture from `make chat-ref`) and Qwen2.5-Coder's ChatML (`make
//! chat-ref-qwen`). Needs `models/`.

use std::path::Path;

use engine::{ChatMessage, ChatTemplate, Tokenizer};
use serde_json::Value;

#[test]
#[ignore = "needs models/ (make models)"]
fn llama32_template_matches_transformers() {
    check("models/llama-3.2-1b-instruct", "llama32_reference.json");
}

#[test]
#[ignore = "needs models/ (make models-qwen)"]
fn qwen25_coder_template_matches_transformers() {
    check(
        "models/qwen2.5-coder-1.5b-instruct",
        "qwen25_coder_reference.json",
    );
}

fn check(model_dir: &str, fixture: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join(model_dir);
    let template = ChatTemplate::load(&model_dir).unwrap();
    let tokenizer = Tokenizer::load(&model_dir).unwrap();
    let fixture = std::fs::read_to_string(root.join("tests/fixtures/chat").join(fixture)).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&fixture).unwrap();
    for case in cases {
        let messages: Vec<ChatMessage> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| ChatMessage {
                role: m["role"].as_str().unwrap().into(),
                content: m["content"].as_str().unwrap().into(),
            })
            .collect();
        let text = template
            .render_with(&messages, true, Some("26 Jul 2024"))
            .unwrap();
        assert_eq!(text, case["text"].as_str().unwrap());
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tokenizer.encode_raw(&text).unwrap(), ids);
    }
}
