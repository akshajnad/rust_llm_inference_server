//! A tiny random-weight Llama and a byte-level tokenizer, so the engine and server can be
//! exercised end to end without downloading a model.

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use model::{Llama, LlamaConfig};

use crate::tokenizer::Tokenizer;

pub const BOS: u32 = 256;
pub const EOS: u32 = 257;

/// Byte-level BPE with one token per byte (ids 0..=255), `<s>` = 256 and `</s>` = 257. The
/// returned tokenizer has no EOS ids, so generation always runs to `max_tokens`.
pub fn byte_tokenizer() -> Tokenizer {
    let vocab: serde_json::Map<String, serde_json::Value> = byte_to_unicode()
        .iter()
        .enumerate()
        .map(|(b, c)| (c.to_string(), b.into()))
        .collect();
    let special = |id: u32, content: &str| {
        serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true})
    };
    let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false,
        "trim_offsets": true, "use_regex": true});
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [special(BOS, "<s>"), special(EOS, "</s>")],
        "normalizer": null,
        "pre_tokenizer": byte_level,
        "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [BOS], "tokens": ["<s>"]}}},
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
            "vocab": vocab, "merges": []},
    });
    let inner: tokenizers::Tokenizer = json.to_string().parse().expect("valid tokenizer json");
    Tokenizer::new(inner, Vec::new())
}

/// `byte_tokenizer` with the given ids treated as end-of-sequence.
pub fn byte_tokenizer_with_eos(eos: Vec<u32>) -> Tokenizer {
    Tokenizer::new(byte_tokenizer().into_inner(), eos)
}

/// GPT-2's reversible byte-to-printable-char mapping used by byte-level BPE.
fn byte_to_unicode() -> Vec<char> {
    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut next = 256;
    (0..256u32)
        .map(|b| {
            if printable(b) {
                char::from_u32(b).unwrap()
            } else {
                next += 1;
                char::from_u32(next - 1).unwrap()
            }
        })
        .collect()
}

/// Random-weight Llama on CPU sized for the byte tokenizer.
pub fn tiny_model(max_seq_len: usize) -> Llama {
    let cfg = LlamaConfig {
        vocab_size: 258,
        max_position_embeddings: max_seq_len,
        ..LlamaConfig::tiny()
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    Llama::new(cfg, vb, max_seq_len).expect("tiny model")
}
