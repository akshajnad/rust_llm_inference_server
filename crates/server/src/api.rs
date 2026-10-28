//! OpenAI-compatible request and response shapes, and request validation.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use engine::{ChatMessage, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    /// Accepted for client compatibility; the server serves one model.
    #[allow(dead_code)]
    pub model: Option<String>,
    pub prompt: Prompt,
    #[serde(flatten)]
    pub sampling: SamplingFields,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Accepted for client compatibility; the server serves one model.
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<RawMessage>,
    /// Newer OpenAI name for `max_tokens` in chat.
    pub max_completion_tokens: Option<i64>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Text(String),
    Batch(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct RawMessage {
    pub role: String,
    /// A string, or an array of `{"type": "text", "text": ...}` parts.
    #[serde(default)]
    pub content: Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

/// Fields shared by both endpoints.
#[derive(Debug, Deserialize)]
pub struct SamplingFields {
    pub max_tokens: Option<i64>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i64>,
    pub stop: Option<Stop>,
    pub seed: Option<u64>,
    pub repetition_penalty: Option<f32>,
    /// vLLM/llama.cpp extension: generate exactly `max_tokens`, ignoring EOS.
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub n: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// OpenAI default for `/v1/completions`; chat defaults to the rest of the context window.
pub const COMPLETIONS_DEFAULT_MAX_TOKENS: usize = 16;
const MAX_STOP_SEQUENCES: usize = 4;

impl SamplingFields {
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage)
    }

    /// Checks ranges and builds engine params. `max_tokens` here is the already-resolved value
    /// (request field or endpoint default); the context-length check happens after tokenizing.
    pub fn to_params(
        &self,
        max_tokens: Option<i64>,
        default_max: usize,
    ) -> Result<SamplingParams, ApiError> {
        if let Some(n) = self.n
            && n != 1
        {
            return Err(ApiError::invalid("only n=1 is supported", "n"));
        }
        let temperature = self.temperature.unwrap_or(1.0);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::invalid(
                "temperature must be between 0 and 2",
                "temperature",
            ));
        }
        let top_p = self.top_p.unwrap_or(1.0);
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(ApiError::invalid("top_p must be in (0, 1]", "top_p"));
        }
        let top_k = match self.top_k.unwrap_or(0) {
            -1 | 0 => 0,
            k if k > 0 => k as usize,
            _ => {
                return Err(ApiError::invalid(
                    "top_k must be -1, 0, or positive",
                    "top_k",
                ));
            }
        };
        let repetition_penalty = self.repetition_penalty.unwrap_or(1.0);
        if !(repetition_penalty > 0.0 && repetition_penalty <= 2.0) {
            return Err(ApiError::invalid(
                "repetition_penalty must be in (0, 2]",
                "repetition_penalty",
            ));
        }
        let max_tokens = match max_tokens {
            None => default_max,
            Some(m) if m >= 1 => m as usize,
            Some(_) => {
                return Err(ApiError::invalid(
                    "max_tokens must be at least 1",
                    "max_tokens",
                ));
            }
        };
        let stop = match &self.stop {
            None => Vec::new(),
            Some(Stop::One(s)) => vec![s.clone()],
            Some(Stop::Many(v)) => v.clone(),
        };
        if stop.len() > MAX_STOP_SEQUENCES {
            return Err(ApiError::invalid(
                "at most 4 stop sequences are allowed",
                "stop",
            ));
        }
        if stop.iter().any(String::is_empty) {
            return Err(ApiError::invalid(
                "stop sequences must be non-empty",
                "stop",
            ));
        }
        Ok(SamplingParams {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed: self.seed,
            max_tokens,
            stop,
            ignore_eos: self.ignore_eos,
        })
    }
}

impl CompletionRequest {
    pub fn prompt_text(&self) -> Result<&str, ApiError> {
        match &self.prompt {
            Prompt::Text(s) => Ok(s),
            Prompt::Batch(v) if v.len() == 1 => Ok(&v[0]),
            Prompt::Batch(_) => Err(ApiError::invalid(
                "batched prompts are not supported; send one prompt",
                "prompt",
            )),
        }
    }
}

impl ChatRequest {
    pub fn messages(&self) -> Result<Vec<ChatMessage>, ApiError> {
        if self.messages.is_empty() {
            return Err(ApiError::invalid("messages must not be empty", "messages"));
        }
        self.messages
            .iter()
            .map(|m| {
                let content = match &m.content {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    Value::Array(parts) => parts
                        .iter()
                        .map(|p| match (p["type"].as_str(), p["text"].as_str()) {
                            (Some("text"), Some(t)) => Ok(t),
                            _ => Err(ApiError::invalid(
                                "only text content parts are supported",
                                "messages",
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .concat(),
                    _ => {
                        return Err(ApiError::invalid(
                            "message content must be a string or text parts",
                            "messages",
                        ));
                    }
                };
                Ok(ChatMessage {
                    role: m.role.clone(),
                    content,
                })
            })
            .collect()
    }

    pub fn max_tokens(&self) -> Option<i64> {
        self.max_completion_tokens.or(self.sampling.max_tokens)
    }
}

#[derive(Debug, Serialize, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// An error in OpenAI's `{"error": {...}}` shape.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub param: Option<&'static str>,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>, param: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            param: Some(param),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            param: None,
        }
    }

    pub fn overloaded() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "the server is at capacity; retry shortly".into(),
            kind: "server_error",
            param: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            param: None,
        }
    }

    pub fn body(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.kind, "param": self.param, "code": null}})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut resp = (self.status, axum::Json(self.body())).into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        resp
    }
}
