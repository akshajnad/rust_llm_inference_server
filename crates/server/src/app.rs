//! HTTP routes. Handlers validate and tokenize on the tokio side, hand the request to the
//! engine thread, and relay its events back as a JSON body or an SSE stream. Dropping the
//! response (client disconnect) drops the event receiver, which cancels the request.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use engine::{
    ChatTemplate, EngineHandle, Event, Finish, Request, SamplingParams, SubmitError, Tokenizer,
};
use futures_util::stream::{self, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::api::{ApiError, COMPLETIONS_DEFAULT_MAX_TOKENS, ChatRequest, CompletionRequest, Usage};

#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<Arc<ChatTemplate>>,
    pub model_name: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn health(State(st): State<AppState>) -> Response {
    let body = json!({
        "status": if st.engine.is_alive() { "ok" } else { "engine_stopped" },
        "running": st.engine.running(),
        "waiting": st.engine.stats().waiting.load(Ordering::Relaxed),
        "queued": st.engine.queue_depth(),
        "preemptions": st.engine.stats().preemptions.load(Ordering::Relaxed),
        "kv_cache_utilization": st.engine.kv_utilization(),
    });
    let status = if st.engine.is_alive() {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body)).into_response()
}

async fn models(State(st): State<AppState>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": st.model_name, "object": "model", "owned_by": "rlis"}]}),
    )
}

/// Which endpoint a response is for; decides the object names and choice shape.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Completion,
    Chat,
}

async fn completions(
    State(st): State<AppState>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body.map_err(|e| ApiError::bad_request(e.body_text()))?;
    let prompt = tokenize(&st, req.prompt_text()?.to_string(), true).await?;
    let params = req
        .sampling
        .to_params(req.sampling.max_tokens, COMPLETIONS_DEFAULT_MAX_TOKENS)?;
    run(
        st,
        Kind::Completion,
        prompt,
        params,
        req.sampling.stream,
        req.sampling.include_usage(),
    )
    .await
}

async fn chat_completions(
    State(st): State<AppState>,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body.map_err(|e| ApiError::bad_request(e.body_text()))?;
    let template = st.chat_template.as_ref().ok_or_else(|| {
        ApiError::bad_request("this model has no chat template; use /v1/completions")
    })?;
    let text = template
        .render(&req.messages()?, true)
        .map_err(|e| ApiError::invalid(format!("chat template: {e:#}"), "messages"))?;
    let prompt = tokenize(&st, text, false).await?;
    let default_max = st.engine.max_seq_len().saturating_sub(prompt.len()).max(1);
    let params = req.sampling.to_params(req.max_tokens(), default_max)?;
    run(
        st,
        Kind::Chat,
        prompt,
        params,
        req.sampling.stream,
        req.sampling.include_usage(),
    )
    .await
}

/// Tokenizes off the async workers: a long prompt takes milliseconds of CPU.
async fn tokenize(st: &AppState, text: String, add_bos: bool) -> Result<Vec<u32>, ApiError> {
    let tokenizer = st.tokenizer.clone();
    tokio::task::spawn_blocking(move || {
        if add_bos {
            tokenizer.encode(&text)
        } else {
            tokenizer.encode_raw(&text)
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(|e| ApiError::internal(e.to_string()))
}

async fn run(
    st: AppState,
    kind: Kind,
    prompt: Vec<u32>,
    params: SamplingParams,
    stream: bool,
    include_usage: bool,
) -> Result<Response, ApiError> {
    let limit = st.engine.max_seq_len();
    if prompt.len() >= limit {
        let param = if kind == Kind::Chat {
            "messages"
        } else {
            "prompt"
        };
        return Err(ApiError::invalid(
            format!(
                "the prompt is {} tokens; this model's maximum context length is {limit}",
                prompt.len()
            ),
            param,
        ));
    }
    let requested = prompt.len() + params.max_tokens;
    if requested > limit {
        return Err(ApiError::invalid(
            format!(
                "this model's maximum context length is {limit} tokens, but {requested} were requested \
                 ({} in the prompt, {} for the completion)",
                prompt.len(),
                params.max_tokens
            ),
            "max_tokens",
        ));
    }
    let (req, rx) = Request::new(prompt, params);
    st.engine.submit(req).map_err(|e| match e {
        SubmitError::QueueFull => ApiError::overloaded(),
        SubmitError::Stopped => ApiError::internal("engine has stopped"),
    })?;
    let meta = Meta {
        id: format!(
            "{}-{:016x}",
            if kind == Kind::Chat {
                "chatcmpl"
            } else {
                "cmpl"
            },
            engine::sampler::random_u64()
        ),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        model: st.model_name.clone(),
        kind,
    };
    if stream {
        Ok(Sse::new(sse_stream(meta, rx, include_usage))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        collect(meta, rx).await.map(|v| Json(v).into_response())
    }
}

struct Meta {
    id: String,
    created: u64,
    model: String,
    kind: Kind,
}

impl Meta {
    fn object(&self, chunk: bool) -> &'static str {
        match (self.kind, chunk) {
            (Kind::Completion, _) => "text_completion",
            (Kind::Chat, false) => "chat.completion",
            (Kind::Chat, true) => "chat.completion.chunk",
        }
    }

    fn envelope(&self, chunk: bool, choices: Value) -> Value {
        json!({"id": self.id, "object": self.object(chunk), "created": self.created, "model": self.model, "choices": choices})
    }

    /// One streamed choice carrying `text` (and `finish_reason` on the last chunk).
    fn chunk(&self, text: Option<&str>, finish: Option<&str>) -> Value {
        let choice = match self.kind {
            Kind::Completion => {
                json!({"index": 0, "text": text.unwrap_or(""), "logprobs": null, "finish_reason": finish})
            }
            Kind::Chat => {
                let delta = text.map_or(json!({}), |t| json!({"content": t}));
                json!({"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish})
            }
        };
        self.envelope(true, json!([choice]))
    }
}

async fn collect(meta: Meta, mut rx: UnboundedReceiver<Event>) -> Result<Value, ApiError> {
    let mut text = String::new();
    loop {
        match rx.recv().await {
            Some(Event::Text(t)) => text.push_str(&t),
            Some(Event::Finished(f)) => return Ok(full_response(&meta, &text, &f)),
            Some(Event::Error(e)) => return Err(ApiError::internal(e)),
            None => return Err(ApiError::internal("engine dropped the request")),
        }
    }
}

fn full_response(meta: &Meta, text: &str, f: &Finish) -> Value {
    let reason = f.reason.as_openai();
    let choice = match meta.kind {
        Kind::Completion => {
            json!({"index": 0, "text": text, "logprobs": null, "finish_reason": reason})
        }
        Kind::Chat => json!({
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": null,
            "finish_reason": reason,
        }),
    };
    let mut v = meta.envelope(false, json!([choice]));
    v["usage"] = json!(Usage::new(f.prompt_tokens, f.completion_tokens));
    v
}

/// Turns engine events into SSE `data:` lines: an initial role delta for chat, one chunk per
/// text event, a final chunk with `finish_reason`, optional usage, then `[DONE]`.
fn sse_stream(
    meta: Meta,
    rx: UnboundedReceiver<Event>,
    include_usage: bool,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let data = |v: &Value| SseEvent::default().data(v.to_string());
    let opening = match meta.kind {
        Kind::Chat => {
            let role = json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}]);
            vec![data(&meta.envelope(true, role))]
        }
        Kind::Completion => Vec::new(),
    };
    let body = stream::unfold(Some((meta, rx)), move |state| async move {
        let (meta, mut rx) = state?;
        let events: Vec<SseEvent> = match rx.recv().await {
            Some(Event::Text(t)) => {
                let ev = data(&meta.chunk(Some(&t), None));
                return Some((vec![ev], Some((meta, rx))));
            }
            Some(Event::Finished(f)) => {
                let mut out = vec![data(&meta.chunk(None, Some(f.reason.as_openai())))];
                if include_usage {
                    let mut u = meta.envelope(true, json!([]));
                    u["usage"] = json!(Usage::new(f.prompt_tokens, f.completion_tokens));
                    out.push(data(&u));
                }
                out
            }
            Some(Event::Error(e)) => vec![data(&ApiError::internal(e).body())],
            None => vec![data(
                &ApiError::internal("engine dropped the request").body(),
            )],
        };
        let mut events = events;
        events.push(SseEvent::default().data("[DONE]"));
        Some((events, None))
    });
    stream::iter(opening)
        .chain(body.flat_map(stream::iter))
        .map(Ok)
}
