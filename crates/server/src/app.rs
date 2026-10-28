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
    /// Renders the Prometheus exposition for `/metrics`; `None` disables the endpoint.
    pub metrics: Option<metrics_exporter_prometheus::PrometheusHandle>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/metrics", get(prometheus))
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
        "kv_blocks_used": st.engine.stats().kv_blocks_used.load(Ordering::Relaxed),
        "kv_blocks_total": st.engine.stats().kv_blocks_total.load(Ordering::Relaxed),
        "kv_block_size": st.engine.stats().kv_block_size.load(Ordering::Relaxed),
        "prefix_cache_hit_tokens": st.engine.stats().cached_tokens.load(Ordering::Relaxed),
        "admitted_tokens": st.engine.stats().admitted_tokens.load(Ordering::Relaxed),
    });
    let status = if st.engine.is_alive() {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body)).into_response()
}

/// Queue depth and memory are sampled at scrape time; engine metrics are recorded as they happen.
async fn prometheus(State(st): State<AppState>) -> Response {
    let Some(handle) = &st.metrics else {
        return (axum::http::StatusCode::NOT_FOUND, "metrics disabled").into_response();
    };
    metrics::gauge!("rlis_queue_depth").set(st.engine.queue_depth() as f64);
    if let Some(f) = crate::memory::footprint() {
        metrics::gauge!("rlis_process_memory_footprint_bytes").set(f.current_bytes as f64);
        metrics::gauge!("rlis_process_memory_footprint_peak_bytes").set(f.peak_bytes as f64);
    }
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        handle.render(),
    )
        .into_response()
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

/// `params` holds one entry per completion (`n`); they differ only in seed.
async fn run(
    st: AppState,
    kind: Kind,
    prompt: Vec<u32>,
    params: Vec<SamplingParams>,
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
    let max_tokens = params[0].max_tokens;
    let requested = prompt.len() + max_tokens;
    if requested > limit {
        return Err(ApiError::invalid(
            format!(
                "this model's maximum context length is {limit} tokens, but {requested} were requested \
                 ({} in the prompt, {} for the completion)",
                prompt.len(),
                max_tokens
            ),
            "max_tokens",
        ));
    }
    let prompt_tokens = prompt.len();
    let (req, rxs) = Request::with_forks(prompt, params);
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
        Ok(
            Sse::new(sse_stream(meta, rxs, prompt_tokens, include_usage))
                .keep_alive(KeepAlive::default())
                .into_response(),
        )
    } else {
        collect(meta, rxs).await.map(|v| Json(v).into_response())
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

    /// One streamed chunk of choice `index` carrying `text` (and `finish_reason` on its last).
    fn chunk(&self, index: usize, text: Option<&str>, finish: Option<&str>) -> Value {
        let choice = match self.kind {
            Kind::Completion => {
                json!({"index": index, "text": text.unwrap_or(""), "logprobs": null, "finish_reason": finish})
            }
            Kind::Chat => {
                let delta = text.map_or(json!({}), |t| json!({"content": t}));
                json!({"index": index, "delta": delta, "logprobs": null, "finish_reason": finish})
            }
        };
        self.envelope(true, json!([choice]))
    }
}

async fn collect(meta: Meta, rxs: Vec<UnboundedReceiver<Event>>) -> Result<Value, ApiError> {
    let mut choices = Vec::new();
    let mut usage = Usage::new(0, 0);
    for (index, mut rx) in rxs.into_iter().enumerate() {
        let mut text = String::new();
        let f = loop {
            match rx.recv().await {
                Some(Event::Text(t)) => text.push_str(&t),
                Some(Event::Finished(f)) => break f,
                Some(Event::Error(e)) => return Err(ApiError::internal(e)),
                None => return Err(ApiError::internal("engine dropped the request")),
            }
        };
        choices.push(full_choice(&meta, index, &text, &f));
        usage = Usage::new(
            f.prompt_tokens,
            usage.completion_tokens + f.completion_tokens,
        );
    }
    let mut v = meta.envelope(false, json!(choices));
    v["usage"] = json!(usage);
    Ok(v)
}

fn full_choice(meta: &Meta, index: usize, text: &str, f: &Finish) -> Value {
    let reason = f.reason.as_openai();
    match meta.kind {
        Kind::Completion => {
            json!({"index": index, "text": text, "logprobs": null, "finish_reason": reason})
        }
        Kind::Chat => json!({
            "index": index,
            "message": {"role": "assistant", "content": text},
            "logprobs": null,
            "finish_reason": reason,
        }),
    }
}

/// Turns engine events into SSE `data:` lines: an initial role delta per choice for chat, one
/// chunk per text event, a final chunk per choice with `finish_reason`, then (once every
/// choice is done) optional usage and `[DONE]`. Choices interleave as their tokens arrive.
fn sse_stream(
    meta: Meta,
    rxs: Vec<UnboundedReceiver<Event>>,
    prompt_tokens: usize,
    include_usage: bool,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let data = |v: &Value| SseEvent::default().data(v.to_string());
    let opening: Vec<SseEvent> = match meta.kind {
        Kind::Chat => (0..rxs.len())
            .map(|i| {
                let role = json!([{"index": i, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}]);
                data(&meta.envelope(true, role))
            })
            .collect(),
        Kind::Completion => Vec::new(),
    };
    let remaining = rxs.len();
    let events = stream::select_all(rxs.into_iter().enumerate().map(|(i, rx)| {
        stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|e| (e, rx)) })
            .map(move |e| (i, e))
            .boxed()
    }));
    let body = stream::unfold(
        Some((meta, events, remaining, 0usize)),
        move |state| async move {
            let (meta, mut events, mut remaining, mut completion_tokens) = state?;
            let mut out = match events.next().await {
                Some((i, Event::Text(t))) => {
                    let ev = data(&meta.chunk(i, Some(&t), None));
                    return Some((vec![ev], Some((meta, events, remaining, completion_tokens))));
                }
                Some((i, Event::Finished(f))) => {
                    let ev = data(&meta.chunk(i, None, Some(f.reason.as_openai())));
                    remaining -= 1;
                    completion_tokens += f.completion_tokens;
                    if remaining > 0 {
                        return Some((
                            vec![ev],
                            Some((meta, events, remaining, completion_tokens)),
                        ));
                    }
                    let mut out = vec![ev];
                    if include_usage {
                        let mut u = meta.envelope(true, json!([]));
                        u["usage"] = json!(Usage::new(prompt_tokens, completion_tokens));
                        out.push(data(&u));
                    }
                    out
                }
                Some((_, Event::Error(e))) => vec![data(&ApiError::internal(e).body())],
                None => vec![data(
                    &ApiError::internal("engine dropped the request").body(),
                )],
            };
            out.push(SseEvent::default().data("[DONE]"));
            Some((out, None))
        },
    );
    stream::iter(opening)
        .chain(body.flat_map(stream::iter))
        .map(Ok)
}
