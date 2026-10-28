//! HTTP-level tests against a tiny random model (no weights needed).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use engine::testing::{byte_tokenizer, tiny_model};
use engine::{ChatTemplate, EngineConfig, EngineHandle};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use server::{AppState, router};
use tower::ServiceExt;

const MAX_SEQ_LEN: usize = 128;

fn state_with(engine: EngineHandle) -> AppState {
    let template = "{{ bos_token }}{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
                    {% if add_generation_prompt %}assistant: {% endif %}";
    AppState {
        engine,
        tokenizer: Arc::new(byte_tokenizer()),
        chat_template: Some(Arc::new(
            ChatTemplate::new(template, "<s>", "</s>").unwrap(),
        )),
        model_name: "tiny".into(),
        metrics: Some(prometheus()),
    }
}

/// The recorder is process-global, so every test shares one.
fn prometheus() -> metrics_exporter_prometheus::PrometheusHandle {
    static HANDLE: std::sync::OnceLock<metrics_exporter_prometheus::PrometheusHandle> =
        std::sync::OnceLock::new();
    HANDLE
        .get_or_init(|| server::install_prometheus().unwrap())
        .clone()
}

fn state() -> AppState {
    let tok = Arc::new(byte_tokenizer());
    let engine = engine::spawn(
        tiny_model(MAX_SEQ_LEN),
        tok,
        EngineConfig {
            max_num_seqs: 8,
            queue_size: 8,
            kv_cache_bytes: 4096 * 512,
            block_size: 4,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    state_with(engine)
}

fn post(path: &str, body: Value) -> Request<Body> {
    Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn send(st: &AppState, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, String) {
    let resp = router(st.clone()).oneshot(req).await.unwrap();
    let (parts, body) = resp.into_parts();
    let bytes = body.collect().await.unwrap().to_bytes();
    (
        parts.status,
        parts.headers,
        String::from_utf8(bytes.to_vec()).unwrap(),
    )
}

/// The JSON payloads of an SSE body, with the final `[DONE]` checked and removed.
fn sse_payloads(body: &str) -> Vec<Value> {
    let data: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(
        data.last(),
        Some(&"[DONE]"),
        "stream must end with [DONE]: {body}"
    );
    data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

#[tokio::test]
async fn completion_has_openai_shape() {
    let st = state();
    let (status, _, body) = send(
        &st,
        post(
            "/v1/completions",
            json!({"model": "tiny", "prompt": "hi", "max_tokens": 5, "temperature": 0}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(v["choices"][0]["text"].is_string());
    assert_eq!(
        v["usage"],
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
    );
}

#[tokio::test]
async fn streamed_text_equals_non_streamed() {
    let st = state();
    let req = |stream| {
        json!({"prompt": "abc", "max_tokens": 12, "temperature": 0, "stream": stream,
                              "stream_options": {"include_usage": true}})
    };
    let (_, _, full) = send(&st, post("/v1/completions", req(false))).await;
    let full: Value = serde_json::from_str(&full).unwrap();

    let (status, headers, body) = send(&st, post("/v1/completions", req(true))).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let chunks = sse_payloads(&body);
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, full["choices"][0]["text"].as_str().unwrap());
    let finish = &chunks[chunks.len() - 2];
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 12);
}

#[tokio::test]
async fn chat_streams_role_then_deltas() {
    let st = state();
    let body = json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 6,
                      "temperature": 0, "stream": true});
    let (status, _, body) = send(&st, post("/v1/chat/completions", body)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = sse_payloads(&body);
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "length"
    );
    assert!(
        chunks[1..chunks.len() - 1]
            .iter()
            .all(|c| c["choices"][0]["delta"]["content"].is_string())
    );
}

#[tokio::test]
async fn chat_non_streamed_counts_template_tokens() {
    let st = state();
    let body = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "yo"}]}],
                      "max_completion_tokens": 3, "temperature": 0});
    let (status, _, body) = send(&st, post("/v1/chat/completions", body)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    // "<s>" + "user: yo\nassistant: " = 1 + 20 bytes.
    assert_eq!(v["usage"]["prompt_tokens"], 21);
}

/// `n` completions come back as indexed choices: seeded choices differ from each other, and
/// the streamed text of each choice equals the non-streamed one.
#[tokio::test]
async fn n_completions_are_indexed_choices() {
    let st = state();
    let req = |stream| {
        json!({"prompt": "abc", "max_tokens": 12, "temperature": 0.9, "seed": 7, "n": 3,
               "stream": stream, "stream_options": {"include_usage": true}})
    };
    let (status, _, body) = send(&st, post("/v1/completions", req(false))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let texts: Vec<&str> = (0..3)
        .map(|i| {
            assert_eq!(v["choices"][i]["index"], i);
            v["choices"][i]["text"].as_str().unwrap()
        })
        .collect();
    assert_ne!(
        texts[0], texts[1],
        "seed + i gives each choice its own draw"
    );
    assert_eq!(v["usage"]["prompt_tokens"], 4);
    assert_eq!(v["usage"]["completion_tokens"], 36);

    let (_, _, body) = send(&st, post("/v1/completions", req(true))).await;
    let mut streamed = vec![String::new(); 3];
    let mut finished = 0;
    let payloads = sse_payloads(&body);
    for p in &payloads[..payloads.len() - 1] {
        let c = &p["choices"][0];
        streamed[c["index"].as_u64().unwrap() as usize] += c["text"].as_str().unwrap();
        finished += usize::from(!c["finish_reason"].is_null());
    }
    assert_eq!(finished, 3);
    assert_eq!(streamed, texts);
    assert_eq!(payloads.last().unwrap()["usage"]["completion_tokens"], 36);
}

#[tokio::test]
async fn invalid_requests_get_openai_errors() {
    let st = state();
    let cases = [
        (json!({"prompt": "x", "temperature": 3.0}), "temperature"),
        (json!({"prompt": "x", "top_p": 0.0}), "top_p"),
        (json!({"prompt": "x", "top_k": -5}), "top_k"),
        (
            json!({"prompt": "x", "repetition_penalty": 0}),
            "repetition_penalty",
        ),
        (json!({"prompt": "x", "max_tokens": 0}), "max_tokens"),
        (json!({"prompt": "x", "n": 0}), "n"),
        (json!({"prompt": "x", "n": 17}), "n"),
        (
            json!({"prompt": "x", "stop": ["a", "b", "c", "d", "e"]}),
            "stop",
        ),
        (json!({"prompt": ["a", "b"]}), "prompt"),
        (
            json!({"prompt": "x", "max_tokens": MAX_SEQ_LEN}),
            "max_tokens",
        ),
    ];
    for (body, param) in cases {
        let (status, _, resp) = send(&st, post("/v1/completions", body.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} -> {resp}");
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], param, "{body}");
    }
    let (status, _, resp) = send(&st, post("/v1/chat/completions", json!({"messages": []}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
}

#[tokio::test]
async fn malformed_json_is_a_400() {
    let st = state();
    let req = Request::post("/v1/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{not json"))
        .unwrap();
    let (status, _, body) = send(&st, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(serde_json::from_str::<Value>(&body).unwrap()["error"]["message"].is_string());
}

#[tokio::test]
async fn full_queue_returns_503_with_retry_after() {
    // A queue with room for one request and no engine draining it; fill it directly.
    let (engine, _rx) = EngineHandle::channel(
        &EngineConfig {
            queue_size: 1,
            ..EngineConfig::default()
        },
        MAX_SEQ_LEN,
    );
    let (filler, _events) = engine::Request::new(vec![1], engine::SamplingParams::greedy(1));
    engine.submit(filler).unwrap();
    let st = state_with(engine);
    let (status, headers, body) = send(&st, post("/v1/completions", json!({"prompt": "b"}))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(headers[header::RETRY_AFTER], "1");
}

#[tokio::test]
async fn prompt_longer_than_context_names_the_prompt() {
    let st = state();
    let long = "x".repeat(MAX_SEQ_LEN);
    let (status, _, body) = send(&st, post("/v1/completions", json!({"prompt": long}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["param"],
        "prompt"
    );
}

#[tokio::test]
async fn client_disconnect_cancels_the_request() {
    let st = state();
    let req = post(
        "/v1/completions",
        json!({"prompt": "go", "max_tokens": 120, "temperature": 0, "stream": true}),
    );
    let resp = router(st.clone()).oneshot(req).await.unwrap();
    let mut body = resp.into_body();
    // Wait for the first token so the request is definitely running, then hang up.
    let first = body.frame().await.unwrap().unwrap();
    assert!(first.is_data());
    drop(body);
    let deadline = Instant::now() + Duration::from_secs(5);
    while st.engine.running() != 0 {
        assert!(
            Instant::now() < deadline,
            "request still running after disconnect"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

#[tokio::test]
async fn health_reports_engine_state() {
    let st = state();
    let (status, _, body) = send(&st, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["status"],
        "ok"
    );
}

#[tokio::test]
async fn metrics_expose_engine_and_memory_series() {
    let st = state();
    let (status, _, _) = send(
        &st,
        post(
            "/v1/completions",
            json!({"prompt": "metrics", "max_tokens": 4, "temperature": 0}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, headers, body) =
        send(&st, Request::get("/metrics").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/plain")
    );
    for series in [
        "rlis_time_to_first_token_seconds_bucket",
        "rlis_inter_token_latency_seconds_bucket",
        "rlis_generated_tokens_total",
        "rlis_requests_finished_total{reason=\"length\"}",
        "rlis_kv_cache_utilization_ratio",
        "rlis_queue_depth",
        "rlis_process_memory_footprint_bytes",
        "rlis_process_memory_footprint_peak_bytes",
    ] {
        assert!(body.contains(series), "missing {series} in:\n{body}");
    }
}
