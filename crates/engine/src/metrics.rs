//! Metric names the engine records through the `metrics` facade. Recording is a no-op until a
//! recorder is installed (the server installs a Prometheus one), so tests and the CLI pay
//! nothing.

pub const TTFT: &str = "rlis_time_to_first_token_seconds";
pub const ITL: &str = "rlis_inter_token_latency_seconds";
pub const REQUEST_DECODE_TPS: &str = "rlis_request_decode_tokens_per_second";
pub const GENERATED_TOKENS: &str = "rlis_generated_tokens_total";
pub const REQUESTS_FINISHED: &str = "rlis_requests_finished_total";
pub const PREEMPTIONS: &str = "rlis_preemptions_total";
pub const RUNNING: &str = "rlis_running_sequences";
pub const WAITING: &str = "rlis_waiting_sequences";
pub const KV_UTILIZATION: &str = "rlis_kv_cache_utilization_ratio";
pub const STEP_TOKENS: &str = "rlis_step_tokens";

/// Histogram buckets per metric, for exporters that need them up front.
pub fn buckets() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        (
            TTFT,
            vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        ),
        (
            ITL,
            vec![0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0],
        ),
        (
            REQUEST_DECODE_TPS,
            vec![5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 160.0, 240.0],
        ),
        (
            STEP_TOKENS,
            vec![1.0, 4.0, 16.0, 32.0, 64.0, 128.0, 512.0, 2048.0],
        ),
    ]
}
