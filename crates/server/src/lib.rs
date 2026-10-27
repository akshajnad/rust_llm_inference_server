//! OpenAI-compatible HTTP API over the engine.

pub mod api;
pub mod app;
pub mod memory;

pub use app::{AppState, router};

/// Installs the global Prometheus recorder with the engine's histogram buckets.
pub fn install_prometheus() -> anyhow::Result<metrics_exporter_prometheus::PrometheusHandle> {
    let mut builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    for (name, buckets) in engine::metrics::buckets() {
        builder = builder.set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Full(name.into()),
            &buckets,
        )?;
    }
    Ok(builder.install_recorder()?)
}
