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

/// Drains histogram samples into their buckets every few seconds. `install_recorder` doesn't
/// spawn this, and without it every recorded sample (one per generated token for inter-token
/// latency) is kept until the next scrape, so memory grows without bound when nobody scrapes.
pub fn spawn_metrics_upkeep(handle: metrics_exporter_prometheus::PrometheusHandle) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            tick.tick().await;
            handle.run_upkeep();
        }
    });
}
