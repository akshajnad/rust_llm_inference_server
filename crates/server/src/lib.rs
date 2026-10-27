//! OpenAI-compatible HTTP API over the engine.

pub mod api;
pub mod app;

pub use app::{AppState, router};
