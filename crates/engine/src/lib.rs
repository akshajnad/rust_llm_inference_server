//! Request execution: the engine thread, sampling, stop strings, detokenization and chat
//! templating. The continuous-batching scheduler lands here in M3.

pub mod chat;
pub mod detokenizer;
pub mod engine;
pub mod metrics;
pub mod perplexity;
pub mod sampler;
pub mod scheduler;
pub mod stop;
#[cfg(feature = "test-util")]
pub mod testing;
pub mod tokenizer;

pub use chat::{ChatMessage, ChatTemplate};
pub use engine::{
    EngineConfig, EngineHandle, Event, Finish, FinishReason, Request, SubmitError, gib, spawn,
};
pub use model::AttentionImpl;
pub use sampler::{Sampler, SamplingParams};
pub use scheduler::Policy;
pub use tokenizer::Tokenizer;
