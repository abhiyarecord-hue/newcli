//! `llm-client` (L1): provider-agnostic LLM API + SSE streaming parser.
//!
//! - [`sse`]: pure incremental Server-Sent-Events parser (TASK-1.1).
//! - [`provider`]: [`LlmProvider`] trait + [`SseEvent`] (TASK-1.2).
//! - [`anthropic`]: Anthropic Messages API streaming impl (TASK-1.2).
//! - [`gemini`]: Google Gemini API streaming impl.

pub mod anthropic;
pub mod embeddings;
pub mod gemini;
pub mod openai_compat;
pub mod provider;
pub mod sse;

pub use anthropic::AnthropicProvider;
pub use embeddings::GeminiEmbedder;
pub use gemini::GeminiProvider;
pub use openai_compat::OpenAiCompatProvider;
pub use provider::{LlmProvider, SseEvent, StopReason};
pub use sse::{RawSseFrame, SseParser};
