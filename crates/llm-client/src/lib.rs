//! `llm-client` (L1): provider-agnostic LLM API + SSE streaming parser.
//!
//! - [`sse`]: pure incremental Server-Sent-Events parser (TASK-1.1).
//! - [`provider`]: [`LlmProvider`] trait + [`SseEvent`] (TASK-1.2).
//! - [`anthropic`]: Anthropic Messages API streaming impl (TASK-1.2).
//! - [`gemini`]: Google Gemini API streaming impl.

pub mod anthropic;
pub mod embedder;
pub mod embeddings;
pub mod gemini;
pub mod openai_compat;
pub mod openai_embeddings;
pub mod provider;
mod secret;
pub mod sse;

pub use anthropic::AnthropicProvider;
pub use embedder::{resolve_dimension, Embedder};
pub use embeddings::{GeminiEmbedder, GEMINI_EMBEDDING_DIMENSION};
pub use gemini::GeminiProvider;
pub use openai_compat::{OpenAiCompatProvider, TokenLimitField};
pub use openai_embeddings::OpenAiCompatEmbedder;
pub use provider::{LlmProvider, SseEvent, StopReason};
pub use sse::{RawSseFrame, SseParser, DEFAULT_MAX_FRAME_BYTES};
