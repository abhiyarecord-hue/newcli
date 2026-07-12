//! Provider-agnostic streaming LLM interface. Signatures verbatim from
//! plan.md section 3.

use agent_types::{Message, Result, ToolSchema};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub enum SseEvent {
    Delta(String),
    /// Thought summary from a thinking model (streamed incrementally).
    Thinking(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Stop {
        reason: StopReason,
    },
    /// Token usage reported by the provider (emitted once per stream, typically
    /// on the final chunk).
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        cancel: &CancellationToken,
    ) -> Result<tokio::sync::mpsc::Receiver<SseEvent>>;
}
