//! Agent lifecycle events broadcast over the [`EventBus`](../../runtime-core).
//! Consumed by the CLI (progress) and the evals trajectory recorder (TASK-10.2).

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AgentEvent {
    TurnStarted,
    Thinking { text: String },
    ToolInvoked { name: String },
    ToolCompleted { name: String },
    TokenUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },
    TurnEnded,
}
