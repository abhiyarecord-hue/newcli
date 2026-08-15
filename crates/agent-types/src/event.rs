//! Agent lifecycle events broadcast over the [`EventBus`](../../runtime-core).
//! Consumed by the CLI (progress) and the evals trajectory recorder (TASK-10.2).

/// Recoverable evidence that a bounded event receiver skipped messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EventLag {
    pub skipped: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AgentEvent {
    TurnStarted,
    ApiCallStarted,
    Thinking {
        text: String,
    },
    ToolInvoked {
        name: String,
    },
    ToolCompleted {
        name: String,
    },
    TokenUsage {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    },
    /// An auditable, non-terminal representation of receiver lag.
    EventLagged(EventLag),
    TurnEnded,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_lag_round_trips_the_exact_skipped_count() {
        // **Validates: Requirements 2.43**
        let event = AgentEvent::EventLagged(EventLag { skipped: 17 });
        let json = serde_json::to_string(&event).unwrap();
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            restored,
            AgentEvent::EventLagged(EventLag { skipped: 17 })
        ));
    }
}
