//! `runtime-core` (L1): structured-concurrency runtime.
//!
//! - [`TaskScope`]: hierarchical, cooperatively-cancellable task group.
//! - [`EventBus`]: broadcast channel of [`agent_types::AgentEvent`].
//! - [`Scheduler`]: owns the root token + shared bus, hands out child scopes.
//!
//! Cancellation is hierarchical: a parent cancel implies children cancel
//! (plan.md section 3, cross-cutting rule 2).

pub mod event_bus;
pub mod scheduler;
pub mod task_scope;

pub use event_bus::EventBus;
pub use scheduler::Scheduler;
pub use task_scope::TaskScope;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use agent_types::AgentEvent;

    #[tokio::test]
    async fn parent_cancel_propagates_to_children_within_100ms() {
        let mut scope = TaskScope::new();
        let ran_to_completion = Arc::new(AtomicBool::new(false));

        for _ in 0..5 {
            let flag = ran_to_completion.clone();
            scope.spawn(async move {
                // Long sleep that must be cut short by cancellation.
                tokio::time::sleep(Duration::from_secs(10)).await;
                flag.store(true, Ordering::SeqCst);
                Ok(())
            });
        }

        scope.cancel();
        let start = Instant::now();
        let result = scope.join_all().await;
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_millis(100),
            "children did not cancel promptly: {elapsed:?}"
        );
        assert!(
            !ran_to_completion.load(Ordering::SeqCst),
            "a child ran to completion despite cancellation (orphaned future)"
        );
        assert!(matches!(result, Err(agent_types::AgentError::Cancelled)));
    }

    #[tokio::test]
    async fn first_error_cancels_remaining_children() {
        let mut scope = TaskScope::new();
        let sibling_completed = Arc::new(AtomicBool::new(false));

        // A task that fails quickly.
        scope.spawn(async {
            Err(agent_types::AgentError::Llm("boom".to_string()))
        });
        // A sibling that would take a while; must be cancelled by the error.
        let flag = sibling_completed.clone();
        scope.spawn(async move {
            tokio::time::sleep(Duration::from_secs(10)).await;
            flag.store(true, Ordering::SeqCst);
            Ok(())
        });

        let result = scope.join_all().await;
        assert!(result.is_err());
        assert!(!sibling_completed.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn event_bus_broadcasts_to_subscribers() {
        let bus = EventBus::new(16);
        let mut rx = bus.subscribe();
        bus.emit(AgentEvent::TurnStarted);
        bus.emit(AgentEvent::ToolInvoked {
            name: "bash".to_string(),
        });

        assert!(matches!(rx.recv().await.unwrap(), AgentEvent::TurnStarted));
        assert!(matches!(
            rx.recv().await.unwrap(),
            AgentEvent::ToolInvoked { .. }
        ));
    }
}
