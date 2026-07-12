//! Thin wrapper over a `tokio::sync::broadcast` channel of [`AgentEvent`]s.
//!
//! Multiple consumers (CLI progress printer, evals trajectory recorder) can
//! subscribe independently. Sends never block the producer and never fail the
//! turn: with no live subscribers the event is simply dropped.

use agent_types::AgentEvent;
use tokio::sync::broadcast;

#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<AgentEvent>,
}

impl EventBus {
    /// Create a bus with a bounded ring buffer of `capacity` events per lagging
    /// subscriber.
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Register a new consumer. Events emitted before subscription are not
    /// replayed.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.tx.subscribe()
    }

    /// Emit an event. Ignores the "no active receivers" case by design.
    pub fn emit(&self, event: AgentEvent) {
        let _ = self.tx.send(event);
    }

    /// Number of currently live subscribers.
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}
