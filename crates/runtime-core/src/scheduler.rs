//! Top-level scheduler: owns the shared [`EventBus`] and the root
//! cancellation token, and hands out child [`TaskScope`]s so all concurrency
//! in the process is hierarchically cancellable from a single root (e.g. the
//! CLI's Ctrl-C handler in TASK-9.3).

use tokio_util::sync::CancellationToken;

use crate::event_bus::EventBus;
use crate::task_scope::TaskScope;

pub struct Scheduler {
    root: CancellationToken,
    bus: EventBus,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            root: CancellationToken::new(),
            bus: EventBus::default(),
        }
    }

    /// The shared event bus for lifecycle events.
    pub fn event_bus(&self) -> &EventBus {
        &self.bus
    }

    /// The root cancellation token. Cancel it to drain the whole process.
    pub fn root_token(&self) -> &CancellationToken {
        &self.root
    }

    /// A fresh child scope. Cancelling the root cancels every scope handed out.
    pub fn scope(&self) -> TaskScope {
        TaskScope::with_token(self.root.child_token())
    }

    /// Signal graceful shutdown to every scope derived from this scheduler.
    pub fn shutdown(&self) {
        self.root.cancel();
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
