//! Structured concurrency primitive.
//!
//! A [`TaskScope`] owns a [`CancellationToken`] and a set of spawned tasks.
//! Every spawned future gets its OWN child token (never a clone of the parent —
//! that would break hierarchical cancel, TASK-0.2 context guard). Cancelling
//! the scope cancels every child; the first task error cancels the remaining
//! children.
//!
//! `JoinHandle::abort()` does NOT run async `Drop` cleanup, so instead of
//! relying on abort we `select!` on the child token *inside* the wrapped
//! future — cooperative cancellation with no orphaned work.

use std::future::Future;

use agent_types::{AgentError, Result};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub struct TaskScope {
    token: CancellationToken,
    handles: Vec<JoinHandle<Result<()>>>,
}

impl TaskScope {
    /// New scope rooted at a fresh cancellation token.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            handles: Vec::new(),
        }
    }

    /// New scope whose token is a child of `parent` — parent cancel cascades in.
    pub fn with_token(parent: CancellationToken) -> Self {
        Self {
            token: parent,
            handles: Vec::new(),
        }
    }

    /// The scope's own token. Children derive from this via `child_token()`.
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }

    /// Cancel the scope and, cooperatively, every task spawned into it.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Spawn a fallible future bound to a fresh child token. If the scope is
    /// cancelled the future is dropped at its next `.await` and the task
    /// resolves to [`AgentError::Cancelled`].
    pub fn spawn<F>(&mut self, f: F)
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        let child = self.token.child_token();
        let handle = tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = child.cancelled() => Err(AgentError::Cancelled),
                res = f => res,
            }
        });
        self.handles.push(handle);
    }

    /// Await every spawned task. On the first task error, cancel the remaining
    /// children (so they stop cooperatively) and return that error; a panicked
    /// or aborted task is reported as a tool error.
    pub async fn join_all(mut self) -> Result<()> {
        let mut outcome: Result<()> = Ok(());
        for handle in self.handles.drain(..) {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if outcome.is_ok() {
                        outcome = Err(err);
                        self.token.cancel();
                    }
                }
                Err(join_err) => {
                    if outcome.is_ok() {
                        outcome = Err(AgentError::Tool {
                            name: "task_scope".to_string(),
                            reason: join_err.to_string(),
                        });
                        self.token.cancel();
                    }
                }
            }
        }
        outcome
    }
}

impl Default for TaskScope {
    fn default() -> Self {
        Self::new()
    }
}
