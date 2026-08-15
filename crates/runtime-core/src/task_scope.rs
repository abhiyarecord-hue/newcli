//! Completion-order structured concurrency.
//!
//! A [`TaskScope`] owns every spawned task until it completes or is aborted.
//! Parent cancellation cascades into the scope, while cancelling this scope
//! does not cancel its parent or unrelated sibling scopes. The first completed
//! error or panic is observed promptly; remaining work is cancelled, aborted
//! if non-cooperative, and fully drained before [`TaskScope::join_all`] returns.

use std::future::Future;

use agent_types::{AgentError, Result};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

pub struct TaskScope {
    token: CancellationToken,
    tasks: JoinSet<Result<()>>,
}

impl TaskScope {
    /// New scope rooted at a fresh cancellation token.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            tasks: JoinSet::new(),
        }
    }

    /// New child scope. Parent cancellation cascades into this scope without
    /// allowing scope-local cancellation to affect the parent.
    pub fn with_token(parent: CancellationToken) -> Self {
        Self {
            token: parent.child_token(),
            tasks: JoinSet::new(),
        }
    }

    /// The scope's own token. Children derive from this via `child_token()`.
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }

    /// Cancel the scope and every task spawned into it.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Spawn a fallible future bound to a fresh child token. Cooperative tasks
    /// observe cancellation at an await point; non-cooperative tasks are
    /// aborted by [`TaskScope::join_all`] after the first failure.
    pub fn spawn<F>(&mut self, future: F)
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        let child = self.token.child_token();
        self.tasks.spawn(async move {
            tokio::select! {
                biased;
                _ = child.cancelled() => Err(AgentError::Cancelled),
                result = future => result,
            }
        });
    }

    /// Observe tasks in completion order. After the first error or panic, all
    /// siblings are cancelled and aborted, then every join result is drained.
    pub async fn join_all(mut self) -> Result<()> {
        let mut first_error = None;

        while let Some(joined) = self.tasks.join_next().await {
            let error = match joined {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(join_error) if join_error.is_cancelled() && first_error.is_some() => None,
                Err(join_error) => Some(AgentError::Tool {
                    name: "task_scope".to_string(),
                    reason: join_error.to_string(),
                }),
            };

            if first_error.is_none() {
                if let Some(error) = error {
                    first_error = Some(error);
                    self.token.cancel();
                    self.tasks.abort_all();
                }
            }
        }

        first_error.map_or(Ok(()), Err)
    }
}

impl Drop for TaskScope {
    fn drop(&mut self) {
        self.token.cancel();
        self.tasks.abort_all();
    }
}

impl Default for TaskScope {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn successful_tasks_complete_and_preserve_their_results() {
        let results = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut scope = TaskScope::new();

        for (delay_ms, value) in [(10, "first"), (0, "second")] {
            let results = results.clone();
            scope.spawn(async move {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                results.lock().await.push(value);
                Ok(())
            });
        }

        assert!(scope.join_all().await.is_ok());
        let mut completed = results.lock().await.clone();
        completed.sort_unstable();
        assert_eq!(completed, ["first", "second"]);
    }

    #[tokio::test]
    async fn later_spawned_failure_is_observed_before_hanging_sibling() {
        let stopped = Arc::new(AtomicBool::new(false));
        let mut scope = TaskScope::new();
        let stopped_by_drop = stopped.clone();
        scope.spawn(async move {
            struct StopGuard(Arc<AtomicBool>);
            impl Drop for StopGuard {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let _guard = StopGuard(stopped_by_drop);
            std::future::pending::<Result<()>>().await
        });
        scope.spawn(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Err(AgentError::Llm("later failure".into()))
        });

        let result = tokio::time::timeout(Duration::from_millis(100), scope.join_all())
            .await
            .expect("failure must not wait for spawn-order sibling");
        assert!(matches!(result, Err(AgentError::Llm(message)) if message == "later failure"));
        assert!(stopped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn panic_is_reported_and_non_cooperative_sibling_is_aborted() {
        let mut scope = TaskScope::new();
        scope.spawn(std::future::pending::<Result<()>>());
        scope.spawn(async { panic!("scoped panic") });

        let result = tokio::time::timeout(Duration::from_millis(100), scope.join_all())
            .await
            .expect("panic must be observed promptly");
        assert!(matches!(result, Err(AgentError::Tool { name, reason })
            if name == "task_scope" && reason.contains("scoped panic")));
    }

    #[tokio::test]
    async fn scope_cancellation_is_isolated_but_parent_cancellation_cascades() {
        let parent = CancellationToken::new();
        let mut cancelled_scope = TaskScope::with_token(parent.clone());
        let mut sibling_scope = TaskScope::with_token(parent.clone());
        cancelled_scope.spawn(std::future::pending::<Result<()>>());
        sibling_scope.spawn(std::future::pending::<Result<()>>());

        cancelled_scope.cancel();
        assert!(matches!(
            cancelled_scope.join_all().await,
            Err(AgentError::Cancelled)
        ));
        assert!(!parent.is_cancelled());
        assert!(!sibling_scope.token().is_cancelled());

        parent.cancel();
        assert!(matches!(
            sibling_scope.join_all().await,
            Err(AgentError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn dropping_scope_before_first_poll_prevents_task_start() {
        let started = Arc::new(AtomicBool::new(false));
        {
            let mut scope = TaskScope::new();
            let task_started = started.clone();
            scope.spawn(async move {
                task_started.store(true, Ordering::SeqCst);
                Ok(())
            });
        }

        tokio::task::yield_now().await;
        assert!(!started.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn dropping_running_scope_stops_owned_work_without_cancelling_parent() {
        let parent = CancellationToken::new();
        let ticks = Arc::new(AtomicUsize::new(0));
        {
            let mut scope = TaskScope::with_token(parent.clone());
            let task_ticks = ticks.clone();
            scope.spawn(async move {
                loop {
                    task_ticks.fetch_add(1, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                }
                #[allow(unreachable_code)]
                Ok(())
            });
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let after_drop = ticks.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(ticks.load(Ordering::SeqCst), after_drop);
        assert!(!parent.is_cancelled());
    }
}
