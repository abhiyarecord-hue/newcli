//! Process-based command executor.
//!
//! This is intentionally not a MicroVM or a hard network sandbox. Commands run
//! as OS processes with a cleared environment, bounded output, a hard deadline,
//! and complete process-tree ownership delegated to [`ProcessSupervisor`].

use std::path::Path;
use std::time::Duration;

use agent_types::{AgentError, Result};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::process_tree::ProcessSupervisor;
pub use crate::process_tree::{ExecResult, Termination};

#[async_trait::async_trait]
pub trait SandboxExecutor: Send + Sync {
    async fn execute(
        &self,
        command: &str,
        timeout: Duration,
        cancel: &CancellationToken,
        cwd: &Path,
    ) -> Result<ExecResult>;
}

/// Default process-based executor. It preserves the historical timeout and
/// cancellation errors while the lower-level supervisor exposes typed outcomes.
pub struct ProcessFallback;

impl ProcessFallback {
    /// Run `command` and return the typed outcome without collapsing a timeout
    /// or cancellation into an error.
    ///
    /// [`SandboxExecutor::execute`] maps those terminations to `Err`, which
    /// discards the bounded output captured before termination — exactly the
    /// output a caller needs to explain why a check was killed. This method
    /// keeps it. `Err` is reserved for a command that could not run at all.
    pub async fn execute_typed(
        &self,
        command: &str,
        timeout: Duration,
        cancel: &CancellationToken,
        cwd: &Path,
    ) -> Result<ExecResult> {
        let mut child = Command::new(shell_program());
        child
            .args(shell_args(command))
            .current_dir(cwd)
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env(
                "SYSTEMROOT",
                std::env::var("SYSTEMROOT").unwrap_or_default(),
            );

        ProcessSupervisor::default()
            .spawn(child)?
            .wait_with(timeout, cancel)
            .await
    }
}

#[async_trait::async_trait]
impl SandboxExecutor for ProcessFallback {
    async fn execute(
        &self,
        command: &str,
        timeout: Duration,
        cancel: &CancellationToken,
        cwd: &Path,
    ) -> Result<ExecResult> {
        // One spawn path shared with `execute_typed`; this method only maps the
        // terminal outcome onto the historical error contract. Duplicating the
        // shell and environment setup would let the two diverge.
        let result = self.execute_typed(command, timeout, cancel, cwd).await?;
        match result.termination {
            Termination::Exit => Ok(result),
            Termination::Timeout => Err(AgentError::Sandbox("timeout".into())),
            Termination::Cancelled => Err(AgentError::Cancelled),
        }
    }
}

#[cfg(windows)]
fn shell_program() -> &'static str {
    "cmd.exe"
}

#[cfg(windows)]
fn shell_args(command: &str) -> [&str; 2] {
    ["/C", command]
}

#[cfg(not(windows))]
fn shell_program() -> &'static str {
    "sh"
}

#[cfg(not(windows))]
fn shell_args(command: &str) -> [&str; 2] {
    ["-c", command]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn normal_and_nonzero_execution_are_preserved() {
        let cwd = std::env::current_dir().unwrap();
        let token = CancellationToken::new();
        let normal = ProcessFallback
            .execute("echo hi", Duration::from_secs(5), &token, &cwd)
            .await
            .unwrap();
        assert_eq!(normal.termination, Termination::Exit);
        assert_eq!(normal.exit_code, 0);
        assert!(normal.stdout.contains("hi"));

        #[cfg(windows)]
        let failing = "exit /B 42";
        #[cfg(not(windows))]
        let failing = "exit 42";
        let result = ProcessFallback
            .execute(failing, Duration::from_secs(5), &token, &cwd)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn legacy_timeout_and_cancellation_reporting_is_preserved() {
        let cwd = std::env::current_dir().unwrap();
        #[cfg(windows)]
        let sleeping = "ping -n 30 127.0.0.1 >NUL";
        #[cfg(not(windows))]
        let sleeping = "sleep 30";

        let timeout = ProcessFallback
            .execute(
                sleeping,
                Duration::from_millis(30),
                &CancellationToken::new(),
                &cwd,
            )
            .await;
        assert!(matches!(timeout, Err(AgentError::Sandbox(reason)) if reason == "timeout"));

        let cancel = CancellationToken::new();
        cancel.cancel();
        let cancelled = ProcessFallback
            .execute(sleeping, Duration::from_secs(5), &cancel, &cwd)
            .await;
        assert!(matches!(cancelled, Err(AgentError::Cancelled)));
    }
}
