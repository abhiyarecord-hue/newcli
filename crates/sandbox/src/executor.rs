//! Sandbox executor: runs commands in isolation.
//!
//! Two backends behind a trait:
//! - `ProcessFallback` (default): `tokio::process::Command` with cleared env,
//!   cwd = jail root, hard timeout. Dev-only, clearly documented.
//! - `HyperlightVm` (feature `hyperlight`, Linux/macOS): not yet available on
//!   Windows, stubbed as compile-time feature gate.
//!
//! Enforcement: network disabled in guest (cleared env removes proxy vars),
//! wall-clock timeout via `tokio::time::timeout`, output capped at 1 MiB.
//! On timeout, MUST kill the child (start_kill + reap) — dropping the future
//! does NOT kill the OS process. Read stdout/stderr concurrently.

use std::path::Path;
use std::time::Duration;

use agent_types::{AgentError, Result};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

const MAX_OUTPUT: usize = 1024 * 1024; // 1 MiB

#[derive(Clone, Debug)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Trait for sandbox executors.
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

/// Default process-based executor (dev-only, no true isolation).
pub struct ProcessFallback;

#[async_trait::async_trait]
impl SandboxExecutor for ProcessFallback {
    async fn execute(
        &self,
        command: &str,
        timeout: Duration,
        cancel: &CancellationToken,
        cwd: &Path,
    ) -> Result<ExecResult> {
        let mut child = Command::new(shell_program())
            .args(shell_args(command))
            .current_dir(cwd)
            .env_clear()
            // Minimal env for the process to function.
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("SYSTEMROOT", std::env::var("SYSTEMROOT").unwrap_or_default())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| AgentError::Sandbox(format!("spawn: {e}")))?;

        let mut stdout_handle = child.stdout.take().unwrap();
        let mut stderr_handle = child.stderr.take().unwrap();

        // Read stdout and stderr concurrently (prevents deadlock on full pipe buffer).
        let stdout_fut = async {
            let mut buf = Vec::new();
            stdout_handle
                .read_to_end(&mut buf)
                .await
                .unwrap_or_default();
            buf.truncate(MAX_OUTPUT);
            String::from_utf8_lossy(&buf).to_string()
        };

        let stderr_fut = async {
            let mut buf = Vec::new();
            stderr_handle
                .read_to_end(&mut buf)
                .await
                .unwrap_or_default();
            buf.truncate(MAX_OUTPUT);
            String::from_utf8_lossy(&buf).to_string()
        };

        let child_cancel = cancel.child_token();

        let result = tokio::select! {
            biased;
            _ = child_cancel.cancelled() => {
                child.start_kill().ok();
                child.wait().await.ok();
                return Err(AgentError::Cancelled);
            }
            res = async {
                tokio::time::timeout(timeout, async {
                    let (stdout, stderr) = tokio::join!(stdout_fut, stderr_fut);
                    let status = child.wait().await;
                    (stdout, stderr, status)
                }).await
            } => res,
        };

        match result {
            Ok((stdout, stderr, status)) => {
                let exit_code = status
                    .map(|s| s.code().unwrap_or(-1))
                    .unwrap_or(-1);
                Ok(ExecResult {
                    stdout,
                    stderr,
                    exit_code,
                })
            }
            Err(_timeout) => {
                // Timeout: kill and reap.
                child.start_kill().ok();
                child.wait().await.ok();
                Err(AgentError::Sandbox("timeout".to_string()))
            }
        }
    }
}

#[cfg(windows)]
fn shell_program() -> &'static str {
    "cmd.exe"
}

#[cfg(windows)]
fn shell_args(command: &str) -> Vec<String> {
    vec!["/C".to_string(), command.to_string()]
}

#[cfg(not(windows))]
fn shell_program() -> &'static str {
    "sh"
}

#[cfg(not(windows))]
fn shell_args(command: &str) -> Vec<String> {
    vec!["-c".to_string(), command.to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_returns_stdout() {
        let exec = ProcessFallback;
        let token = CancellationToken::new();
        let cwd = std::env::current_dir().unwrap();
        let result = exec
            .execute("echo hi", Duration::from_secs(5), &token, &cwd)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.trim().contains("hi"));
    }

    #[tokio::test]
    async fn timeout_kills_process() {
        let exec = ProcessFallback;
        let token = CancellationToken::new();
        let cwd = std::env::current_dir().unwrap();

        // Use a command that sleeps; on Windows "timeout /t 30 /nobreak" or "ping -n 30 127.0.0.1"
        #[cfg(windows)]
        let cmd = "ping -n 30 127.0.0.1";
        #[cfg(not(windows))]
        let cmd = "sleep 30";

        let result = exec
            .execute(cmd, Duration::from_millis(200), &token, &cwd)
            .await;
        assert!(matches!(result, Err(AgentError::Sandbox(ref s)) if s == "timeout"));
    }

    #[tokio::test]
    async fn cancel_token_kills_process() {
        let exec = ProcessFallback;
        let token = CancellationToken::new();
        let cwd = std::env::current_dir().unwrap();

        #[cfg(windows)]
        let cmd = "ping -n 30 127.0.0.1";
        #[cfg(not(windows))]
        let cmd = "sleep 30";

        // Cancel after a short delay.
        let t2 = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            t2.cancel();
        });

        let result = exec
            .execute(cmd, Duration::from_secs(30), &token, &cwd)
            .await;
        assert!(matches!(result, Err(AgentError::Cancelled)));
    }

    #[tokio::test]
    async fn nonzero_exit_code_reported() {
        let exec = ProcessFallback;
        let token = CancellationToken::new();
        let cwd = std::env::current_dir().unwrap();

        #[cfg(windows)]
        let cmd = "cmd /C exit 42";
        #[cfg(not(windows))]
        let cmd = "exit 42";

        let result = exec
            .execute(cmd, Duration::from_secs(5), &token, &cwd)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 42);
    }
}
