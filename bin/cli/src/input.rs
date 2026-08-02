//! Serialized owner of process standard input for every interactive CLI prompt.

use std::io::IsTerminal;
use std::pin::Pin;
use std::sync::Arc;

use agent_types::{ApprovalDecision, ApprovalProvider, ApprovalRequest, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionMode {
    Vibe,
    RustySpec,
}

type InputReader = Pin<Box<dyn AsyncBufRead + Send>>;

/// Owns the only stdin reader and serializes chat, mode, and approval requests.
pub struct InputBroker {
    reader: Mutex<InputReader>,
    interactive: bool,
}

impl InputBroker {
    pub fn stdio() -> Arc<Self> {
        let interactive = std::io::stdin().is_terminal();
        Arc::new(Self::new(BufReader::new(tokio::io::stdin()), interactive))
    }

    fn new(reader: impl AsyncBufRead + Send + 'static, interactive: bool) -> Self {
        Self {
            reader: Mutex::new(Box::pin(reader)),
            interactive,
        }
    }

    pub fn is_interactive(&self) -> bool {
        self.interactive
    }

    pub async fn select_mode(&self) -> std::io::Result<SessionMode> {
        if !self.interactive {
            return Ok(SessionMode::Vibe);
        }
        Ok(match self.read_line().await?.as_deref().map(str::trim) {
            Some("2") => SessionMode::RustySpec,
            _ => SessionMode::Vibe,
        })
    }
    pub async fn read_line(&self) -> std::io::Result<Option<String>> {
        let mut reader = self.reader.lock().await;
        let mut line = String::new();
        match reader.as_mut().read_line(&mut line).await? {
            0 => Ok(None),
            _ => Ok(Some(line)),
        }
    }

    pub async fn read_multiline_until_blank(&self) -> std::io::Result<String> {
        let mut reader = self.reader.lock().await;
        let mut input = String::new();
        loop {
            let mut line = String::new();
            if reader.as_mut().read_line(&mut line).await? == 0 || line.trim().is_empty() {
                break;
            }
            input.push_str(&line);
        }
        Ok(input)
    }
}

#[async_trait::async_trait]
impl ApprovalProvider for InputBroker {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalDecision> {
        if !self.interactive {
            return Ok(ApprovalDecision::Denied {
                reason: "interactive approval is unavailable on redirected stdin".into(),
            });
        }

        use std::io::Write;
        eprintln!("\n  [!] {}", request.prompt);
        eprint!("  Allow? [y/N]: ");
        std::io::stderr().flush().ok();
        let approved = self
            .read_line()
            .await?
            .is_some_and(|line| matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes"));
        Ok(if approved {
            ApprovalDecision::Approved
        } else {
            ApprovalDecision::Denied {
                reason: "denied by user".into(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_types::{ApprovalKind, ToolEffects};
    use tokio::io::BufReader;

    fn request() -> ApprovalRequest {
        ApprovalRequest {
            id: "test".into(),
            kind: ApprovalKind::ToolExecution,
            prompt: "run test command".into(),
            tool_name: Some("bash".into()),
            requested_input: serde_json::json!({"command":"test"}),
            effects: ToolEffects::PROCESS_SPAWN,
        }
    }

    #[tokio::test]
    async fn redirected_mode_and_approval_never_consume_the_first_chat_line() {
        // **Validates: Requirements 2.27, 2.42**
        let broker = InputBroker::new(BufReader::new(&b"FIRST_PROMPT\n"[..]), false);
        assert_eq!(broker.select_mode().await.unwrap(), SessionMode::Vibe);
        assert!(matches!(
            broker.request_approval(request()).await.unwrap(),
            ApprovalDecision::Denied { .. }
        ));
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "FIRST_PROMPT\n");
    }

    #[tokio::test]
    async fn interactive_mode_and_approval_share_one_serial_reader() {
        // **Validates: Requirements 2.27, 3.9**
        let broker = InputBroker::new(BufReader::new(&b"2\nyes\nchat prompt\n"[..]), true);
        assert_eq!(broker.select_mode().await.unwrap(), SessionMode::RustySpec);
        assert_eq!(
            broker.request_approval(request()).await.unwrap(),
            ApprovalDecision::Approved
        );
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "chat prompt\n");
    }
}
