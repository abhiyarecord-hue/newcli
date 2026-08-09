//! Serialized owner of process standard input for every interactive CLI prompt.

use std::io::IsTerminal;
use std::pin::Pin;
use std::sync::Arc;

use agent_types::{ApprovalDecision, ApprovalProvider, ApprovalRequest, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

use crate::spec_input::DESCRIPTION_TERMINATOR;

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

    /// Read a multi-line block, ending at a line equal to
    /// [`DESCRIPTION_TERMINATOR`] or at end of input.
    ///
    /// Blank lines are preserved. Terminating on a blank line, as this once
    /// did, corrupted two things at once: a pasted document was cut at its
    /// first paragraph break, and the unread remainder stayed in the shared
    /// stdin buffer where the next prompt — including an approval prompt — read
    /// it as its answer. Input beyond `max_bytes` is therefore drained to the
    /// terminator before the error returns, so a rejected paste cannot leak
    /// into a later prompt either.
    pub async fn read_multiline_until_terminator(
        &self,
        max_bytes: usize,
    ) -> std::io::Result<String> {
        let mut reader = self.reader.lock().await;
        let mut input = String::new();
        let mut overflowed = false;
        loop {
            let mut line = String::new();
            if reader.as_mut().read_line(&mut line).await? == 0 {
                break;
            }
            if line.trim().eq_ignore_ascii_case(DESCRIPTION_TERMINATOR) {
                break;
            }
            if overflowed {
                continue;
            }
            if input.len() + line.len() > max_bytes {
                overflowed = true;
                input = String::new();
                continue;
            }
            input.push_str(&line);
        }

        if overflowed {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "description exceeds the {max_bytes}-byte limit. \
                     Put it in a file and pass --from-file <path>, \
                     or reference it as @<path>"
                ),
            ));
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

    #[tokio::test]
    async fn pasted_description_keeps_blank_lines_and_leaves_later_input_intact() {
        // The regression this replaces: a blank line ended the description, so
        // only "# Goal" reached the model and every following line was read by
        // the next prompt as its answer.
        let script = b"# Goal\n\nBuild a todo CLI.\n\n- keep it small\n/end\n/quit\n";
        let broker = InputBroker::new(BufReader::new(&script[..]), true);

        let description = broker
            .read_multiline_until_terminator(64 * 1024)
            .await
            .unwrap();

        assert_eq!(
            description,
            "# Goal\n\nBuild a todo CLI.\n\n- keep it small\n"
        );
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "/quit\n");
    }

    #[tokio::test]
    async fn end_of_input_terminates_a_piped_description() {
        let broker = InputBroker::new(BufReader::new(&b"line one\n\nline two\n"[..]), false);
        assert_eq!(
            broker
                .read_multiline_until_terminator(64 * 1024)
                .await
                .unwrap(),
            "line one\n\nline two\n"
        );
        assert!(broker.read_line().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn terminator_is_case_insensitive_and_may_be_indented() {
        let broker = InputBroker::new(BufReader::new(&b"body\n  /END  \nafter\n"[..]), true);
        assert_eq!(
            broker
                .read_multiline_until_terminator(64 * 1024)
                .await
                .unwrap(),
            "body\n"
        );
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "after\n");
    }

    #[tokio::test]
    async fn oversized_paste_fails_and_is_drained_rather_than_left_buffered() {
        let script = b"aaaaaaaaaa\nbbbbbbbbbb\n/end\nnext prompt\n";
        let broker = InputBroker::new(BufReader::new(&script[..]), true);

        let error = broker
            .read_multiline_until_terminator(12)
            .await
            .expect_err("a paste over the cap must fail visibly");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

        // The rejected paste must not become the answer to the next prompt.
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "next prompt\n");
    }
}
