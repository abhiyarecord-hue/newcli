//! Serialized owner of process standard input for every interactive CLI prompt.

use std::io::IsTerminal;
use std::pin::Pin;
use std::sync::Arc;

use agent_types::{ApprovalDecision, ApprovalProvider, ApprovalRequest, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

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
    /// [`DESCRIPTION_TERMINATOR`], at end of input, or on cancellation.
    ///
    /// Blank lines are preserved. Terminating on a blank line, as this once
    /// did, corrupted two things at once: a pasted document was cut at its
    /// first paragraph break, and the unread remainder stayed in the shared
    /// stdin buffer where the next prompt — including an approval prompt — read
    /// it as its answer. Input beyond `max_bytes` is therefore drained to the
    /// terminator before the error returns, so a rejected paste cannot leak
    /// into a later prompt either.
    ///
    /// Cancellation is a distinct outcome, never a submission. Ctrl-C makes a
    /// pending console read return end-of-input, which is indistinguishable
    /// from a finished paste by return value alone; treating it as the end of
    /// the description ran the stage on a half-typed prompt, which is the
    /// opposite of what the person asked for. The token is therefore both
    /// selected on and re-checked after the loop.
    ///
    /// `blank_streak_hint` is invoked with the number of consecutive blank
    /// lines seen so far, so a caller can remind the user how to finish. A
    /// blank line no longer submits, and silently doing nothing looks like a
    /// hang.
    pub async fn read_multiline_until_terminator(
        &self,
        max_bytes: usize,
        cancel: &CancellationToken,
        mut blank_streak_hint: impl FnMut(usize),
    ) -> std::io::Result<MultilineInput> {
        let mut reader = self.reader.lock().await;
        let mut input = String::new();
        let mut overflowed = false;
        let mut blank_streak = 0usize;

        loop {
            let mut line = String::new();
            let read = {
                // The pinned handle is bound separately: it is a temporary, and
                // holding the read future across `select!` would outlive it.
                let mut handle = reader.as_mut();
                let pending = handle.read_line(&mut line);
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Ok(MultilineInput::Cancelled),
                    read = pending => read?,
                }
            };
            if read == 0 {
                break;
            }
            if line.trim().eq_ignore_ascii_case(DESCRIPTION_TERMINATOR) {
                break;
            }

            if line.trim().is_empty() {
                blank_streak += 1;
                blank_streak_hint(blank_streak);
            } else {
                blank_streak = 0;
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

        // Ctrl-C during a console read surfaces as end-of-input, so the token
        // is the only reliable way to tell an interrupt from a finished paste.
        if cancel.is_cancelled() {
            return Ok(MultilineInput::Cancelled);
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
        Ok(MultilineInput::Text(input))
    }
}

/// Outcome of reading a multi-line block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultilineInput {
    Text(String),
    /// The reader was interrupted. The caller must not act on partial input.
    Cancelled,
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
    use std::time::Duration;
    use tokio::io::BufReader;

    /// Read with no cancellation and no hint, which is what most cases want.
    async fn read(broker: &InputBroker, max_bytes: usize) -> std::io::Result<MultilineInput> {
        broker
            .read_multiline_until_terminator(max_bytes, &CancellationToken::new(), |_| {})
            .await
    }

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

        let description = read(&broker, 64 * 1024).await.unwrap();

        assert_eq!(
            description,
            MultilineInput::Text("# Goal\n\nBuild a todo CLI.\n\n- keep it small\n".into())
        );
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "/quit\n");
    }

    #[tokio::test]
    async fn end_of_input_terminates_a_piped_description() {
        let broker = InputBroker::new(BufReader::new(&b"line one\n\nline two\n"[..]), false);
        assert_eq!(
            read(&broker, 64 * 1024).await.unwrap(),
            MultilineInput::Text("line one\n\nline two\n".into())
        );
        assert!(broker.read_line().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn terminator_is_case_insensitive_and_may_be_indented() {
        let broker = InputBroker::new(BufReader::new(&b"body\n  /END  \nafter\n"[..]), true);
        assert_eq!(
            read(&broker, 64 * 1024).await.unwrap(),
            MultilineInput::Text("body\n".into())
        );
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "after\n");
    }

    #[tokio::test]
    async fn oversized_paste_fails_and_is_drained_rather_than_left_buffered() {
        let script = b"aaaaaaaaaa\nbbbbbbbbbb\n/end\nnext prompt\n";
        let broker = InputBroker::new(BufReader::new(&script[..]), true);

        let error = read(&broker, 12)
            .await
            .expect_err("a paste over the cap must fail visibly");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

        // The rejected paste must not become the answer to the next prompt.
        assert_eq!(broker.read_line().await.unwrap().unwrap(), "next prompt\n");
    }

    #[tokio::test]
    async fn an_interrupted_description_is_cancelled_not_submitted() {
        // Observed live: Enter did not finish the description, Ctrl-C was
        // pressed, and the stage ran on the half-typed text. Ctrl-C makes the
        // pending console read return end-of-input, so the return value alone
        // cannot distinguish it from a finished paste; the token must.
        let broker = InputBroker::new(BufReader::new(&b"read @req.md and do\n"[..]), true);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let outcome = broker
            .read_multiline_until_terminator(64 * 1024, &cancel, |_| {})
            .await
            .unwrap();

        assert_eq!(outcome, MultilineInput::Cancelled);
    }

    #[tokio::test]
    async fn cancelling_while_still_waiting_discards_what_was_typed() {
        // A slice reader reaches end-of-input immediately, which is not the
        // situation being tested. A console keeps the read pending, so this uses
        // a pipe whose writer stays open to hold the reader mid-description —
        // exactly where Ctrl-C was pressed.
        let (client, server) = tokio::io::duplex(256);
        let broker = InputBroker::new(BufReader::new(server), true);
        let cancel = CancellationToken::new();

        let mut client = client;
        tokio::io::AsyncWriteExt::write_all(&mut client, b"half a thought\n")
            .await
            .unwrap();

        let canceller = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            canceller.cancel();
        });

        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            broker.read_multiline_until_terminator(64 * 1024, &cancel, |_| {}),
        )
        .await
        .expect("cancellation must not hang")
        .unwrap();

        assert_eq!(outcome, MultilineInput::Cancelled);
        // Keep the writer alive so the reader really was pending, not at EOF.
        drop(client);
    }

    #[tokio::test]
    async fn blank_lines_report_a_streak_so_the_prompt_can_explain_itself() {
        // A blank line no longer submits, so doing nothing visible when Enter is
        // pressed looks like a hang. The caller needs to know to say something.
        let broker = InputBroker::new(BufReader::new(&b"one\n\n\n\ntwo\n\n/end\n"[..]), true);
        let streaks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = streaks.clone();

        let outcome = broker
            .read_multiline_until_terminator(64 * 1024, &CancellationToken::new(), move |streak| {
                seen.lock().unwrap().push(streak)
            })
            .await
            .unwrap();

        assert_eq!(
            outcome,
            MultilineInput::Text("one\n\n\n\ntwo\n\n".into()),
            "blank lines must still be preserved verbatim"
        );
        assert_eq!(*streaks.lock().unwrap(), vec![1, 2, 3, 1]);
    }
}
