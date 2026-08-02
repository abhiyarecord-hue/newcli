//! Bounded, honest text completion for non-Implement specification stages.
//!
//! A specification artifact is published only when the provider actually
//! finished. A `MaxTokens` stop is continued a bounded number of times, each
//! continuation is appended with its overlapping prefix removed, and a
//! continuation that repeats without adding new text is reported as a lack of
//! progress instead of being published as complete.
//!
//! The caller must not write an artifact for any [`SpecOutputError`].

use std::sync::Arc;

use llm_client::{LlmProvider, SseEvent, StopReason};
use tokio_util::sync::CancellationToken;

/// Maximum automatic continuations after a `MaxTokens` stop.
///
/// Matches the orchestrator's chat-path bound so both surfaces behave the same.
pub const MAX_SPEC_CONTINUATIONS: u32 = 5;

/// Maximum time to wait for the next event from a stage stream.
///
/// Without this a silent provider would block a stage forever. The bound is per
/// event, not per stage, so a slow but progressing stream is not cut off.
pub const STREAM_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Continuation instruction, kept identical to the orchestrator's wording.
pub const CONTINUATION_PROMPT: &str =
    "Continue from where you left off. Do not repeat what you already wrote.";

/// Provider text that reached a genuine end of turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletedText {
    /// Full accumulated text across every continuation.
    pub text: String,
    /// Terminal stop reason actually reported by the provider.
    pub stop_reason: StopReason,
    /// How many continuations were needed to finish.
    pub continuations: u32,
}

/// Why a stage produced no publishable artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecOutputError {
    /// The continuation bound was exhausted while still truncated.
    BoundExhausted { continuations: u32, bytes: usize },
    /// A continuation added no new text, so the output stopped progressing.
    NoProgress { continuations: u32, bytes: usize },
    /// The stream ended without any terminal event.
    UnterminatedStream { bytes: usize },
    /// The provider produced no event within the idle bound.
    IdleTimeout { bytes: usize, seconds: u64 },
    /// The caller cancelled the stage.
    Cancelled,
    /// The provider reported a failure.
    Provider(String),
}

/// Shared marker, re-exported locally so the format strings stay readable.
const MARKER: &str = agent_types::INCOMPLETE_OUTPUT_MARKER;

impl std::fmt::Display for SpecOutputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Every incomplete variant renders the shared marker so one search
            // detects truncation regardless of which bound was hit.
            Self::BoundExhausted {
                continuations,
                bytes,
            } => write!(
                formatter,
                "{MARKER} output still truncated after {continuations} continuations \
                 ({bytes} bytes); no artifact was written"
            ),
            Self::NoProgress {
                continuations,
                bytes,
            } => write!(
                formatter,
                "{MARKER} the model stopped making progress after {continuations} \
                 continuations ({bytes} bytes); no artifact was written"
            ),
            Self::UnterminatedStream { bytes } => write!(
                formatter,
                "{MARKER} provider stream ended without a terminal event ({bytes} bytes); \
                 no artifact was written"
            ),
            Self::IdleTimeout { bytes, seconds } => write!(
                formatter,
                "{MARKER} provider sent no data for {seconds}s ({bytes} bytes); \
                 no artifact was written"
            ),
            Self::Cancelled => {
                write!(
                    formatter,
                    "{MARKER} stage cancelled; no artifact was written"
                )
            }
            Self::Provider(error) => write!(formatter, "provider error: {error}"),
        }
    }
}

impl std::error::Error for SpecOutputError {}

/// Append `chunk` to `accumulated`, dropping the longest chunk prefix that is
/// already a suffix of `accumulated`.
///
/// Returns `true` when new text was actually added. A continuation that only
/// restates what is already present therefore reports no progress.
pub fn append_with_overlap_dedup(accumulated: &mut String, chunk: &str) -> bool {
    if chunk.is_empty() {
        return false;
    }
    if accumulated.is_empty() {
        accumulated.push_str(chunk);
        return true;
    }

    // Compare on character boundaries so multibyte text is never sliced.
    let overlap = longest_overlap(accumulated, chunk);
    let addition = &chunk[overlap..];
    if addition.is_empty() {
        return false;
    }
    accumulated.push_str(addition);
    true
}

/// Length in bytes of the longest prefix of `chunk` that is a suffix of
/// `accumulated`, respecting UTF-8 character boundaries.
fn longest_overlap(accumulated: &str, chunk: &str) -> usize {
    let max = accumulated.len().min(chunk.len());
    // Longest first: a longer restatement should be removed entirely rather
    // than leaving a partial duplicate behind.
    let mut candidate = max;
    while candidate > 0 {
        if chunk.is_char_boundary(candidate) {
            let tail_start = accumulated.len() - candidate;
            if accumulated.is_char_boundary(tail_start)
                && accumulated[tail_start..] == chunk[..candidate]
            {
                return candidate;
            }
        }
        candidate -= 1;
    }
    0
}

/// Stream one specification stage to completion within the continuation bound.
///
/// `Ok` means the provider reported a real end of turn. Every `Err` variant
/// means the caller must not publish an artifact.
pub async fn complete_stage_text(
    provider: Arc<dyn LlmProvider>,
    prompt: String,
    cancel: &CancellationToken,
) -> Result<CompletedText, SpecOutputError> {
    complete_stage_text_with_bound(provider, prompt, cancel, MAX_SPEC_CONTINUATIONS).await
}

/// Same as [`complete_stage_text`] with an injectable bound for tests.
pub async fn complete_stage_text_with_bound(
    provider: Arc<dyn LlmProvider>,
    prompt: String,
    cancel: &CancellationToken,
    max_continuations: u32,
) -> Result<CompletedText, SpecOutputError> {
    complete_stage_text_with_limits(
        provider,
        prompt,
        cancel,
        max_continuations,
        STREAM_IDLE_TIMEOUT,
    )
    .await
}

/// Same as [`complete_stage_text`] with injectable bound and idle timeout.
pub async fn complete_stage_text_with_limits(
    provider: Arc<dyn LlmProvider>,
    prompt: String,
    cancel: &CancellationToken,
    max_continuations: u32,
    idle_timeout: std::time::Duration,
) -> Result<CompletedText, SpecOutputError> {
    let mut messages = vec![agent_types::Message {
        role: agent_types::Role::User,
        content: vec![agent_types::ContentBlock::Text(prompt)],
        token_estimate: 0,
    }];

    let mut accumulated = String::new();
    let mut continuations = 0u32;

    loop {
        if cancel.is_cancelled() {
            return Err(SpecOutputError::Cancelled);
        }

        let mut receiver = provider
            .stream(&messages, &[], cancel)
            .await
            .map_err(|error| SpecOutputError::Provider(error.to_string()))?;

        let mut chunk = String::new();
        let mut stop_reason: Option<StopReason> = None;

        loop {
            // A stage must not hang on a silent provider, so every event wait is
            // bounded. Cancellation is observed concurrently so Ctrl-C does not
            // have to wait out the idle bound.
            let event = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(SpecOutputError::Cancelled),
                received = tokio::time::timeout(idle_timeout, receiver.recv()) => match received {
                    Ok(event) => event,
                    Err(_) => {
                        let _ = append_with_overlap_dedup(&mut accumulated, &chunk);
                        return Err(SpecOutputError::IdleTimeout {
                            bytes: accumulated.len(),
                            seconds: idle_timeout.as_secs(),
                        });
                    }
                },
            };

            let Some(event) = event else { break };
            match event {
                SseEvent::Delta(delta) => chunk.push_str(&delta),
                SseEvent::Stop { reason } => {
                    stop_reason = Some(reason);
                    break;
                }
                SseEvent::Error(error) => return Err(SpecOutputError::Provider(error)),
                SseEvent::Cancelled => return Err(SpecOutputError::Cancelled),
                _ => {}
            }
        }

        // A stream that closes with no terminal marker produced truncated text.
        let Some(reason) = stop_reason else {
            if cancel.is_cancelled() {
                return Err(SpecOutputError::Cancelled);
            }
            let _ = append_with_overlap_dedup(&mut accumulated, &chunk);
            return Err(SpecOutputError::UnterminatedStream {
                bytes: accumulated.len(),
            });
        };

        let progressed = append_with_overlap_dedup(&mut accumulated, &chunk);

        if reason == StopReason::EndTurn {
            return Ok(CompletedText {
                text: accumulated,
                stop_reason: reason,
                continuations,
            });
        }

        // A specification stage is offered no tools, so a `ToolUse` stop means
        // the provider ended the turn expecting a tool result that will never
        // arrive. The text is therefore not a finished artifact.
        if reason == StopReason::ToolUse {
            return Err(SpecOutputError::UnterminatedStream {
                bytes: accumulated.len(),
            });
        }

        // Truncated. Only continue while the model is still adding text and the
        // bound allows another attempt.
        if !progressed {
            return Err(SpecOutputError::NoProgress {
                continuations,
                bytes: accumulated.len(),
            });
        }
        if continuations >= max_continuations {
            return Err(SpecOutputError::BoundExhausted {
                continuations,
                bytes: accumulated.len(),
            });
        }

        continuations += 1;
        messages.push(agent_types::Message {
            role: agent_types::Role::Assistant,
            content: vec![agent_types::ContentBlock::Text(chunk)],
            token_estimate: 0,
        });
        messages.push(agent_types::Message {
            role: agent_types::Role::User,
            content: vec![agent_types::ContentBlock::Text(CONTINUATION_PROMPT.into())],
            token_estimate: 0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_types::{Message, Result as AgentResult, ToolSchema};
    use std::sync::Mutex;
    use tokio::sync::mpsc;

    #[test]
    fn every_incomplete_outcome_reports_the_same_marker_literal() {
        // **Validates: Requirements 2.40**
        // Consumers detect incompleteness by searching for `[incomplete]`, so a
        // variant that formats the word differently would silently look complete.
        for error in [
            SpecOutputError::BoundExhausted {
                continuations: 5,
                bytes: 10,
            },
            SpecOutputError::NoProgress {
                continuations: 2,
                bytes: 10,
            },
            SpecOutputError::UnterminatedStream { bytes: 10 },
            SpecOutputError::IdleTimeout {
                bytes: 10,
                seconds: 120,
            },
            SpecOutputError::Cancelled,
        ] {
            let rendered = error.to_string();
            assert!(
                rendered.contains(agent_types::INCOMPLETE_OUTPUT_MARKER),
                "variant must carry the exact marker: {rendered}"
            );
        }

        // A provider failure is a distinct error, not a truncation, so it is
        // deliberately not marked incomplete.
        assert!(!SpecOutputError::Provider("boom".into())
            .to_string()
            .contains(agent_types::INCOMPLETE_OUTPUT_MARKER));
    }

    #[test]
    fn overlap_dedup_removes_restated_prefix() {
        let mut accumulated = String::from("alpha beta");
        assert!(append_with_overlap_dedup(&mut accumulated, " beta gamma"));
        assert_eq!(accumulated, "alpha beta gamma");
    }

    #[test]
    fn overlap_dedup_reports_no_progress_for_exact_restatement() {
        let mut accumulated = String::from("same fragment");
        assert!(!append_with_overlap_dedup(
            &mut accumulated,
            "same fragment"
        ));
        assert_eq!(accumulated, "same fragment");
    }

    #[test]
    fn overlap_dedup_reports_no_progress_for_empty_chunk() {
        let mut accumulated = String::from("text");
        assert!(!append_with_overlap_dedup(&mut accumulated, ""));
        assert_eq!(accumulated, "text");
    }

    #[test]
    fn overlap_dedup_never_splits_a_multibyte_scalar() {
        let mut accumulated = String::from("भारत😀");
        assert!(append_with_overlap_dedup(&mut accumulated, "😀 ok"));
        assert_eq!(accumulated, "भारत😀 ok");
        assert!(accumulated.is_char_boundary(accumulated.len()));
    }

    #[test]
    fn overlap_dedup_appends_fully_when_there_is_no_overlap() {
        let mut accumulated = String::from("0");
        assert!(append_with_overlap_dedup(&mut accumulated, "1"));
        assert_eq!(accumulated, "01");
    }

    struct ScriptedProvider {
        scripts: Mutex<std::collections::VecDeque<(String, Option<StopReason>)>>,
        calls: Mutex<usize>,
    }

    impl ScriptedProvider {
        fn new(scripts: Vec<(&str, Option<StopReason>)>) -> Self {
            Self {
                scripts: Mutex::new(
                    scripts
                        .into_iter()
                        .map(|(text, reason)| (text.to_string(), reason))
                        .collect(),
                ),
                calls: Mutex::new(0),
            }
        }

        fn calls(&self) -> usize {
            *self.calls.lock().unwrap()
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for ScriptedProvider {
        async fn stream(
            &self,
            _messages: &[Message],
            _tools: &[ToolSchema],
            _cancel: &CancellationToken,
        ) -> AgentResult<mpsc::Receiver<SseEvent>> {
            *self.calls.lock().unwrap() += 1;
            let script = self.scripts.lock().unwrap().pop_front();
            let (text, reason) = script.unwrap_or_else(|| (String::new(), None));
            let (sender, receiver) = mpsc::channel(4);
            tokio::spawn(async move {
                if !text.is_empty() {
                    let _ = sender.send(SseEvent::Delta(text)).await;
                }
                if let Some(reason) = reason {
                    let _ = sender.send(SseEvent::Stop { reason }).await;
                }
            });
            Ok(receiver)
        }
    }

    async fn run(
        scripts: Vec<(&str, Option<StopReason>)>,
    ) -> std::result::Result<CompletedText, SpecOutputError> {
        let provider = Arc::new(ScriptedProvider::new(scripts));
        complete_stage_text_with_bound(
            provider,
            "prompt".into(),
            &CancellationToken::new(),
            MAX_SPEC_CONTINUATIONS,
        )
        .await
    }

    #[tokio::test]
    async fn end_turn_completes_without_continuation() {
        let completed = run(vec![("full answer", Some(StopReason::EndTurn))])
            .await
            .unwrap();
        assert_eq!(completed.text, "full answer");
        assert_eq!(completed.stop_reason, StopReason::EndTurn);
        assert_eq!(completed.continuations, 0);
    }

    #[tokio::test]
    async fn max_tokens_is_continued_and_deduplicated_until_complete() {
        let completed = run(vec![
            ("part one", Some(StopReason::MaxTokens)),
            (" part two", Some(StopReason::EndTurn)),
        ])
        .await
        .unwrap();
        assert_eq!(completed.text, "part one part two");
        assert_eq!(completed.continuations, 1);
    }

    #[tokio::test]
    async fn repeated_chunk_without_progress_is_incomplete() {
        let error = run(vec![
            ("same fragment", Some(StopReason::MaxTokens)),
            ("same fragment", Some(StopReason::MaxTokens)),
        ])
        .await
        .unwrap_err();
        assert!(matches!(error, SpecOutputError::NoProgress { .. }));
        assert!(error.to_string().contains("[incomplete]"));
    }

    #[tokio::test]
    async fn single_truncated_response_is_never_published_as_complete() {
        let error = run(vec![("PARTIAL_ONLY", Some(StopReason::MaxTokens))])
            .await
            .unwrap_err();
        // The second call yields no script, so the stream ends unterminated.
        assert!(error.to_string().contains("[incomplete]"));
    }

    #[tokio::test]
    async fn bound_exhaustion_is_reported_incomplete() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            ("a", Some(StopReason::MaxTokens)),
            ("b", Some(StopReason::MaxTokens)),
            ("c", Some(StopReason::MaxTokens)),
        ]));
        let error = complete_stage_text_with_bound(
            provider.clone(),
            "prompt".into(),
            &CancellationToken::new(),
            2,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error,
            SpecOutputError::BoundExhausted {
                continuations: 2,
                ..
            }
        ));
        assert_eq!(provider.calls(), 3, "the bound must cap provider calls");
    }

    #[tokio::test]
    async fn tool_use_stop_on_a_tool_free_stage_is_incomplete() {
        // Stages are offered no tools, so a ToolUse stop awaits a result that
        // will never arrive and must not be published.
        let error = run(vec![("calling a tool", Some(StopReason::ToolUse))])
            .await
            .unwrap_err();
        assert!(matches!(error, SpecOutputError::UnterminatedStream { .. }));
        assert!(error.to_string().contains("[incomplete]"));
    }

    #[tokio::test]
    async fn unterminated_stream_is_incomplete() {
        let error = run(vec![("no terminal event", None)]).await.unwrap_err();
        assert!(matches!(error, SpecOutputError::UnterminatedStream { .. }));
    }

    struct SilentProvider;

    #[async_trait::async_trait]
    impl LlmProvider for SilentProvider {
        async fn stream(
            &self,
            _messages: &[Message],
            _tools: &[ToolSchema],
            _cancel: &CancellationToken,
        ) -> AgentResult<mpsc::Receiver<SseEvent>> {
            // Keep the sender alive without ever emitting an event.
            let (sender, receiver) = mpsc::channel(1);
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                drop(sender);
            });
            Ok(receiver)
        }
    }

    #[tokio::test]
    async fn silent_provider_hits_the_idle_bound_instead_of_hanging() {
        let error = complete_stage_text_with_limits(
            Arc::new(SilentProvider),
            "prompt".into(),
            &CancellationToken::new(),
            MAX_SPEC_CONTINUATIONS,
            std::time::Duration::from_millis(30),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, SpecOutputError::IdleTimeout { .. }));
        assert!(error.to_string().contains("[incomplete]"));
    }

    #[tokio::test]
    async fn cancellation_during_a_silent_stream_is_observed_immediately() {
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            trigger.cancel();
        });
        let error = complete_stage_text_with_limits(
            Arc::new(SilentProvider),
            "prompt".into(),
            &cancel,
            MAX_SPEC_CONTINUATIONS,
            std::time::Duration::from_secs(3600),
        )
        .await
        .unwrap_err();
        assert_eq!(error, SpecOutputError::Cancelled);
    }

    #[tokio::test]
    async fn cancellation_is_reported_before_streaming() {
        let provider = Arc::new(ScriptedProvider::new(vec![(
            "unused",
            Some(StopReason::EndTurn),
        )]));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = complete_stage_text_with_bound(provider.clone(), "p".into(), &cancel, 5)
            .await
            .unwrap_err();
        assert_eq!(error, SpecOutputError::Cancelled);
        assert_eq!(provider.calls(), 0);
    }
}
