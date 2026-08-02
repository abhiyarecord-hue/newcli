//! Turn orchestrator: the agentic loop.
//!
//! `Orchestrator::run_turn` implements plan.md section 2.1:
//! 1. Activate skills
//! 2. Compact if needed
//! 3. Stream LLM
//! 4. On ToolUse → hooks → dispatch → append ToolResult → re-invoke LLM
//! 5. Loop until the provider emits no tool calls, with a hard cap of 200 iterations

use std::sync::Arc;

use agent_types::{AgentError, AgentEvent, ApprovalProvider, ContentBlock, Message, Result, Role};
use compaction::{CompactionConfig, Compactor, ConversationSummary, ThresholdCompactor};
use harness::SkillRegistry;
use llm_client::{LlmProvider, SseEvent, StopReason};
use runtime_core::EventBus;
use tokio_util::sync::CancellationToken;

use crate::tools::ToolDispatcher;

const MAX_ITERATIONS: usize = 200;
/// How many times we allow the model to auto-continue after a MaxTokens stop
/// before giving up and returning partial output.
const MAX_CONTINUATIONS: usize = 5;

pub struct Orchestrator {
    provider: Arc<dyn LlmProvider>,
    dispatcher: Arc<ToolDispatcher>,
    skills: Arc<SkillRegistry>,
    compactor: ThresholdCompactor,
    compaction_config: CompactionConfig,
    event_bus: EventBus,
    history: Vec<Message>,
    /// Monotonic identity for each entry of `history`, in the same order.
    ///
    /// Compaction truncates `history` from the front, so a positional index is
    /// not a stable cursor. IDs are never reused, so a consumer that recorded
    /// "persisted through ID n" still resolves correctly after any compaction.
    history_ids: Vec<u64>,
    next_message_id: u64,
    /// Monotonic turn counter used to attribute tool executions.
    turn_id: u64,
    /// Workspace mutations actually committed during the most recent turn.
    ///
    /// Reset at the start of every turn so a caller can never credit this run
    /// with work performed earlier.
    committed_mutations_this_turn: usize,
    /// Structured summary of compacted conversation. Sent as untrusted data,
    /// never merged into the policy system prompt.
    compacted_summary: ConversationSummary,
    cancel: CancellationToken,
    approval_provider: Option<Arc<dyn ApprovalProvider>>,
    lang: agent_types::LanguageMode,
    project_root: std::path::PathBuf,
    base_system_prompt: String,
}

impl Orchestrator {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        dispatcher: Arc<ToolDispatcher>,
        skills: Arc<SkillRegistry>,
        event_bus: EventBus,
        cancel: CancellationToken,
        lang: agent_types::LanguageMode,
    ) -> Self {
        Self {
            provider,
            dispatcher,
            skills,
            compactor: ThresholdCompactor,
            compaction_config: CompactionConfig::default(),
            event_bus,
            history: Vec::new(),
            history_ids: Vec::new(),
            next_message_id: 1,
            turn_id: 0,
            committed_mutations_this_turn: 0,
            compacted_summary: ConversationSummary::default(),
            cancel,
            approval_provider: None,
            lang,
            project_root: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            base_system_prompt: String::new(),
        }
    }

    /// Set the workspace root where tools operate (read/write files, run commands).
    pub fn with_project_root(mut self, root: std::path::PathBuf) -> Self {
        self.project_root = root;
        self
    }

    /// Supply the single async approval/input callback used by tools.
    pub fn with_approval_provider(mut self, provider: Arc<dyn ApprovalProvider>) -> Self {
        self.approval_provider = Some(provider);
        self
    }

    /// Set a base system prompt that is always prepended to the system message.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base_system_prompt = prompt.into();
        self
    }

    /// Swap the base system prompt at runtime (e.g. switching between RustySpec
    /// stages that must not write files and the Implement stage that must).
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.base_system_prompt = prompt.into();
    }

    /// Restore prior conversation history (e.g. loaded from disk).
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.history_ids.clear();
        self.history = history;
        // Identities continue from the current counter rather than restarting,
        // so a restored conversation can never hand out an ID that was already
        // observed by a consumer in this process.
        for _ in 0..self.history.len() {
            self.history_ids.push(self.next_message_id);
            self.next_message_id = self.next_message_id.saturating_add(1);
        }
        self
    }

    /// Append one message, assigning it the next monotonic ID.
    fn push_message(&mut self, message: Message) {
        self.history.push(message);
        self.history_ids.push(self.next_message_id);
        self.next_message_id = self.next_message_id.saturating_add(1);
    }

    /// Remove the most recent message, keeping identities aligned.
    ///
    /// The consumed ID is not returned to the counter: identities are never
    /// reused, so a rolled-back message cannot be confused with a later one.
    fn pop_message(&mut self) -> Option<Message> {
        self.history_ids.pop();
        self.history.pop()
    }

    /// Messages appended after `after_id`, by monotonic message ID.
    ///
    /// This is a durable cursor: compaction dropping earlier messages never
    /// shifts the meaning of `after_id`. Pass `0` to receive everything the
    /// orchestrator currently retains.
    pub fn history_since(&self, after_id: usize) -> &[Message] {
        let after_id = after_id as u64;
        // IDs are assigned in ascending order, so the first entry above the
        // cursor bounds the unpersisted tail.
        let start = self.history_ids.partition_point(|id| *id <= after_id);
        &self.history[start..]
    }

    /// Highest message ID assigned so far.
    ///
    /// A caller persists this after a successful write and passes it back to
    /// [`history_since`](Self::history_since) on the next turn.
    pub fn last_message_id(&self) -> usize {
        self.next_message_id.saturating_sub(1) as usize
    }

    /// Number of messages currently retained in the in-memory history.
    ///
    /// Compaction can reduce this between turns, so callers must never assume a
    /// previously stored length still refers to the same boundary. Use
    /// [`last_message_id`](Self::last_message_id) for durable cursors.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Run one turn of the agent loop. Lifecycle events are balanced even when
    /// provider/tool processing returns an error.
    pub async fn run_turn(&mut self, user_msg: String) -> Result<String> {
        // A new turn owns its own mutation evidence. Resetting here means a
        // caller judging this turn can never see a mutation from an earlier one.
        self.turn_id = self.turn_id.saturating_add(1);
        self.committed_mutations_this_turn = 0;
        self.event_bus.emit(AgentEvent::TurnStarted);
        let result = self.run_turn_inner(user_msg).await;
        self.event_bus.emit(AgentEvent::TurnEnded);
        result
    }

    /// Turn currently being executed, or the last one that ran.
    pub fn turn_id(&self) -> u64 {
        self.turn_id
    }

    /// Workspace mutations committed during the most recent turn.
    ///
    /// A stage that must prove it did real work reads this instead of scanning
    /// the filesystem, which cannot tell a pre-existing file from a new one.
    pub fn committed_mutations_this_turn(&self) -> usize {
        self.committed_mutations_this_turn
    }

    async fn run_turn_inner(&mut self, user_msg: String) -> Result<String> {
        // 0. Recover a summary carried in a restored history.
        self.adopt_summary_from_history();

        // 1. Append user message.
        self.push_message(Message {
            role: Role::User,
            content: vec![ContentBlock::Text(user_msg.clone())],
            token_estimate: 0,
        });

        // 2. Activate skills and rebuild the persistent system context.
        let skill_fragments = self.skills.activate(&user_msg, self.lang);
        let mut system_text = self.base_system_prompt.clone();
        if !system_text.is_empty() {
            system_text.push('\n');
        }
        // The compacted summary is deliberately NOT added here: recovered
        // conversation text must never gain system-policy priority. It is sent
        // below as an explicitly delimited untrusted conversation message.
        for fragment in skill_fragments {
            system_text.push_str(fragment);
            system_text.push('\n');
        }

        // 3. Compact if needed. Persist every generated summary so it remains
        // available on later turns even when the retained history is under budget.
        let (summary, retained) = self
            .compactor
            .compact(&self.compaction_config, &self.history);
        // The compactor retains a suffix, so the dropped count maps the retained
        // messages back onto their existing identities. Reassigning IDs here
        // would defeat the cursor: a consumer's stored ID must keep pointing at
        // the same message across compactions.
        let dropped = self.history.len().saturating_sub(retained.len());
        let retained_ids = self
            .history_ids
            .split_off(dropped.min(self.history_ids.len()));
        // Always adopt the retained set: a legitimately empty summary (for
        // example a dropped prefix of only system/tool text) must still shrink
        // the context, otherwise compaction stops trimming entirely.
        let (mut retained, mut retained_ids) = strip_summary_blocks(retained, retained_ids);
        if !summary.is_empty() {
            self.compacted_summary.merge(&summary);
            // Re-apply the budget to the accumulated summary so repeated
            // compactions cannot grow the request without bound.
            self.compacted_summary
                .bound_to_tokens(self.compaction_config.summary_target_tokens);
        }
        // Carry the summary as conversation data inside the history itself, so
        // it survives a restart that restores only `history()`.
        if attach_summary_block(&self.compacted_summary, &mut retained) {
            // The rendered summary block is synthetic conversation data that is
            // rebuilt every turn, not an appended conversation message. ID 0 is
            // reserved for it: real identities start at 1, so a leading 0 keeps
            // the list ascending while never being reported as new work by
            // `history_since`.
            retained_ids.insert(0, 0);
        }
        debug_assert_eq!(retained.len(), retained_ids.len());
        self.history = retained;
        self.history_ids = retained_ids;

        // Prepend the policy system message. The summary is never part of it.
        let mut messages_for_llm: Vec<Message> = Vec::new();
        if !system_text.is_empty() {
            messages_for_llm.push(Message {
                role: Role::System,
                content: vec![ContentBlock::Text(system_text)],
                token_estimate: 0,
            });
        }
        messages_for_llm.extend(self.history.clone());

        let tools = self.dispatcher.schemas();

        // 4. Stream + tool loop.
        let mut iterations = 0;
        let mut final_text = String::new();
        let mut current_messages = messages_for_llm;
        let mut last_text = String::new(); // repetition detection
        let mut continuations = 0u32; // MaxTokens auto-continue counter

        loop {
            iterations += 1;
            if iterations > MAX_ITERATIONS {
                // Cap reached — return whatever we have rather than losing all work.
                if final_text.is_empty() {
                    final_text = "[iteration cap reached — task too large for single turn. Say 'continue' to resume.]".to_string();
                } else {
                    final_text.push_str("\n\n[iteration cap reached. Say 'continue' to resume.]");
                }
                break;
            }

            self.event_bus.emit(AgentEvent::ApiCallStarted);
            let mut rx = self
                .provider
                .stream(&current_messages, &tools, &self.cancel)
                .await?;

            let mut text_accum = String::new();
            let mut tool_uses: Vec<(String, String, serde_json::Value, Option<serde_json::Value>)> =
                Vec::new();
            let mut stop_reason = StopReason::EndTurn;
            // A stream that closes without any terminal marker produced a
            // truncated response. Committing it as a complete turn would report
            // partial output as finished, so the candidate is discarded instead.
            let mut saw_terminal_event = false;
            let mut saw_cancel_event = false;
            let mut stream_error: Option<String> = None;

            while let Some(event) = rx.recv().await {
                match event {
                    SseEvent::Delta(d) => text_accum.push_str(&d),
                    SseEvent::Thinking(_thought) => {
                        // Thought summaries are informational — emit for UI display
                        // but don't include in the conversation history.
                        self.event_bus.emit(AgentEvent::Thinking { text: _thought });
                    }
                    SseEvent::ToolUse {
                        id,
                        name,
                        input,
                        provider_metadata,
                    } => {
                        self.event_bus
                            .emit(AgentEvent::ToolInvoked { name: name.clone() });
                        tool_uses.push((id, name, input, provider_metadata));
                    }
                    SseEvent::Stop { reason } => {
                        // Keep draining until the provider closes the channel;
                        // usage metadata is often sent after the stop marker.
                        stop_reason = reason;
                        saw_terminal_event = true;
                    }
                    SseEvent::Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    } => {
                        self.event_bus.emit(AgentEvent::TokenUsage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                        });
                    }
                    SseEvent::Error(error) => {
                        // Stop draining but fall through to the shared
                        // incomplete-candidate handling below, so tool requests
                        // accepted before the error are still committed with
                        // explicit results instead of vanishing.
                        stream_error = Some(error);
                        break;
                    }
                    SseEvent::Cancelled => {
                        // Stop draining but fall through to the shared
                        // incomplete-candidate handling below, so tool requests
                        // accepted before cancellation are still committed with
                        // explicit cancellation results instead of vanishing.
                        saw_cancel_event = true;
                        break;
                    }
                }
            }

            // The stream closed. Treat it as incomplete when the token is
            // cancelled or when no terminal marker ever arrived: a provider that
            // only closes its channel leaves the buffered candidate holding a
            // truncated response.
            //
            // Partial assistant *text* must never be committed as a complete
            // turn. Tool requests that did arrive are different: dropping them
            // silently would hide accepted work, so they are committed together
            // with explicit cancellation results, keeping the transaction
            // complete for any later restore.
            let cancelled = saw_cancel_event || self.cancel.is_cancelled();
            let incomplete = cancelled || stream_error.is_some() || !saw_terminal_event;
            if incomplete {
                let reason = if cancelled {
                    "not dispatched: operation cancelled"
                } else {
                    "not dispatched: provider stream ended before completion"
                };
                if tool_uses.is_empty() {
                    // Nothing was accepted, so the trailing user message that
                    // opened this exchange has no answer. Leaving it would break
                    // role alternation on the next request, so it is rolled back
                    // together with the discarded candidate.
                    if matches!(self.history.last().map(|m| &m.role), Some(Role::User)) {
                        self.pop_message();
                    }
                } else {
                    // Accepted tool requests are committed with explicit results
                    // so the transaction stays complete for any later restore.
                    let assistant_content: Vec<ContentBlock> = tool_uses
                        .iter()
                        .map(
                            |(id, name, input, provider_metadata)| ContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                                provider_metadata: provider_metadata.clone(),
                            },
                        )
                        .collect();
                    self.push_message(Message {
                        role: Role::Assistant,
                        content: assistant_content,
                        token_estimate: 0,
                    });
                    self.push_message(Message {
                        role: Role::Tool,
                        content: tool_uses
                            .into_iter()
                            .map(|(id, _, _, _)| ContentBlock::ToolResult {
                                tool_use_id: id,
                                output: reason.into(),
                                is_error: true,
                            })
                            .collect(),
                        token_estimate: 0,
                    });
                }

                return if cancelled {
                    Err(AgentError::Cancelled)
                } else if let Some(error) = stream_error {
                    Err(AgentError::Llm(error))
                } else {
                    Err(AgentError::Llm(
                        "provider stream closed without a terminal event; response is incomplete"
                            .into(),
                    ))
                };
            }

            // Build assistant message.
            let mut assistant_content: Vec<ContentBlock> = Vec::new();
            if !text_accum.is_empty() {
                assistant_content.push(ContentBlock::Text(text_accum.clone()));
            }
            for (id, name, input, provider_metadata) in &tool_uses {
                assistant_content.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    provider_metadata: provider_metadata.clone(),
                });
            }

            // Repetition prevention runs before the candidate is committed.
            // Breaking after committing an assistant message that carries tool
            // calls would leave an unresolved transaction in history, which a
            // provider rejects on the next turn. When the candidate repeats and
            // carries tool calls, discard the uncommitted candidate instead.
            //
            // However, repetition is only meaningful when no prior calls are
            // still unresolved. If the history already contains an unresolved
            // ToolUse (e.g. after a cancellation recovery), a repeated response
            // that includes tool calls must be dispatched, not suppressed, so
            // the protocol transaction can be completed.
            let repeated = !text_accum.is_empty() && text_accum == last_text;
            let has_unresolved_prior = !agent_types::unresolved_tool_uses(&self.history).is_empty();
            if repeated && !tool_uses.is_empty() && !has_unresolved_prior {
                final_text = text_accum;
                break;
            }
            last_text = text_accum.clone();

            let assistant_msg = Message {
                role: Role::Assistant,
                content: assistant_content,
                token_estimate: 0,
            };
            self.push_message(assistant_msg.clone());
            current_messages.push(assistant_msg);

            // Actual tool calls take precedence over a provider's inconsistent
            // end-turn reason. Stop only when there is nothing to dispatch.
            if tool_uses.is_empty() {
                // No unresolved calls: repeating text is a genuine loop and the
                // committed message is protocol-complete on its own.
                if repeated {
                    if continuations == 0 {
                        // Not a continuation chain: nothing has been
                        // accumulated yet, so the chunk is still needed.
                        final_text.push_str(&text_accum);
                    } else {
                        // The model restated its previous chunk verbatim instead
                        // of continuing, so the output stopped progressing and
                        // is not complete. Disclose that rather than returning
                        // truncated text as a finished answer. The chunk itself
                        // is not appended again, which would duplicate it.
                        // One shared marker constant keeps every surface's
                        // incompleteness disclosure detectable by one search.
                        final_text.push_str("\n\n");
                        final_text.push_str(agent_types::INCOMPLETE_OUTPUT_MARKER);
                        final_text.push_str(
                            " the model stopped making progress after a continuation, \
                             so this output is truncated",
                        );
                    }
                    break;
                }

                // MaxTokens with no tool calls means the model ran out of output
                // space mid-generation. Automatically re-invoke with a continue
                // instruction so large outputs are not truncated.
                if stop_reason == StopReason::MaxTokens
                    && (continuations as usize) < MAX_CONTINUATIONS
                {
                    continuations += 1;
                    // Append a synthetic user message asking the model to continue.
                    let continue_msg = Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text(
                            "Continue from where you left off. Do not repeat what you already wrote."
                                .to_string(),
                        )],
                        token_estimate: 0,
                    };
                    self.push_message(continue_msg.clone());
                    current_messages.push(continue_msg);
                    // Accumulate text across continuations.
                    final_text.push_str(&text_accum);
                    continue;
                }

                final_text.push_str(&text_accum);
                break;
            }

            // Dispatch tools and append results.
            let ctx = agent_types::ToolCtx {
                project_root: self.project_root.clone(),
                cancel: self.cancel.clone(),
                approval_provider: self.approval_provider.clone(),
            };

            let mut tool_results: Vec<ContentBlock> = Vec::new();
            let mut pending = tool_uses.into_iter();
            let mut policy_failure = None;
            let mut batch_cancelled = false;
            while let Some((id, name, input, _provider_metadata)) = pending.next() {
                if self.cancel.is_cancelled() {
                    tool_results.push(cancelled_tool_result(id));
                    for (remaining_id, _remaining_name, _input, _metadata) in pending {
                        tool_results.push(cancelled_tool_result(remaining_id));
                    }
                    batch_cancelled = true;
                    break;
                }

                let outcome = self
                    .dispatcher
                    .dispatch_for_turn(&name, input, &ctx, self.turn_id)
                    .await;
                if outcome.committed_mutation {
                    self.committed_mutations_this_turn += 1;
                }
                let must_stop = outcome.must_stop_dispatch();
                let failure_reason = outcome.policy_failure.clone();
                let mut result = outcome.block;
                if let ContentBlock::ToolResult {
                    ref mut tool_use_id,
                    ..
                } = result
                {
                    *tool_use_id = id;
                }
                self.event_bus
                    .emit(AgentEvent::ToolCompleted { name: name.clone() });
                tool_results.push(result);

                if must_stop {
                    let reason = failure_reason
                        .unwrap_or_else(|| "post-tool policy failed closed".to_string());
                    for (remaining_id, _remaining_name, _input, _metadata) in pending {
                        tool_results.push(ContentBlock::ToolResult {
                            tool_use_id: remaining_id,
                            output: "not dispatched: prior post-tool policy failure".into(),
                            is_error: true,
                        });
                    }
                    policy_failure = Some(reason);
                    break;
                }
            }

            let tool_msg = Message {
                role: Role::Tool,
                content: tool_results,
                token_estimate: 0,
            };
            self.push_message(tool_msg.clone());
            current_messages.push(tool_msg);

            if batch_cancelled {
                return Err(AgentError::Cancelled);
            }
            if let Some(reason) = policy_failure {
                return Err(AgentError::Tool {
                    name: "post_tool_policy".into(),
                    reason,
                });
            }
        }

        Ok(final_text)
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Rebuild the in-memory summary from a restored history.
    ///
    /// A restart that restores only `history()` still carries the rendered
    /// summary block, so the orchestrator recovers it instead of losing it.
    fn adopt_summary_from_history(&mut self) {
        if !self.compacted_summary.is_empty() {
            return;
        }
        for message in &self.history {
            for block in &message.content {
                if let ContentBlock::Text(text) = block {
                    if compaction::is_summary_block(text) {
                        self.compacted_summary = ConversationSummary::from_rendered_block(text);
                        return;
                    }
                }
            }
        }
    }

    /// Structured summary of conversation already compacted away.
    ///
    /// Exposed so a caller can persist it beside the conversation snapshot; it
    /// is conversation data, not policy.
    pub fn compacted_summary(&self) -> &ConversationSummary {
        &self.compacted_summary
    }

    /// Clear every in-memory conversation layer.
    ///
    /// `/clear` must not leave any layer that a later turn could repopulate
    /// from: the message list, the accumulated compacted summary, and any
    /// derived per-turn state are all reset together. Long-term memory files
    /// (`MEMORY.md`, `SOUL.md`, `HEARTBEAT.md`) are deliberately untouched —
    /// they are documented as separate durable state, not conversation history.
    ///
    /// Callers must clear durable storage as well and report success only after
    /// both this call and the storage removal succeed.
    pub fn clear_conversation(&mut self) {
        self.history.clear();
        self.history_ids.clear();
        // `next_message_id` deliberately keeps advancing. Restarting it would
        // reuse identities a consumer may already hold, so a stale cursor could
        // silently hide messages from the new conversation.
        self.compacted_summary = ConversationSummary::default();
    }

    /// Restore a previously persisted compacted summary.
    ///
    /// Restoration is consistent with generation: the summary re-enters the
    /// request as the same delimited untrusted block it was sent as before.
    pub fn set_compacted_summary(&mut self, summary: ConversationSummary) {
        self.compacted_summary = summary;
    }
}

/// Remove any previously rendered summary block from a message list.
///
/// Exactly one current block is carried at a time, so a stale block never
/// accumulates alongside a fresh one.
/// Returns the surviving messages together with their retained identities, so a
/// message dropped for becoming empty also drops its ID.
fn strip_summary_blocks(messages: Vec<Message>, ids: Vec<u64>) -> (Vec<Message>, Vec<u64>) {
    let mut stripped = Vec::with_capacity(messages.len());
    let mut stripped_ids = Vec::with_capacity(messages.len());
    // Zipping keeps every surviving message paired with its own identity. A
    // shorter ID list can only mean the caller passed unaligned input, and
    // dropping the unpaired tail is safer than inventing an ID that could break
    // the ascending order `history_since` relies on.
    for (mut message, id) in messages.into_iter().zip(ids) {
        message.content.retain(|block| match block {
            ContentBlock::Text(text) => !compaction::is_summary_block(text),
            _ => true,
        });
        if !message.content.is_empty() {
            stripped.push(message);
            stripped_ids.push(id);
        }
    }
    (stripped, stripped_ids)
}

/// Attach the rendered summary as untrusted conversation data.
///
/// The block joins the first retained user message when there is one, so no
/// extra same-role message is introduced for providers that require strict role
/// alternation. Otherwise it becomes its own leading user message.
/// Returns `true` when a new leading message was inserted, so the caller can
/// keep its parallel identity list aligned.
fn attach_summary_block(summary: &ConversationSummary, messages: &mut Vec<Message>) -> bool {
    let block = summary.render_untrusted_block();
    if block.is_empty() {
        return false;
    }

    if let Some(first) = messages.first_mut() {
        if matches!(first.role, Role::User) {
            first.content.insert(0, ContentBlock::Text(block));
            return false;
        }
    }
    messages.insert(
        0,
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text(block)],
            token_estimate: 0,
        },
    );
    true
}

fn cancelled_tool_result(tool_use_id: String) -> ContentBlock {
    ContentBlock::ToolResult {
        tool_use_id,
        output: "not dispatched: operation cancelled".into(),
        is_error: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_iterations_constant_is_200() {
        assert_eq!(MAX_ITERATIONS, 200);
    }

    /// Always answers with the same text and a `MaxTokens` stop, so the model
    /// never makes progress across continuations.
    struct RepeatingMaxTokensProvider {
        calls: std::sync::Mutex<usize>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for RepeatingMaxTokensProvider {
        async fn stream(
            &self,
            _messages: &[Message],
            _tools: &[agent_types::ToolSchema],
            _cancel: &CancellationToken,
        ) -> Result<tokio::sync::mpsc::Receiver<SseEvent>> {
            *self.calls.lock().unwrap() += 1;
            let (sender, receiver) = tokio::sync::mpsc::channel(4);
            tokio::spawn(async move {
                let _ = sender.send(SseEvent::Delta("same fragment".into())).await;
                let _ = sender
                    .send(SseEvent::Stop {
                        reason: StopReason::MaxTokens,
                    })
                    .await;
            });
            Ok(receiver)
        }
    }

    #[tokio::test]
    async fn stalled_continuation_chain_discloses_incompleteness_in_the_returned_text() {
        // **Validates: Requirements 2.40**
        // Callers detect truncation by searching for the marker. A stalled
        // continuation chain must never be returned as a finished answer, and
        // the repeated chunk must not be duplicated.
        let provider = Arc::new(RepeatingMaxTokensProvider {
            calls: std::sync::Mutex::new(0),
        });
        let dispatcher = Arc::new(ToolDispatcher::new(
            vec![],
            Arc::new(harness::HookEngine::new(vec![])),
        ));
        let skills = Arc::new(SkillRegistry::load(None).unwrap());
        let mut orchestrator = Orchestrator::new(
            provider.clone(),
            dispatcher,
            skills,
            EventBus::default(),
            CancellationToken::new(),
            agent_types::LanguageMode::En,
        );

        let answer = orchestrator
            .run_turn("produce artifact".into())
            .await
            .unwrap();

        assert!(
            answer.contains(agent_types::INCOMPLETE_OUTPUT_MARKER),
            "a stalled chain must disclose incompleteness: {answer:?}"
        );
        assert_eq!(
            answer.matches("same fragment").count(),
            1,
            "the repeated chunk must not be duplicated: {answer:?}"
        );
        assert_eq!(
            *provider.calls.lock().unwrap(),
            2,
            "the chain must stop as soon as progress stops"
        );
    }

    fn text(role: Role, value: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text(value.into())],
            token_estimate: 0,
        }
    }

    fn summary_with_task(task: &str) -> ConversationSummary {
        let mut summary = ConversationSummary::default();
        summary.open_tasks.push(task.to_string());
        summary
    }

    #[test]
    fn strip_summary_blocks_drops_ids_of_emptied_messages() {
        let block = summary_with_task("carried").render_untrusted_block();
        let messages = vec![
            text(Role::User, "keep-first"),
            text(Role::Assistant, &block),
            text(Role::User, "keep-last"),
        ];

        let (stripped, ids) = strip_summary_blocks(messages, vec![4, 5, 6]);

        assert_eq!(stripped.len(), 2);
        assert_eq!(
            ids,
            vec![4, 6],
            "the emptied message must drop its identity"
        );
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn attach_summary_block_reports_whether_it_inserted_a_message() {
        let summary = summary_with_task("carried");

        // A leading user message absorbs the block, so no identity is needed.
        let mut joined = vec![text(Role::User, "first")];
        assert!(!attach_summary_block(&summary, &mut joined));
        assert_eq!(joined.len(), 1);

        // A leading non-user message forces a new synthetic message.
        let mut inserted = vec![text(Role::Assistant, "first")];
        assert!(attach_summary_block(&summary, &mut inserted));
        assert_eq!(inserted.len(), 2);
        assert!(matches!(inserted[0].role, Role::User));

        // An empty summary is never attached.
        let mut untouched = vec![text(Role::Assistant, "first")];
        assert!(!attach_summary_block(
            &ConversationSummary::default(),
            &mut untouched
        ));
        assert_eq!(untouched.len(), 1);
    }
}
