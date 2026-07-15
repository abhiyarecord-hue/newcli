//! Turn orchestrator: the agentic loop.
//!
//! `Orchestrator::run_turn` implements plan.md section 2.1:
//! 1. Activate skills
//! 2. Compact if needed
//! 3. Stream LLM
//! 4. On ToolUse → hooks → dispatch → append ToolResult → re-invoke LLM
//! 5. Loop until the provider emits no tool calls, with a hard cap of 200 iterations

use std::sync::Arc;

use agent_types::{AgentError, AgentEvent, ContentBlock, Message, Result, Role};
use compaction::{CompactionConfig, Compactor, ThresholdCompactor};
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
    compacted_summary: String,
    cancel: CancellationToken,
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
            compacted_summary: String::new(),
            cancel,
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
        self.history = history;
        self
    }

    /// Get new messages added since a given index (for incremental persistence).
    pub fn history_since(&self, start: usize) -> &[Message] {
        if start >= self.history.len() {
            &[]
        } else {
            &self.history[start..]
        }
    }

    /// Run one turn of the agent loop. Lifecycle events are balanced even when
    /// provider/tool processing returns an error.
    pub async fn run_turn(&mut self, user_msg: String) -> Result<String> {
        self.event_bus.emit(AgentEvent::TurnStarted);
        let result = self.run_turn_inner(user_msg).await;
        self.event_bus.emit(AgentEvent::TurnEnded);
        result
    }

    async fn run_turn_inner(&mut self, user_msg: String) -> Result<String> {
        // 1. Append user message.
        self.history.push(Message {
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
        if !self.compacted_summary.is_empty() {
            system_text.push_str(&self.compacted_summary);
            system_text.push('\n');
        }
        for fragment in skill_fragments {
            system_text.push_str(fragment);
            system_text.push('\n');
        }

        // 3. Compact if needed. Persist every generated summary so it remains
        // available on later turns even when the retained history is under budget.
        let (summary, retained) = self
            .compactor
            .compact(&self.compaction_config, &self.history);
        if !summary.is_empty() {
            if !self.compacted_summary.is_empty() {
                self.compacted_summary.push('\n');
            }
            self.compacted_summary.push_str(&summary);
            system_text.push_str(&summary);
            system_text.push('\n');
            self.history = retained;
        }

        // Prepend system message if we have content.
        let messages_for_llm = if system_text.is_empty() {
            self.history.clone()
        } else {
            let mut msgs = vec![Message {
                role: Role::System,
                content: vec![ContentBlock::Text(system_text)],
                token_estimate: 0,
            }];
            msgs.extend(self.history.clone());
            msgs
        };

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
            let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut stop_reason = StopReason::EndTurn;

            while let Some(event) = rx.recv().await {
                match event {
                    SseEvent::Delta(d) => text_accum.push_str(&d),
                    SseEvent::Thinking(_thought) => {
                        // Thought summaries are informational — emit for UI display
                        // but don't include in the conversation history.
                        self.event_bus.emit(AgentEvent::Thinking {
                            text: _thought,
                        });
                    }
                    SseEvent::ToolUse { id, name, input } => {
                        self.event_bus.emit(AgentEvent::ToolInvoked {
                            name: name.clone(),
                        });
                        tool_uses.push((id, name, input));
                    }
                    SseEvent::Stop { reason } => {
                        // Keep draining until the provider closes the channel;
                        // usage metadata is often sent after the stop marker.
                        stop_reason = reason;
                    }
                    SseEvent::Usage { prompt_tokens, completion_tokens, total_tokens } => {
                        self.event_bus.emit(AgentEvent::TokenUsage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                        });
                    }
                    SseEvent::Error(e) => return Err(AgentError::Llm(e)),
                }
            }

            // Build assistant message.
            let mut assistant_content: Vec<ContentBlock> = Vec::new();
            if !text_accum.is_empty() {
                assistant_content.push(ContentBlock::Text(text_accum.clone()));
            }
            for (id, name, input) in &tool_uses {
                assistant_content.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }

            let assistant_msg = Message {
                role: Role::Assistant,
                content: assistant_content,
                token_estimate: 0,
            };
            self.history.push(assistant_msg.clone());
            current_messages.push(assistant_msg);

            // Actual tool calls take precedence over a provider's inconsistent
            // end-turn reason. Stop only when there is nothing to dispatch.
            if tool_uses.is_empty() {
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
                    self.history.push(continue_msg.clone());
                    current_messages.push(continue_msg);
                    // Accumulate text across continuations.
                    final_text.push_str(&text_accum);
                    continue;
                }

                final_text.push_str(&text_accum);
                break;
            }

            // Repetition loop detection: if model outputs exact same text twice
            // in a row, it's stuck in a loop. Break to avoid wasting tokens.
            if !text_accum.is_empty() && text_accum == last_text {
                final_text = text_accum;
                break;
            }
            last_text = text_accum.clone();

            // Dispatch tools and append results.
            let ctx = agent_types::ToolCtx {
                project_root: self.project_root.clone(),
                cancel: self.cancel.clone(),
            };

            let mut tool_results: Vec<ContentBlock> = Vec::new();
            for (id, name, input) in tool_uses {
                let mut result = self.dispatcher.dispatch(&name, input, &ctx).await;
                // Set the tool_use_id on the result.
                if let ContentBlock::ToolResult {
                    ref mut tool_use_id,
                    ..
                } = result
                {
                    *tool_use_id = id;
                }
                self.event_bus.emit(AgentEvent::ToolCompleted {
                    name: name.clone(),
                });
                tool_results.push(result);
            }

            let tool_msg = Message {
                role: Role::Tool,
                content: tool_results,
                token_estimate: 0,
            };
            self.history.push(tool_msg.clone());
            current_messages.push(tool_msg);
        }

        Ok(final_text)
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_iterations_constant_is_200() {
        assert_eq!(MAX_ITERATIONS, 200);
    }
}
