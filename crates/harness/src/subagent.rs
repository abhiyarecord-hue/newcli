//! Sub-agent spawner: bounded mini-loop exploration agents.
//!
//! A sub-agent gets its own fresh history, a read-only tool set, max 10 turns,
//! and returns only a summary string (≤ 2000 chars). No sub-agent spawning
//! from within a sub-agent (depth limit 1). Uses TaskScope + Semaphore for
//! max concurrency control.

use std::sync::Arc;

use agent_types::{AgentError, ContentBlock, Message, Result, Role, ToolSchema};
use llm_client::{LlmProvider, SseEvent};
use runtime_core::TaskScope;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

const MAX_TURNS: usize = 10;
const MAX_SUMMARY_CHARS: usize = 2000;

/// A sub-agent handles a single exploration task.
pub struct SubAgent {
    provider: Arc<dyn LlmProvider>,
    tools: Vec<ToolSchema>,
}

impl SubAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, tools: Vec<ToolSchema>) -> Self {
        Self { provider, tools }
    }

    /// Run the sub-agent mini-loop and return a summary string.
    pub async fn run(&self, task: &str, cancel: &CancellationToken) -> Result<String> {
        let mut history: Vec<Message> = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text(task.to_string())],
            token_estimate: 0,
        }];

        for _turn in 0..MAX_TURNS {
            let mut rx = self
                .provider
                .stream(&history, &self.tools, cancel)
                .await?;

            let mut text_accum = String::new();
            let mut stop = false;

            while let Some(event) = rx.recv().await {
                match event {
                    SseEvent::Delta(d) => text_accum.push_str(&d),
                    SseEvent::Stop { .. } => {
                        stop = true;
                        break;
                    }
                    SseEvent::ToolUse { .. } => {
                        // Sub-agents are read-only for now; skip tool use.
                        stop = true;
                        break;
                    }
                    SseEvent::Usage { .. } => {} // informational only
                    SseEvent::Thinking(_) => {} // thought summaries ignored in sub-agents
                    SseEvent::Error(e) => {
                        return Err(AgentError::Llm(e));
                    }
                }
            }

            history.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(text_accum.clone())],
                token_estimate: 0,
            });

            if stop {
                return Ok(truncate_summary(&text_accum));
            }
        }

        // Hit max turns — return whatever we have.
        let last_text = history
            .iter()
            .rev()
            .find_map(|m| {
                if matches!(m.role, Role::Assistant) {
                    m.content.iter().find_map(|b| match b {
                        ContentBlock::Text(t) => Some(t.clone()),
                        _ => None,
                    })
                } else {
                    None
                }
            })
            .unwrap_or_default();
        Ok(truncate_summary(&last_text))
    }
}

/// Run multiple exploration tasks in parallel with bounded concurrency.
pub struct SubAgentPool {
    provider: Arc<dyn LlmProvider>,
    tools: Vec<ToolSchema>,
    max_concurrent: usize,
}

impl SubAgentPool {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        tools: Vec<ToolSchema>,
        max_concurrent: usize,
    ) -> Self {
        Self {
            provider,
            tools,
            max_concurrent,
        }
    }

    /// Run all tasks, return summaries. Parent cancel kills all sub-agents.
    pub async fn run_parallel(
        &self,
        tasks: Vec<String>,
        cancel: &CancellationToken,
    ) -> Result<Vec<String>> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));
        let mut scope = TaskScope::with_token(cancel.child_token());
        let results: Arc<tokio::sync::Mutex<Vec<(usize, String)>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::new()));

        for (idx, task) in tasks.into_iter().enumerate() {
            let sem = semaphore.clone();
            let provider = self.provider.clone();
            let tools = self.tools.clone();
            let child_cancel = cancel.child_token();
            let results_clone = results.clone();

            scope.spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|_| AgentError::Cancelled)?;
                let agent = SubAgent::new(provider, tools);
                let summary = agent.run(&task, &child_cancel).await?;
                results_clone.lock().await.push((idx, summary));
                Ok(())
            });
        }

        scope.join_all().await?;

        let mut collected = results.lock().await;
        collected.sort_by_key(|(idx, _)| *idx);
        Ok(collected.drain(..).map(|(_, s)| s).collect())
    }
}

fn truncate_summary(s: &str) -> String {
    if s.len() <= MAX_SUMMARY_CHARS {
        return s.to_string();
    }
    let mut out: String = s.chars().take(MAX_SUMMARY_CHARS - 20).collect();
    out.push_str("\n[truncated]");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_limit() {
        let long = "a".repeat(3000);
        let truncated = truncate_summary(&long);
        assert!(truncated.len() <= MAX_SUMMARY_CHARS);
        assert!(truncated.ends_with("[truncated]"));
    }

    #[test]
    fn short_text_not_truncated() {
        let short = "hello world";
        assert_eq!(truncate_summary(short), short);
    }
}
