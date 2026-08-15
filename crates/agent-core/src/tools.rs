//! Tool dispatcher: routes tool calls to implementations, wraps in hooks,
//! resolves paths through PathJail, truncates output.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_types::{
    AgentError, ContentBlock, Tool, ToolCtx, ToolEffects, ToolExecutionRecord, ToolExecutionStatus,
    ToolExecutionTiming, ToolSchema,
};
use harness::{HookEngine, HookPoint, HookVerdict};
use serde_json::Value;

const MAX_OUTPUT_CHARS: usize = 30_000;
const TRUNCATED_MARKER: &str = "\n[truncated]";

pub struct ToolDispatcher {
    tools: HashMap<String, Arc<dyn Tool>>,
    hooks: Arc<HookEngine>,
}

/// Extended dispatch result used by the orchestrator to stop a tool batch after
/// a fail-closed PostTool verdict while keeping `dispatch` source-compatible.
pub struct DispatchOutcome {
    pub block: ContentBlock,
    pub policy_failure: Option<String>,
    /// Whether this dispatch actually committed a workspace mutation.
    ///
    /// A caller that must prove real work happened in the current run reads
    /// this instead of inspecting the filesystem, which cannot distinguish a
    /// pre-existing file from one this run created.
    pub committed_mutation: bool,
}

impl DispatchOutcome {
    #[must_use]
    pub fn must_stop_dispatch(&self) -> bool {
        self.policy_failure.is_some()
    }
}

impl ToolDispatcher {
    pub fn new(tools: Vec<Arc<dyn Tool>>, hooks: Arc<HookEngine>) -> Self {
        let map = tools
            .into_iter()
            .map(|t| (t.schema().name.clone(), t))
            .collect();
        Self { tools: map, hooks }
    }

    /// Get all tool schemas (for sending to the LLM).
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }

    /// Compatibility dispatch API. Callers managing a batch should use
    /// `dispatch_with_outcome` so PostTool policy failures stop later calls.
    pub async fn dispatch(&self, name: &str, input: Value, ctx: &ToolCtx) -> ContentBlock {
        self.dispatch_with_outcome(name, input, ctx).await.block
    }

    /// Dispatch with complete execution evidence and an enforceable PostTool
    /// outcome. PreTool remains the only point that can prevent side effects.
    pub async fn dispatch_with_outcome(
        &self,
        name: &str,
        input: Value,
        ctx: &ToolCtx,
    ) -> DispatchOutcome {
        self.dispatch_for_turn(name, input, ctx, 0).await
    }

    /// Dispatch and attribute the execution to `turn_id`.
    ///
    /// The orchestrator supplies its current turn so a caller can prove a
    /// mutation belongs to the run it is judging.
    pub async fn dispatch_for_turn(
        &self,
        name: &str,
        input: Value,
        ctx: &ToolCtx,
        turn_id: u64,
    ) -> DispatchOutcome {
        let requested_input = input.clone();
        let effects = self
            .tools
            .get(name)
            .map(|tool| tool.schema().effects)
            .unwrap_or(ToolEffects::UNKNOWN);
        if ctx.cancel.is_cancelled() {
            return DispatchOutcome {
                block: ContentBlock::ToolResult {
                    tool_use_id: String::new(),
                    output: "not dispatched: operation cancelled".into(),
                    is_error: true,
                },
                policy_failure: None,
                committed_mutation: false,
            };
        }
        let started_at_unix_ms = unix_time_ms();

        let pre = self
            .hooks
            .run_with_effects(HookPoint::PreTool, name, &input, effects);
        let (effective_input, status, output, error) = match pre {
            Ok(HookVerdict::Allow) => self.invoke(name, input, ctx).await,
            Ok(HookVerdict::Rewrite(effective)) => self.invoke(name, effective, ctx).await,
            Ok(HookVerdict::Deny { reason }) => (
                input,
                ToolExecutionStatus::Denied,
                None,
                Some(format!("blocked by policy: {reason}")),
            ),
            Err(error) => (
                input,
                ToolExecutionStatus::Denied,
                None,
                Some(format!("hook error (fail-closed): {error}")),
            ),
        };

        let record = ToolExecutionRecord {
            requested_input,
            effective_input,
            status,
            output,
            error,
            timing: ToolExecutionTiming {
                started_at_unix_ms,
                finished_at_unix_ms: unix_time_ms(),
            },
            effects,
            turn_id,
            committed_mutation: matches!(status, ToolExecutionStatus::Success)
                && commits_workspace_mutation(effects),
        };

        let post_payload = match serde_json::to_value(&record) {
            Ok(payload) => payload,
            Err(error) => {
                return policy_failure(format!(
                    "post-tool record serialization failed (fail-closed): {error}"
                ));
            }
        };

        match self
            .hooks
            .run_with_effects(HookPoint::PostTool, name, &post_payload, effects)
        {
            Ok(HookVerdict::Allow) => DispatchOutcome {
                block: block_from_record(&record, None),
                policy_failure: None,
                committed_mutation: record.committed_mutation,
            },
            Ok(HookVerdict::Rewrite(rewritten)) => match rewritten_result(&rewritten) {
                Some(presented) => DispatchOutcome {
                    block: block_from_record(&record, Some(presented)),
                    policy_failure: None,
                    // A rewrite changes only the presented result, not whether
                    // the side effect already happened.
                    committed_mutation: record.committed_mutation,
                },
                None => policy_failure(
                    "post-tool rewrite omitted a string output/error (fail-closed)".into(),
                ),
            },
            Ok(HookVerdict::Deny { reason }) => {
                policy_failure(format!("blocked by post-tool policy: {reason}"))
            }
            Err(error) => policy_failure(format!("post-tool hook error (fail-closed): {error}")),
        }
    }

    async fn invoke(
        &self,
        name: &str,
        effective_input: Value,
        ctx: &ToolCtx,
    ) -> (Value, ToolExecutionStatus, Option<String>, Option<String>) {
        let result = match self.tools.get(name) {
            Some(tool) => tool.invoke(effective_input.clone(), ctx).await,
            None => Err(AgentError::Tool {
                name: name.to_string(),
                reason: format!("unknown tool '{name}'"),
            }),
        };
        match result {
            Ok(output) => (
                effective_input,
                ToolExecutionStatus::Success,
                Some(output),
                None,
            ),
            Err(error) => {
                let status = if matches!(&error, AgentError::Cancelled) {
                    ToolExecutionStatus::Cancelled
                } else {
                    ToolExecutionStatus::Error
                };
                (effective_input, status, None, Some(error.to_string()))
            }
        }
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn rewritten_result(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => map
            .get("output")
            .or_else(|| map.get("error"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

fn block_from_record(record: &ToolExecutionRecord, rewritten: Option<String>) -> ContentBlock {
    let is_error = record.status != ToolExecutionStatus::Success;
    let presented = rewritten.unwrap_or_else(|| {
        record
            .output
            .as_ref()
            .or(record.error.as_ref())
            .cloned()
            .unwrap_or_else(|| "tool completed without a result".into())
    });
    ContentBlock::ToolResult {
        tool_use_id: String::new(),
        output: truncate_output(&presented),
        is_error,
    }
}

/// Whether a successful call with these effects *necessarily* changed the
/// workspace.
///
/// Declared effects are an approval-time superset, not evidence. `bash` and
/// `check_code` declare `workspace_write` because they *may* write, so counting
/// them would let a successful `check_code` on a prose-only answer masquerade as
/// completed work. Only a tool whose sole side effect is a workspace write
/// proves a mutation happened, so any process, network, external-write, or
/// unknown capability disqualifies the call as evidence.
fn commits_workspace_mutation(effects: ToolEffects) -> bool {
    effects.workspace_write
        && !effects.process_spawn
        && !effects.network_outbound
        && !effects.external_write
        && !effects.unknown
}

fn policy_failure(reason: String) -> DispatchOutcome {
    DispatchOutcome {
        block: ContentBlock::ToolResult {
            tool_use_id: String::new(),
            output: reason.clone(),
            is_error: true,
        },
        policy_failure: Some(reason),
        // A policy failure is never creditable work. Even when a side effect
        // already landed before PostTool denied it, the outcome is denied, so a
        // stage must not report completion from it.
        committed_mutation: false,
    }
}

fn truncate_output(s: &str) -> String {
    if s.len() <= MAX_OUTPUT_CHARS {
        return s.to_string();
    }
    let mut out: String = s
        .chars()
        .take(MAX_OUTPUT_CHARS - TRUNCATED_MARKER.len())
        .collect();
    out.push_str(TRUNCATED_MARKER);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the test tools return the crate-wide `Result` alias.
    use agent_types::{Result, ToolCtx};
    use std::path::PathBuf;
    use tokio_util::sync::CancellationToken;

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "echo".into(),
                description: "Echoes input".into(),
                input_schema: serde_json::json!({"type": "object"}),
                effects: Default::default(),
            }
        }
        async fn invoke(&self, input: Value, _ctx: &ToolCtx) -> Result<String> {
            Ok(input.to_string())
        }
    }

    #[tokio::test]
    async fn dispatch_known_tool_returns_output() {
        let hooks = Arc::new(HookEngine::new(vec![]));
        let dispatcher = ToolDispatcher::new(vec![Arc::new(EchoTool)], hooks);
        let ctx = ToolCtx {
            project_root: PathBuf::from("."),
            cancel: CancellationToken::new(),
            approval_provider: None,
        };
        let result = dispatcher
            .dispatch("echo", serde_json::json!({"msg": "hi"}), &ctx)
            .await;
        match result {
            ContentBlock::ToolResult {
                output, is_error, ..
            } => {
                assert!(!is_error);
                assert!(output.contains("hi"));
            }
            _ => panic!("expected ToolResult"),
        }
    }

    struct WritingTool {
        fail: bool,
    }

    #[async_trait::async_trait]
    impl Tool for WritingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "writer".into(),
                description: "Writes a workspace file".into(),
                input_schema: serde_json::json!({"type": "object"}),
                effects: ToolEffects::WORKSPACE_WRITE,
            }
        }
        async fn invoke(&self, _input: Value, _ctx: &ToolCtx) -> Result<String> {
            if self.fail {
                Err(AgentError::Tool {
                    name: "writer".into(),
                    reason: "disk full".into(),
                })
            } else {
                Ok("Wrote src/new.rs".into())
            }
        }
    }

    fn test_ctx() -> ToolCtx {
        ToolCtx {
            project_root: PathBuf::from("."),
            cancel: CancellationToken::new(),
            approval_provider: None,
        }
    }

    #[tokio::test]
    async fn successful_workspace_write_is_a_committed_mutation_attributed_to_its_turn() {
        // **Validates: Requirements 2.41**
        let dispatcher = ToolDispatcher::new(
            vec![Arc::new(WritingTool { fail: false })],
            Arc::new(HookEngine::new(vec![])),
        );
        let outcome = dispatcher
            .dispatch_for_turn("writer", serde_json::json!({}), &test_ctx(), 7)
            .await;
        assert!(outcome.committed_mutation);
        assert!(outcome.policy_failure.is_none());
    }

    #[tokio::test]
    async fn failed_write_read_only_and_unknown_tools_are_not_committed_mutations() {
        // **Validates: Requirements 2.41**
        let dispatcher = ToolDispatcher::new(
            vec![Arc::new(WritingTool { fail: true }), Arc::new(EchoTool)],
            Arc::new(HookEngine::new(vec![])),
        );

        // A workspace-writing tool that errored committed nothing.
        let failed = dispatcher
            .dispatch_for_turn("writer", serde_json::json!({}), &test_ctx(), 1)
            .await;
        assert!(!failed.committed_mutation);

        // A tool without the workspace_write effect can never count.
        let read_only = dispatcher
            .dispatch_for_turn("echo", serde_json::json!({}), &test_ctx(), 1)
            .await;
        assert!(!read_only.committed_mutation);

        // An unknown tool has unknown effects and fails closed.
        let unknown = dispatcher
            .dispatch_for_turn("missing", serde_json::json!({}), &test_ctx(), 1)
            .await;
        assert!(!unknown.committed_mutation);
    }

    struct CheckerLikeTool;

    #[async_trait::async_trait]
    impl Tool for CheckerLikeTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "checker".into(),
                description: "Runs a checker; may write".into(),
                input_schema: serde_json::json!({"type": "object"}),
                // Same shape as `check_code`/`bash`: workspace_write is declared
                // as an approval-time superset alongside process_spawn.
                effects: ToolEffects::WORKSPACE_READ
                    .union(ToolEffects::WORKSPACE_WRITE)
                    .union(ToolEffects::PROCESS_SPAWN),
            }
        }
        async fn invoke(&self, _input: Value, _ctx: &ToolCtx) -> Result<String> {
            Ok("no project type recognized".into())
        }
    }

    #[tokio::test]
    async fn process_spawning_tools_are_not_mutation_evidence() {
        // **Validates: Requirements 2.41**
        // `check_code` and `bash` declare workspace_write because they *may*
        // write. A successful run proves nothing was written, so it must not
        // satisfy the Implement completion gate.
        let dispatcher = ToolDispatcher::new(
            vec![Arc::new(CheckerLikeTool)],
            Arc::new(HookEngine::new(vec![])),
        );
        let outcome = dispatcher
            .dispatch_for_turn("checker", serde_json::json!({}), &test_ctx(), 1)
            .await;
        match &outcome.block {
            ContentBlock::ToolResult { is_error, .. } => assert!(!is_error),
            _ => panic!("expected ToolResult"),
        }
        assert!(
            !outcome.committed_mutation,
            "a successful checker run is not evidence of a workspace mutation"
        );
    }

    #[tokio::test]
    async fn cancelled_dispatch_is_not_a_committed_mutation() {
        // **Validates: Requirements 2.41**
        let dispatcher = ToolDispatcher::new(
            vec![Arc::new(WritingTool { fail: false })],
            Arc::new(HookEngine::new(vec![])),
        );
        let cancel = CancellationToken::new();
        cancel.cancel();
        let ctx = ToolCtx {
            project_root: PathBuf::from("."),
            cancel,
            approval_provider: None,
        };
        let outcome = dispatcher
            .dispatch_for_turn("writer", serde_json::json!({}), &ctx, 1)
            .await;
        assert!(!outcome.committed_mutation);
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_returns_error() {
        let hooks = Arc::new(HookEngine::new(vec![]));
        let dispatcher = ToolDispatcher::new(vec![], hooks);
        let ctx = ToolCtx {
            project_root: PathBuf::from("."),
            cancel: CancellationToken::new(),
            approval_provider: None,
        };
        let result = dispatcher
            .dispatch("nonexistent", serde_json::json!({}), &ctx)
            .await;
        match result {
            ContentBlock::ToolResult { is_error, .. } => assert!(is_error),
            _ => panic!("expected ToolResult"),
        }
    }

    struct RecordingRewriteHook {
        seen: Arc<std::sync::Mutex<Vec<Value>>>,
    }

    impl harness::Hook for RecordingRewriteHook {
        fn point(&self) -> HookPoint {
            HookPoint::PostTool
        }

        fn evaluate(&self, _tool_name: &str, payload: &Value) -> HookVerdict {
            self.seen.lock().unwrap().push(payload.clone());
            HookVerdict::Rewrite(serde_json::json!({"output": "redacted result"}))
        }
    }

    struct DenyingPostHook;

    impl harness::Hook for DenyingPostHook {
        fn point(&self) -> HookPoint {
            HookPoint::PostTool
        }

        fn evaluate(&self, _tool_name: &str, _payload: &Value) -> HookVerdict {
            HookVerdict::Deny {
                reason: "post denied fixture".into(),
            }
        }
    }

    #[tokio::test]
    async fn posttool_receives_complete_record_and_rewrites_only_presented_result() {
        // **Validates: Requirements 2.21**
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hooks = Arc::new(HookEngine::new(vec![Arc::new(RecordingRewriteHook {
            seen: seen.clone(),
        })]));
        let dispatcher = ToolDispatcher::new(vec![Arc::new(EchoTool)], hooks);
        let ctx = ToolCtx {
            project_root: PathBuf::from("."),
            cancel: CancellationToken::new(),
            approval_provider: None,
        };
        let requested = serde_json::json!({"msg": "original"});
        let outcome = dispatcher
            .dispatch_with_outcome("echo", requested.clone(), &ctx)
            .await;
        assert!(!outcome.must_stop_dispatch());
        let record = seen.lock().unwrap().first().cloned().unwrap();
        assert_eq!(record["requested_input"], requested);
        assert_eq!(record["effective_input"], requested);
        assert_eq!(record["status"], "success");
        assert!(record["output"].as_str().unwrap().contains("original"));
        match outcome.block {
            ContentBlock::ToolResult {
                output, is_error, ..
            } => {
                assert_eq!(output, "redacted result");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[tokio::test]
    async fn posttool_deny_suppresses_output_and_requires_batch_stop() {
        // **Validates: Requirements 2.21**
        let hooks = Arc::new(HookEngine::new(vec![Arc::new(DenyingPostHook)]));
        let dispatcher = ToolDispatcher::new(vec![Arc::new(EchoTool)], hooks);
        let ctx = ToolCtx {
            project_root: PathBuf::from("."),
            cancel: CancellationToken::new(),
            approval_provider: None,
        };
        let outcome = dispatcher
            .dispatch_with_outcome("echo", serde_json::json!({"secret_output": "hidden"}), &ctx)
            .await;
        assert!(outcome.must_stop_dispatch());
        match outcome.block {
            ContentBlock::ToolResult {
                output, is_error, ..
            } => {
                assert!(is_error);
                assert!(output.contains("post denied fixture"));
                assert!(!output.contains("hidden"));
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn output_truncation_works() {
        let long = "x".repeat(40_000);
        let truncated = truncate_output(&long);
        assert!(truncated.len() <= MAX_OUTPUT_CHARS);
        assert!(truncated.ends_with("[truncated]"));
    }
}
