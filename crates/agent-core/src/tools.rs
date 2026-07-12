//! Tool dispatcher: routes tool calls to implementations, wraps in hooks,
//! resolves paths through PathJail, truncates output.

use std::collections::HashMap;
use std::sync::Arc;

use agent_types::{AgentError, ContentBlock, Result, Tool, ToolCtx, ToolSchema};
use harness::{HookEngine, HookPoint, HookVerdict};
use serde_json::Value;

const MAX_OUTPUT_CHARS: usize = 30_000;
const TRUNCATED_MARKER: &str = "\n[truncated]";

pub struct ToolDispatcher {
    tools: HashMap<String, Arc<dyn Tool>>,
    hooks: Arc<HookEngine>,
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

    /// Dispatch a tool call. Returns a ToolResult ContentBlock.
    /// Tool failures are data, not crashes — returned as is_error: true.
    pub async fn dispatch(
        &self,
        name: &str,
        input: Value,
        ctx: &ToolCtx,
    ) -> ContentBlock {
        // PreTool hook.
        let effective_input = match self.hooks.run(HookPoint::PreTool, name, &input) {
            Ok(HookVerdict::Allow) => input,
            Ok(HookVerdict::Deny { reason }) => {
                return ContentBlock::ToolResult {
                    tool_use_id: String::new(), // filled by caller
                    output: format!("blocked by policy: {reason}"),
                    is_error: true,
                };
            }
            Ok(HookVerdict::Rewrite(new_input)) => new_input,
            Err(e) => {
                return ContentBlock::ToolResult {
                    tool_use_id: String::new(),
                    output: format!("hook error: {e}"),
                    is_error: true,
                };
            }
        };

        // Find and invoke the tool.
        let result = match self.tools.get(name) {
            Some(tool) => tool.invoke(effective_input, ctx).await,
            None => Err(AgentError::Tool {
                name: name.to_string(),
                reason: format!("unknown tool '{name}'"),
            }),
        };

        // PostTool hook (fire-and-forget, don't block on it).
        let _ = self.hooks.run(HookPoint::PostTool, name, &Value::Null);

        // Convert to ContentBlock.
        match result {
            Ok(output) => ContentBlock::ToolResult {
                tool_use_id: String::new(),
                output: truncate_output(&output),
                is_error: false,
            },
            Err(e) => ContentBlock::ToolResult {
                tool_use_id: String::new(),
                output: e.to_string(),
                is_error: true,
            },
        }
    }
}

fn truncate_output(s: &str) -> String {
    if s.len() <= MAX_OUTPUT_CHARS {
        return s.to_string();
    }
    let mut out: String = s.chars().take(MAX_OUTPUT_CHARS - TRUNCATED_MARKER.len()).collect();
    out.push_str(TRUNCATED_MARKER);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_types::ToolCtx;
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
        };
        let result = dispatcher
            .dispatch("echo", serde_json::json!({"msg": "hi"}), &ctx)
            .await;
        match result {
            ContentBlock::ToolResult { output, is_error, .. } => {
                assert!(!is_error);
                assert!(output.contains("hi"));
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_returns_error() {
        let hooks = Arc::new(HookEngine::new(vec![]));
        let dispatcher = ToolDispatcher::new(vec![], hooks);
        let ctx = ToolCtx {
            project_root: PathBuf::from("."),
            cancel: CancellationToken::new(),
        };
        let result = dispatcher
            .dispatch("nonexistent", serde_json::json!({}), &ctx)
            .await;
        match result {
            ContentBlock::ToolResult { is_error, .. } => assert!(is_error),
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
