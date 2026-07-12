//! MCP server: expose this agent's own tools to external MCP hosts.
//! Reads JSON-RPC requests from stdin, dispatches, writes responses to stdout.

use std::sync::Arc;

use agent_types::{Result, Tool, ToolSchema};
use serde_json::{json, Value};

/// The MCP server exposing our tools to external hosts.
pub struct McpServer {
    tools: Vec<Arc<dyn Tool>>,
}

impl McpServer {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self { tools }
    }

    /// Handle a single JSON-RPC request and return the response.
    pub async fn handle_request(&self, request: &Value) -> Value {
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("");

        let result = match method {
            "initialize" => json!({"capabilities": {"tools": {}}}),
            "tools/list" => {
                let schemas: Vec<Value> = self
                    .tools
                    .iter()
                    .map(|t| {
                        let s = t.schema();
                        json!({
                            "name": s.name,
                            "description": s.description,
                            "inputSchema": s.input_schema,
                        })
                    })
                    .collect();
                json!({"tools": schemas})
            }
            "tools/call" => {
                let params = request.get("params").cloned().unwrap_or(Value::Null);
                let tool_name = params
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let input = params.get("arguments").cloned().unwrap_or(Value::Null);

                match self.tools.iter().find(|t| t.schema().name == tool_name) {
                    Some(tool) => {
                        let ctx = agent_types::ToolCtx {
                            project_root: std::path::PathBuf::from("."),
                            cancel: tokio_util::sync::CancellationToken::new(),
                        };
                        match tool.invoke(input, &ctx).await {
                            Ok(output) => json!({"content": [{"type": "text", "text": output}]}),
                            Err(e) => json!({"content": [{"type": "text", "text": e.to_string()}], "isError": true}),
                        }
                    }
                    None => {
                        return json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32601, "message": format!("tool '{tool_name}' not found")}
                        });
                    }
                }
            }
            _ => {
                return json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32601, "message": format!("method '{method}' not found")}
                });
            }
        };

        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn handles_tools_list() {
        let server = McpServer::new(vec![]);
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}});
        let resp = server.handle_request(&req).await;
        assert_eq!(resp["id"], 1);
        assert!(resp["result"]["tools"].is_array());
    }

    #[tokio::test]
    async fn unknown_method_returns_error() {
        let server = McpServer::new(vec![]);
        let req = json!({"jsonrpc": "2.0", "id": 2, "method": "unknown/method", "params": {}});
        let resp = server.handle_request(&req).await;
        assert!(resp["error"]["code"].as_i64().unwrap() == -32601);
    }
}
