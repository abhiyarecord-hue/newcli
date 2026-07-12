//! MCP client: spawns an external MCP server process, performs initialize
//! handshake, and exposes remote tools as `Vec<ToolSchema>`.

use std::collections::HashMap;
use std::process::Stdio;

use agent_types::{AgentError, Result, ToolSchema};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::schema::McpToolSchema;

pub struct McpClient {
    child: Option<Child>,
    stdin: Mutex<tokio::process::ChildStdin>,
    stdout: Mutex<BufReader<tokio::process::ChildStdout>>,
    next_id: std::sync::atomic::AtomicI64,
    server_name: String,
    tools: Vec<ToolSchema>,
}

impl McpClient {
    /// Connect to an MCP server by spawning `cmd`.
    pub async fn connect(cmd: &str, args: &[&str], server_name: &str) -> Result<Self> {
        let mut child = Command::new(cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| AgentError::Tool {
                name: "mcp_client".into(),
                reason: format!("spawn '{cmd}': {e}"),
            })?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();

        let mut client = Self {
            child: Some(child),
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            next_id: std::sync::atomic::AtomicI64::new(1),
            server_name: server_name.to_string(),
            tools: Vec::new(),
        };

        // Initialize handshake.
        let init_result = client
            .request("initialize", json!({"capabilities": {}}))
            .await?;

        // tools/list to discover remote tools.
        let tools_result = client.request("tools/list", json!({})).await?;
        if let Some(tools_arr) = tools_result.get("tools").and_then(Value::as_array) {
            for raw_tool in tools_arr {
                if let Ok((_, tool_schema)) =
                    McpToolSchema::validate_and_namespace(raw_tool.clone(), server_name)
                {
                    client.tools.push(tool_schema);
                }
            }
        }

        Ok(client)
    }

    /// Get discovered remote tools (already namespaced).
    pub fn tools(&self) -> &[ToolSchema] {
        &self.tools
    }

    /// Invoke a remote tool via `tools/call`.
    pub async fn call_tool(&self, tool_name: &str, input: Value) -> Result<String> {
        let result = self
            .request(
                "tools/call",
                json!({"name": tool_name, "arguments": input}),
            )
            .await?;

        // Extract text content from result.
        let content = result
            .get("content")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("");
        Ok(content.to_string())
    }

    /// Send a JSON-RPC request and read the response (newline-delimited).
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let line = serde_json::to_string(&msg)
            .map_err(|e| AgentError::Tool {
                name: "mcp_client".into(),
                reason: e.to_string(),
            })?;

        {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(line.as_bytes())
                .await
                .map_err(|e| AgentError::Tool {
                    name: "mcp_client".into(),
                    reason: format!("write: {e}"),
                })?;
            stdin.write_all(b"\n").await.map_err(|e| AgentError::Tool {
                name: "mcp_client".into(),
                reason: format!("write newline: {e}"),
            })?;
            stdin.flush().await.map_err(|e| AgentError::Tool {
                name: "mcp_client".into(),
                reason: format!("flush: {e}"),
            })?;
        }

        // Read response line.
        let mut response_line = String::new();
        {
            let mut stdout = self.stdout.lock().await;
            stdout
                .read_line(&mut response_line)
                .await
                .map_err(|e| AgentError::Tool {
                    name: "mcp_client".into(),
                    reason: format!("read: {e}"),
                })?;
        }

        let resp: Value = serde_json::from_str(&response_line).map_err(|e| AgentError::Tool {
            name: "mcp_client".into(),
            reason: format!("parse response: {e}"),
        })?;

        if let Some(err) = resp.get("error") {
            return Err(AgentError::Tool {
                name: "mcp_client".into(),
                reason: format!("rpc error: {err}"),
            });
        }

        Ok(resp.get("result").cloned().unwrap_or(Value::Null))
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.start_kill();
        }
    }
}
