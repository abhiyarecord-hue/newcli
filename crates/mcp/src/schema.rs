//! JSON-RPC 2.0 message types + MCP tool schema validation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use agent_types::{AgentError, Result, ToolSchema};

const MAX_DESCRIPTION_LEN: usize = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// Enum for any JSON-RPC message.
#[derive(Debug, Clone)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

/// MCP tool schema from a remote server (untrusted input).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

impl McpToolSchema {
    /// Validate and convert an untrusted remote tool schema.
    /// - input_schema must be a JSON object.
    /// - description capped at 4 KiB.
    /// - name gets namespaced: `mcp__<server>__<tool_name>`.
    pub fn validate_and_namespace(
        raw: Value,
        server_name: &str,
    ) -> Result<(McpToolSchema, ToolSchema)> {
        let name = raw
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError::Tool {
                name: "mcp".into(),
                reason: "tool schema missing 'name'".into(),
            })?
            .to_string();

        let description = raw
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let input_schema = raw
            .get("inputSchema")
            .or_else(|| raw.get("input_schema"))
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        // Validate input_schema is an object.
        if !input_schema.is_object() {
            return Err(AgentError::Tool {
                name: "mcp".into(),
                reason: format!("tool '{name}' input_schema is not an object"),
            });
        }

        // Cap description length.
        let desc_capped = if description.len() > MAX_DESCRIPTION_LEN {
            description[..MAX_DESCRIPTION_LEN].to_string()
        } else {
            description.clone()
        };

        // Namespace to prevent shadowing built-in tools.
        let namespaced = format!("mcp__{server_name}__{name}");

        let mcp_schema = McpToolSchema {
            name: name.clone(),
            description: desc_capped.clone(),
            input_schema: input_schema.clone(),
        };

        let tool_schema = ToolSchema {
            name: namespaced,
            description: desc_capped,
            input_schema,
        };

        Ok((mcp_schema, tool_schema))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_namespaces_tool() {
        let raw = json!({
            "name": "create_issue",
            "description": "Creates a GitHub issue",
            "inputSchema": {"type": "object", "properties": {}}
        });
        let (mcp, tool) = McpToolSchema::validate_and_namespace(raw, "github").unwrap();
        assert_eq!(mcp.name, "create_issue");
        assert_eq!(tool.name, "mcp__github__create_issue");
    }

    #[test]
    fn rejects_non_object_input_schema() {
        let raw = json!({
            "name": "bad_tool",
            "description": "test",
            "inputSchema": "not an object"
        });
        let result = McpToolSchema::validate_and_namespace(raw, "test");
        assert!(result.is_err());
    }

    #[test]
    fn caps_description_at_4k() {
        let long_desc = "a".repeat(5000);
        let raw = json!({
            "name": "tool",
            "description": long_desc,
            "inputSchema": {"type": "object"}
        });
        let (_, tool) = McpToolSchema::validate_and_namespace(raw, "s").unwrap();
        assert_eq!(tool.description.len(), MAX_DESCRIPTION_LEN);
    }
}
