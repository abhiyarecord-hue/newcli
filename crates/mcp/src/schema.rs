//! JSON-RPC 2.0 message types + MCP tool schema validation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use agent_types::{AgentError, Result, ToolEffects, ToolSchema};

/// JSON-RPC identifiers are type-sensitive: integer `7` and string `"7"`
/// are distinct pending requests. Other JSON value kinds are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RpcId {
    Integer(i64),
    String(String),
}

impl std::fmt::Display for RpcId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer(value) => write!(formatter, "{value}"),
            Self::String(value) => write!(formatter, "{value:?}"),
        }
    }
}

const MAX_DESCRIPTION_CHARS: usize = 4096;
const DESCRIPTION_TRUNCATION_MARKER: &str = "[truncated]";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: RpcId,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: RpcId,
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

/// A validated discovered tool with protocol identity kept separate from the
/// collision-safe name exposed to the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedMcpTool {
    pub local_name: String,
    pub remote_name: String,
    pub description: String,
    pub input_schema: Value,
}

impl ValidatedMcpTool {
    /// Validate untrusted discovery data while preserving the exact name that
    /// must be sent back to the server for `tools/call`.
    pub fn validate(raw: Value, server_name: &str) -> Result<Self> {
        let remote_name = raw
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError::Tool {
                name: "mcp".into(),
                reason: "tool schema missing 'name'".into(),
            })?
            .to_string();

        let description = raw.get("description").and_then(Value::as_str).unwrap_or("");

        let input_schema = raw
            .get("inputSchema")
            .or_else(|| raw.get("input_schema"))
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        if !input_schema.is_object() {
            return Err(AgentError::Tool {
                name: "mcp".into(),
                reason: format!("tool '{remote_name}' input_schema is not an object"),
            });
        }

        Ok(Self {
            local_name: format!("mcp__{server_name}__{remote_name}"),
            remote_name,
            description: truncate_description(description, MAX_DESCRIPTION_CHARS),
            input_schema,
        })
    }

    /// Return only model-facing data. The remote protocol name deliberately
    /// does not become part of the model's tool schema.
    pub fn model_schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.local_name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            // Name the effects an MCP call definitely has — it crosses a process
            // boundary to a server started with inherited environment secrets,
            // and can write outside the workspace — while keeping `unknown` set,
            // because a remote tool may also spawn processes or touch the
            // workspace and its schema cannot say.
            //
            // Both halves matter. Declaring only `UNKNOWN` left the outbound and
            // credential policies unaware of a real network/secret path.
            // Declaring only the three named effects would have *removed* remote
            // calls from the destructive-command scan, which keys on
            // `process_spawn || unknown`. The union is strictly stronger than
            // either alone.
            effects: ToolEffects::NETWORK_OUTBOUND
                .union(ToolEffects::EXTERNAL_WRITE)
                .union(ToolEffects::CREDENTIAL_BEARING)
                .union(ToolEffects::UNKNOWN),
        }
    }
}

impl McpToolSchema {
    /// Validate and convert an untrusted remote tool schema.
    ///
    /// This compatibility API keeps returning the remote schema and the
    /// model-facing namespaced schema. New consumers that invoke tools should
    /// retain [`ValidatedMcpTool`] so the two names cannot be conflated.
    pub fn validate_and_namespace(
        raw: Value,
        server_name: &str,
    ) -> Result<(McpToolSchema, ToolSchema)> {
        let validated = ValidatedMcpTool::validate(raw, server_name)?;
        let remote = McpToolSchema {
            name: validated.remote_name.clone(),
            description: validated.description.clone(),
            input_schema: validated.input_schema.clone(),
        };
        let local = validated.model_schema();
        Ok((remote, local))
    }
}

fn truncate_description(description: &str, max_chars: usize) -> String {
    if description.chars().count() <= max_chars {
        return description.to_string();
    }

    let marker_chars = DESCRIPTION_TRUNCATION_MARKER.chars().count();
    if max_chars <= marker_chars {
        return DESCRIPTION_TRUNCATION_MARKER
            .chars()
            .take(max_chars)
            .collect();
    }

    let mut truncated: String = description.chars().take(max_chars - marker_chars).collect();
    truncated.push_str(DESCRIPTION_TRUNCATION_MARKER);
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validated_tool_separates_exact_remote_name_from_model_alias() {
        // **Validates: Requirements 2.3, 3.11**
        let raw = json!({
            "name": "search/मुद्दे",
            "description": "Search issues",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}
        });

        let validated = ValidatedMcpTool::validate(raw, "github").unwrap();
        let model_schema = validated.model_schema();

        assert_eq!(validated.remote_name, "search/मुद्दे");
        assert_eq!(validated.local_name, "mcp__github__search/मुद्दे");
        assert_eq!(model_schema.name, validated.local_name);
        assert_ne!(model_schema.name, validated.remote_name);
        assert!(model_schema.input_schema.is_object());
    }

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
        // **Validates: Requirements 3.11**
        let raw = json!({
            "name": "bad_tool",
            "description": "test",
            "inputSchema": "not an object"
        });
        let result = McpToolSchema::validate_and_namespace(raw, "test");
        assert!(result.is_err());
    }

    #[test]
    fn description_at_scalar_cap_is_not_truncated() {
        // **Validates: Requirements 2.5**
        let description = "😀".repeat(MAX_DESCRIPTION_CHARS);
        let raw = json!({
            "name": "tool",
            "description": description,
            "inputSchema": {"type": "object"}
        });

        let validated = ValidatedMcpTool::validate(raw, "s").unwrap();
        assert_eq!(validated.description.chars().count(), MAX_DESCRIPTION_CHARS);
        assert!(!validated
            .description
            .ends_with(DESCRIPTION_TRUNCATION_MARKER));
    }

    #[test]
    fn description_over_scalar_cap_includes_marker_inside_cap() {
        // **Validates: Requirements 2.5**
        let description = "ह".repeat(MAX_DESCRIPTION_CHARS + 1);
        let raw = json!({
            "name": "tool",
            "description": description,
            "inputSchema": {"type": "object"}
        });

        let validated = ValidatedMcpTool::validate(raw, "s").unwrap();
        assert_eq!(validated.description.chars().count(), MAX_DESCRIPTION_CHARS);
        assert!(validated
            .description
            .ends_with(DESCRIPTION_TRUNCATION_MARKER));
    }

    #[test]
    fn generated_unicode_boundaries_are_scalar_safe_and_capped() {
        // **Validates: Requirements 2.5**
        for prefix_len in (MAX_DESCRIPTION_CHARS - 4)..=(MAX_DESCRIPTION_CHARS + 4) {
            for scalar in ['é', 'ह', '😀'] {
                let description = format!("{}{scalar}", "a".repeat(prefix_len));
                let original_chars = description.chars().count();
                let capped = truncate_description(&description, MAX_DESCRIPTION_CHARS);

                assert!(capped.is_char_boundary(capped.len()));
                assert!(capped.chars().count() <= MAX_DESCRIPTION_CHARS);
                if original_chars > MAX_DESCRIPTION_CHARS {
                    assert_eq!(capped.chars().count(), MAX_DESCRIPTION_CHARS);
                    assert!(capped.ends_with(DESCRIPTION_TRUNCATION_MARKER));
                } else {
                    assert_eq!(capped, description);
                }
            }
        }
    }

    #[test]
    fn marker_is_always_inside_small_injected_caps() {
        // **Validates: Requirements 2.5**
        for cap in 0..=DESCRIPTION_TRUNCATION_MARKER.chars().count() + 2 {
            let capped = truncate_description("description that exceeds every small cap", cap);
            assert!(capped.chars().count() <= cap);
        }
    }
}
