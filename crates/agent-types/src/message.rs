//! Conversation message model shared by providers, persistence, and orchestration.

/// Provider-owned protocol data that must never be mixed into executable tool
/// arguments. Its JSON shape remains provider-specific and optional.
pub type ProviderMetadata = serde_json::Value;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub token_estimate: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ContentBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolResult {
        tool_use_id: String,
        output: String,
        is_error: bool,
    },
}
