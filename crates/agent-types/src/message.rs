//! Conversation message model shared by the LLM client, compaction engine,
//! and orchestrator. Signatures are verbatim from plan.md section 3.

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
    },
    ToolResult {
        tool_use_id: String,
        output: String,
        is_error: bool,
    },
}
