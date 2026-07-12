//! Tool abstraction. Every capability the agent exposes to the model is a
//! [`Tool`]. Signatures verbatim from plan.md section 3.

use crate::error::Result;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;
    async fn invoke(&self, input: serde_json::Value, ctx: &ToolCtx) -> Result<String>;
}

/// Everything a tool may touch. NO global state anywhere in the system.
pub struct ToolCtx {
    pub project_root: std::path::PathBuf,
    pub cancel: tokio_util::sync::CancellationToken,
}
