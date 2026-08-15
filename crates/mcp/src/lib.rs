//! `mcp` (L3): MCP client + server (JSON-RPC 2.0 over stdio/SSE).

pub mod client;
pub mod schema;
pub mod server;

pub use client::{
    McpClient, McpClientConfig, DEFAULT_INITIALIZATION_TIMEOUT, DEFAULT_MAX_JSON_LINE_BYTES,
    DEFAULT_NOTIFICATION_QUEUE_CAPACITY, DEFAULT_REQUEST_TIMEOUT, DEFAULT_WRITER_QUEUE_CAPACITY,
};
pub use schema::{JsonRpcMessage, JsonRpcNotification, McpToolSchema, RpcId, ValidatedMcpTool};
pub use server::McpServer;
