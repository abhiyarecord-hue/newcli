//! `mcp` (L3): MCP client + server (JSON-RPC 2.0 over stdio/SSE).

pub mod client;
pub mod schema;
pub mod server;

pub use client::McpClient;
pub use schema::{JsonRpcMessage, McpToolSchema};
pub use server::McpServer;
