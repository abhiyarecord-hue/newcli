//! `agent-core` (L4): turn-based orchestrator wiring everything together.
//!
//! - [`tools`]: [`ToolDispatcher`] + built-in tool implementations.
//! - [`orchestrator`]: [`Orchestrator::run_turn`] — the agentic loop.

pub mod builtin;
pub mod orchestrator;
pub mod tools;

pub use builtin::{
    default_tools, default_tools_with_subagent, mcp_tools, BashTool, CheckCodeTool, ListFilesTool,
    McpTool, ReadFileTool, SearchTextTool, SubAgentTool, WebFetchTool, WriteFileTool,
};
pub use orchestrator::Orchestrator;
pub use tools::ToolDispatcher;
