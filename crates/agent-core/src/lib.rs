//! `agent-core` (L4): turn-based orchestrator wiring everything together.
//!
//! - [`tools`]: [`ToolDispatcher`] + built-in tool implementations.
//! - [`orchestrator`]: [`Orchestrator::run_turn`] — the agentic loop.

pub mod builtin;
pub mod orchestrator;
pub mod tools;

pub use builtin::{
    default_tools, BashTool, ListFilesTool, ReadFileTool, SearchTextTool, WebFetchTool,
    WriteFileTool,
};
pub use orchestrator::Orchestrator;
pub use tools::ToolDispatcher;
