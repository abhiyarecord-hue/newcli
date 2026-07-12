//! Shared error type for the whole workspace.
//!
//! Every public API in every crate returns [`Result`] — never `panic!`,
//! `unwrap`, or `expect` outside tests (plan.md section 3, cross-cutting rule 1).

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("llm provider error: {0}")]
    Llm(String),
    #[error("tool `{name}` failed: {reason}")]
    Tool { name: String, reason: String },
    #[error("sandbox violation: {0}")]
    Sandbox(String),
    #[error("path jail violation: {0}")]
    PathJail(String),
    #[error("index error: {0}")]
    Index(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("lsp error: {0}")]
    Lsp(String),
    #[error("cancelled")]
    Cancelled,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;
