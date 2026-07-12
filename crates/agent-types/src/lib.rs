//! `agent-types` (L0): shared enums/structs/errors for the whole workspace.
//!
//! ZERO heavy deps. This crate is transitively paid for by every other crate,
//! so it stays dependency-light (serde / thiserror / async-trait / tokio-util).
//!
//! Cross-cutting rules (plan.md section 3):
//! 1. Public APIs return [`Result`] — never `panic!`/`unwrap`/`expect` in lib code.
//! 2. `serde` derives on every wire/persisted type.

pub mod error;
pub mod event;
pub mod lang;
pub mod message;
pub mod tool;

pub use error::{AgentError, Result};
pub use event::AgentEvent;
pub use lang::LanguageMode;
pub use message::{ContentBlock, Message, Role};
pub use tool::{Tool, ToolCtx, ToolSchema};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_serde_round_trip() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("hello".to_string()),
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({ "path": "src/main.rs" }),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    output: "fn main() {}".to_string(),
                    is_error: false,
                },
            ],
            token_estimate: 42,
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        let back: Message = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.token_estimate, 42);
        assert_eq!(back.content.len(), 3);
        matches!(back.role, Role::Assistant);
        match &back.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "src/main.rs");
            }
            _ => panic!("expected ToolUse block"),
        }
    }

    #[test]
    fn language_mode_default_is_en() {
        assert_eq!(LanguageMode::default(), LanguageMode::En);
        assert_eq!(
            serde_json::to_string(&LanguageMode::Hinglish).unwrap(),
            "\"hinglish\""
        );
    }
}
