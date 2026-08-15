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
pub mod tool_transactions;

pub use error::{AgentError, Result};
pub use event::{AgentEvent, EventLag};
pub use lang::LanguageMode;
pub use message::{ContentBlock, Message, ProviderMetadata, Role};
pub use tool::{
    ApprovalDecision, ApprovalKind, ApprovalProvider, ApprovalRequest, Tool, ToolCtx, ToolEffects,
    ToolExecutionRecord, ToolExecutionStatus, ToolExecutionTiming, ToolSchema,
    INCOMPLETE_OUTPUT_MARKER,
};
pub use tool_transactions::{
    apply_bounded_window, complete_prefix_len, leading_orphan_len, trim_to_complete_groups,
    unresolved_tool_uses, validate_tool_transactions, ToolTransactionIssue,
};

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
                    provider_metadata: None,
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
        assert!(
            !json.contains("provider_metadata"),
            "absent provider metadata must preserve the legacy wire shape"
        );
        let back: Message = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.token_estimate, 42);
        assert_eq!(back.content.len(), 3);
        matches!(back.role, Role::Assistant);
        match &back.content[1] {
            ContentBlock::ToolUse {
                id,
                name,
                input,
                provider_metadata,
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "src/main.rs");
                assert!(provider_metadata.is_none());
            }
            _ => panic!("expected ToolUse block"),
        }
    }

    #[test]
    fn legacy_tool_use_deserializes_without_provider_metadata() {
        // **Validates: Requirements 2.38, 3.10**
        let legacy = serde_json::json!({
            "role": "Assistant",
            "content": [{
                "ToolUse": {
                    "id": "legacy-call",
                    "name": "custom_tool",
                    "input": {"thought_signature": "user-owned"}
                }
            }],
            "token_estimate": 7
        });

        let message: Message = serde_json::from_value(legacy).unwrap();
        match &message.content[0] {
            ContentBlock::ToolUse {
                input,
                provider_metadata,
                ..
            } => {
                assert_eq!(input["thought_signature"], "user-owned");
                assert!(provider_metadata.is_none());
            }
            _ => panic!("expected legacy ToolUse block"),
        }
    }

    #[test]
    fn provider_metadata_round_trips_outside_executable_input() {
        // **Validates: Requirements 2.38, 3.10**
        let block = ContentBlock::ToolUse {
            id: "gemini-call".into(),
            name: "custom_tool".into(),
            input: serde_json::json!({"thought_signature": "user-owned"}),
            provider_metadata: Some(serde_json::json!({
                "thought_signature": "provider-owned"
            })),
        };
        let json = serde_json::to_value(&block).unwrap();
        let restored: ContentBlock = serde_json::from_value(json).unwrap();

        match restored {
            ContentBlock::ToolUse {
                input,
                provider_metadata,
                ..
            } => {
                assert_eq!(input["thought_signature"], "user-owned");
                assert_eq!(
                    provider_metadata.unwrap()["thought_signature"],
                    "provider-owned"
                );
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
