//! Shared tool-transaction validation for conversation history.
//!
//! A tool transaction is one `ToolUse` followed by exactly one later
//! `ToolResult` carrying the same ID. Providers reject histories that break this
//! pairing, so every path that loads, restores, compacts, or commits history
//! uses the rules here instead of re-deriving them.

use crate::message::{ContentBlock, Message};

/// A way in which a message list breaks tool-transaction pairing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolTransactionIssue {
    /// A `ToolResult` whose `ToolUse` never appears earlier.
    OrphanResult { index: usize, tool_use_id: String },
    /// A `ToolUse` that never receives a later `ToolResult`.
    UnresolvedUse { index: usize, tool_use_id: String },
    /// A `ToolUse` answered more than once.
    DuplicateResult { index: usize, tool_use_id: String },
}

impl ToolTransactionIssue {
    /// Index of the offending message.
    pub fn index(&self) -> usize {
        match self {
            Self::OrphanResult { index, .. }
            | Self::UnresolvedUse { index, .. }
            | Self::DuplicateResult { index, .. } => *index,
        }
    }

    /// ID of the tool transaction involved.
    pub fn tool_use_id(&self) -> &str {
        match self {
            Self::OrphanResult { tool_use_id, .. }
            | Self::UnresolvedUse { tool_use_id, .. }
            | Self::DuplicateResult { tool_use_id, .. } => tool_use_id,
        }
    }
}

/// Report every pairing violation, in message order.
///
/// An empty result means every `ToolUse` has exactly one later `ToolResult` and
/// no `ToolResult` is orphaned.
pub fn validate_tool_transactions(messages: &[Message]) -> Vec<ToolTransactionIssue> {
    let mut issues = Vec::new();
    // Declared uses, in order, with their answer count.
    let mut declared: Vec<(String, usize, usize)> = Vec::new();

    for (index, message) in messages.iter().enumerate() {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, .. } => declared.push((id.clone(), index, 0)),
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    match declared.iter_mut().find(|(id, _, _)| id == tool_use_id) {
                        Some((_, _, answers)) => {
                            *answers += 1;
                            if *answers > 1 {
                                issues.push(ToolTransactionIssue::DuplicateResult {
                                    index,
                                    tool_use_id: tool_use_id.clone(),
                                });
                            }
                        }
                        None => issues.push(ToolTransactionIssue::OrphanResult {
                            index,
                            tool_use_id: tool_use_id.clone(),
                        }),
                    }
                }
                ContentBlock::Text(_) => {}
            }
        }
    }

    for (id, index, answers) in declared {
        if answers == 0 {
            issues.push(ToolTransactionIssue::UnresolvedUse {
                index,
                tool_use_id: id,
            });
        }
    }
    issues.sort_by_key(|issue| issue.index());
    issues
}

/// IDs of `ToolUse` blocks that have no later `ToolResult`.
///
/// A non-empty result means the conversation is mid-transaction, so loop
/// prevention must not abandon the turn.
pub fn unresolved_tool_uses(messages: &[Message]) -> Vec<String> {
    let mut unresolved: Vec<String> = Vec::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, .. } => unresolved.push(id.clone()),
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    unresolved.retain(|pending| pending != tool_use_id);
                }
                ContentBlock::Text(_) => {}
            }
        }
    }
    unresolved
}

/// Number of leading messages that reference a `ToolUse` which is absent.
///
/// A window cut through a transaction leaves an orphan `ToolResult` at the
/// front; those messages must be dropped before the history is used.
pub fn leading_orphan_len(messages: &[Message]) -> usize {
    let mut declared: Vec<&str> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let mut orphaned = false;
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, .. } => declared.push(id.as_str()),
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    if !declared.contains(&tool_use_id.as_str()) {
                        orphaned = true;
                    }
                }
                ContentBlock::Text(_) => {}
            }
        }
        if !orphaned {
            return index;
        }
    }
    messages.len()
}

/// Length of the longest prefix in which every `ToolUse` is resolved.
///
/// Messages beyond it form an incomplete trailing transaction.
pub fn complete_prefix_len(messages: &[Message]) -> usize {
    let mut unresolved: Vec<&str> = Vec::new();
    let mut complete_len = 0usize;

    for (index, message) in messages.iter().enumerate() {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, .. } => unresolved.push(id.as_str()),
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    unresolved.retain(|pending| *pending != tool_use_id.as_str());
                }
                ContentBlock::Text(_) => {}
            }
        }
        if unresolved.is_empty() {
            complete_len = index + 1;
        }
    }
    complete_len
}

/// Drop leading orphans and trailing incomplete transactions in place.
///
/// Returns the number of messages removed from the front and from the back.
pub fn trim_to_complete_groups(messages: &mut Vec<Message>) -> (usize, usize) {
    let leading = leading_orphan_len(messages);
    if leading > 0 {
        messages.drain(..leading);
    }
    let complete = complete_prefix_len(messages);
    let trailing = messages.len() - complete;
    messages.truncate(complete);
    (leading, trailing)
}

/// Apply a message cap without cutting through a tool transaction.
pub fn apply_bounded_window(messages: &mut Vec<Message>, max: usize) {
    if messages.len() > max {
        let excess = messages.len() - max;
        messages.drain(..excess);
    }
    // Dropping from the front can expose an orphan result.
    let leading = leading_orphan_len(messages);
    if leading > 0 {
        messages.drain(..leading);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;

    fn text(value: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text(value.into())],
            token_estimate: 0,
        }
    }

    fn tool_use(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: "read_file".into(),
                input: serde_json::json!({}),
                provider_metadata: None,
            }],
            token_estimate: 0,
        }
    }

    fn tool_result(id: &str) -> Message {
        Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                output: "ok".into(),
                is_error: false,
            }],
            token_estimate: 0,
        }
    }

    #[test]
    fn complete_transactions_report_no_issue() {
        // **Validates: Requirements 2.33, 2.39**
        let messages = vec![text("ask"), tool_use("a"), tool_result("a")];
        assert!(validate_tool_transactions(&messages).is_empty());
        assert!(unresolved_tool_uses(&messages).is_empty());
        assert_eq!(complete_prefix_len(&messages), 3);
        assert_eq!(leading_orphan_len(&messages), 0);
    }

    #[test]
    fn orphan_unresolved_and_duplicate_results_are_reported() {
        // **Validates: Requirements 2.33, 2.39**
        let messages = vec![
            tool_result("missing"),
            tool_use("a"),
            tool_result("a"),
            tool_result("a"),
            tool_use("b"),
        ];
        let issues = validate_tool_transactions(&messages);

        assert!(issues.contains(&ToolTransactionIssue::OrphanResult {
            index: 0,
            tool_use_id: "missing".into()
        }));
        assert!(issues.contains(&ToolTransactionIssue::DuplicateResult {
            index: 3,
            tool_use_id: "a".into()
        }));
        assert!(issues.contains(&ToolTransactionIssue::UnresolvedUse {
            index: 4,
            tool_use_id: "b".into()
        }));
        assert_eq!(unresolved_tool_uses(&messages), vec!["b".to_string()]);
    }

    #[test]
    fn trimming_removes_leading_orphans_and_trailing_incomplete_groups() {
        // **Validates: Requirements 2.39**
        let mut messages = vec![
            tool_result("gone"),
            text("kept"),
            tool_use("a"),
            tool_result("a"),
            tool_use("b"),
        ];
        let (leading, trailing) = trim_to_complete_groups(&mut messages);

        assert_eq!((leading, trailing), (1, 1));
        assert_eq!(messages.len(), 3);
        assert!(validate_tool_transactions(&messages).is_empty());
    }

    #[test]
    fn bounded_window_never_leaves_an_orphan_result() {
        // **Validates: Requirements 2.39**
        let mut messages = vec![text("old"), tool_use("a"), tool_result("a")];
        apply_bounded_window(&mut messages, 1);
        assert!(validate_tool_transactions(&messages).is_empty());
        assert!(messages.iter().all(|message| message
            .content
            .iter()
            .all(|block| !matches!(block, ContentBlock::ToolResult { .. }))));
    }

    #[test]
    fn multiple_uses_in_one_message_are_each_tracked() {
        // **Validates: Requirements 2.33**
        let batch = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "a".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({}),
                    provider_metadata: None,
                },
                ContentBlock::ToolUse {
                    id: "b".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({}),
                    provider_metadata: None,
                },
            ],
            token_estimate: 0,
        };
        let messages = vec![batch, tool_result("a")];
        assert_eq!(unresolved_tool_uses(&messages), vec!["b".to_string()]);
        assert_eq!(complete_prefix_len(&messages), 0);
    }
}
