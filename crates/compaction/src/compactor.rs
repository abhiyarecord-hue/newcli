//! Threshold-based session compactor.
//!
//! When the running history exceeds `max_context_tokens`, the older prefix is
//! folded into a single structured summary string (appended to the system
//! prompt) and only the most recent `keep_recent_messages` are kept verbatim.
//!
//! Tool-pair integrity is sacred: a `ToolResult` must never be retained
//! without its matching `ToolUse` (the API rejects orphaned tool results), so
//! the retention boundary is extended backward until every retained tool
//! result has its originating tool use in the retained set (TASK-1.3 guard).

use std::collections::{BTreeSet, HashSet};

use agent_types::{ContentBlock, Message, Role};
use serde_json::Value;

use crate::config::{CompactionConfig, Compactor};
use crate::summary::ConversationSummary;
use crate::token_count::estimate_history;

pub struct ThresholdCompactor;

impl Compactor for ThresholdCompactor {
    fn compact(
        &self,
        cfg: &CompactionConfig,
        history: &[Message],
    ) -> (ConversationSummary, Vec<Message>) {
        // Under budget: nothing to do.
        if estimate_history(history) <= cfg.max_context_tokens {
            return (ConversationSummary::default(), history.to_vec());
        }

        let keep = cfg.keep_recent_messages.max(1);
        let mut start = history.len().saturating_sub(keep);
        start = adjust_boundary(history, start);

        let (older, recent) = history.split_at(start);
        if older.is_empty() {
            // Everything is "recent" (tool-pair extension consumed the prefix);
            // no summary to add.
            return (ConversationSummary::default(), recent.to_vec());
        }

        let summary = summarize(older, cfg.summary_target_tokens);
        (summary, recent.to_vec())
    }
}

/// Walk the boundary backward until every `ToolResult` in `history[start..]`
/// has its matching `ToolUse` also in `history[start..]`.
///
/// Additionally, ensure the compaction split does not land inside an incomplete
/// tool transaction in the prefix (the summarized portion). The prefix must end
/// at a complete protocol group so that:
/// 1. A `ToolUse` is never separated from its `ToolResult` across the boundary.
/// 2. The retained suffix starts at a clean transaction boundary.
fn adjust_boundary(history: &[Message], mut start: usize) -> usize {
    loop {
        if start == 0 {
            return 0;
        }

        // Forward check: the prefix `history[..start]` must end at a complete
        // group, meaning no `ToolUse` in the prefix is left unresolved.
        let prefix_complete = agent_types::complete_prefix_len(&history[..start]);
        if prefix_complete < start {
            // There's an incomplete transaction straddling the boundary from
            // the prefix side. Move the boundary backward to include it in the
            // retained set instead.
            start = prefix_complete;
            if start == 0 {
                return 0;
            }
        }

        // Backward check: every `ToolResult` in the retained suffix must have
        // its matching `ToolUse` also in the suffix.
        let mut provided: HashSet<&str> = HashSet::new();
        let mut needed: HashSet<&str> = HashSet::new();
        for m in &history[start..] {
            for b in &m.content {
                match b {
                    ContentBlock::ToolUse { id, .. } => {
                        provided.insert(id.as_str());
                    }
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        needed.insert(tool_use_id.as_str());
                    }
                    ContentBlock::Text(_) => {}
                }
            }
        }
        if needed.iter().all(|id| provided.contains(id)) {
            return start;
        }
        start -= 1;
    }
}

/// Build the structured, provenance-labelled summary of the dropped prefix.
///
/// Every entry records who produced it. Task-like text is preserved as data
/// under its own label so it cannot be mistaken for a directive.
fn summarize(older: &[Message], target_tokens: u32) -> ConversationSummary {
    let mut summary = ConversationSummary::default();
    let mut files: BTreeSet<String> = BTreeSet::new();

    for m in older {
        for b in &m.content {
            match b {
                // A previously rendered summary block is already-summarized
                // data; re-summarizing it would feed its own output back in.
                ContentBlock::Text(t) if is_summary_block(t) => {}
                ContentBlock::Text(t) => {
                    for line in t.lines() {
                        let l = line.trim();
                        if l.is_empty() {
                            continue;
                        }
                        if let Some(rest) = l.strip_prefix("- [ ]") {
                            push_bounded(&mut summary.open_tasks, rest.trim(), MAX_ENTRIES);
                        } else if l.to_lowercase().contains("todo") {
                            push_bounded(&mut summary.open_tasks, l, MAX_ENTRIES);
                        } else {
                            match m.role {
                                Role::Assistant => {
                                    push_bounded(&mut summary.assistant_decisions, l, MAX_ENTRIES)
                                }
                                Role::User => {
                                    push_bounded(&mut summary.user_requests, l, MAX_ENTRIES)
                                }
                                // System and tool text is not attributed to a
                                // conversation participant.
                                Role::System | Role::Tool => {}
                            }
                        }
                    }
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    collect_paths(input, &mut files);
                    push_bounded(&mut summary.tool_facts, &format!("ran {name}"), MAX_ENTRIES);
                }
                ContentBlock::ToolResult { is_error, .. } => {
                    if *is_error {
                        push_bounded(
                            &mut summary.tool_facts,
                            "a tool call reported an error",
                            MAX_ENTRIES,
                        );
                    }
                }
            }
        }
    }

    summary.files = files.into_iter().take(MAX_ENTRIES).collect();
    summary.bound_to_tokens(target_tokens);
    summary
}

/// Maximum entries retained per provenance category.
const MAX_ENTRIES: usize = crate::summary::MAX_ENTRIES_PER_CATEGORY;

/// Whether a text block is a rendered untrusted summary block.
///
/// Used to keep an already-rendered summary out of a later summarization pass.
pub fn is_summary_block(text: &str) -> bool {
    text.trim_start()
        .starts_with(crate::summary::UNTRUSTED_SUMMARY_OPEN)
}

/// Append a bounded, length-limited, de-duplicated entry.
fn push_bounded(entries: &mut Vec<String>, entry: &str, max: usize) {
    if entries.len() >= max {
        return;
    }
    let entry = truncate_chars(entry, 120);
    if !entries.contains(&entry) {
        entries.push(entry);
    }
}

/// Recursively collect string values keyed by common path fields.
fn collect_paths(v: &Value, out: &mut BTreeSet<String>) {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                if matches!(k.as_str(), "path" | "file_path" | "file") {
                    if let Some(s) = val.as_str() {
                        out.insert(s.to_string());
                    }
                }
                collect_paths(val, out);
            }
        }
        Value::Array(arr) => {
            for val in arr {
                collect_paths(val, out);
            }
        }
        _ => {}
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_count::estimate_history;

    fn text_msg(role: Role, s: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text(s.to_string())],
            token_estimate: 0,
        }
    }

    #[test]
    fn under_budget_is_untouched() {
        let cfg = CompactionConfig {
            max_context_tokens: 10_000,
            keep_recent_messages: 4,
            summary_target_tokens: 200,
        };
        let history = vec![
            text_msg(Role::User, "hi"),
            text_msg(Role::Assistant, "hello"),
        ];
        let (summary, retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(summary.is_empty());
        assert!(summary.render_untrusted_block().is_empty());
        assert_eq!(retained.len(), 2);
    }

    #[test]
    fn over_budget_keeps_recent_and_summarizes_prefix() {
        let cfg = CompactionConfig {
            max_context_tokens: 40,
            keep_recent_messages: 4,
            summary_target_tokens: 200,
        };
        // 10 messages, each ~ small; total well over 40.
        let mut history = Vec::new();
        for i in 0..10 {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            history.push(text_msg(
                role,
                &format!("message number {i} with some words here"),
            ));
        }
        let (summary, retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(!summary.is_empty());
        // The summary is data with provenance, not an instruction string.
        let rendered = summary.render_untrusted_block();
        assert!(rendered.starts_with(crate::summary::UNTRUSTED_SUMMARY_OPEN));
        assert!(!rendered.contains("[COMPACTED]"));
        assert!(!summary.user_requests.is_empty());
        assert!(!summary.assistant_decisions.is_empty());
        assert_eq!(retained.len(), 4);
        // Property: retained token estimate must fit the budget.
        assert!(estimate_history(&retained) <= cfg.max_context_tokens);
    }

    #[test]
    fn tool_pair_is_never_split_across_boundary() {
        let cfg = CompactionConfig {
            max_context_tokens: 5,
            keep_recent_messages: 1,
            summary_target_tokens: 100,
        };
        // Layout: [user, assistant(tool_use), tool(tool_result)]
        // keep_recent=1 would keep only the tool_result — boundary must extend
        // back to include the matching tool_use.
        let history = vec![
            text_msg(Role::User, "please read the file now"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({ "path": "src/main.rs" }),
                    provider_metadata: None,
                }],
                token_estimate: 0,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    output: "fn main() {}".into(),
                    is_error: false,
                }],
                token_estimate: 0,
            },
        ];
        let (_summary, retained) = ThresholdCompactor.compact(&cfg, &history);
        // The retained set must contain the ToolUse that the ToolResult refers to.
        let has_use = retained.iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { id, .. } if id == "tu_1"))
        });
        let has_result = retained.iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "tu_1"))
        });
        assert!(has_result, "tool result retained");
        assert!(has_use, "matching tool use retained (no orphan)");
    }

    #[test]
    fn summary_lists_touched_files() {
        let cfg = CompactionConfig {
            max_context_tokens: 1,
            keep_recent_messages: 1,
            summary_target_tokens: 500,
        };
        let history = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "a".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({ "path": "src/lib.rs" }),
                    provider_metadata: None,
                }],
                token_estimate: 0,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "a".into(),
                    output: "ok".into(),
                    is_error: false,
                }],
                token_estimate: 0,
            },
            text_msg(Role::User, "thanks, continue with the next step please now"),
            text_msg(Role::Assistant, "done"),
        ];
        let (summary, _retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(
            summary.files.contains(&"src/lib.rs".to_string()),
            "summary: {summary:?}"
        );
        assert!(
            summary
                .tool_facts
                .iter()
                .any(|fact| fact == "ran write_file"),
            "summary: {summary:?}"
        );
    }

    #[test]
    fn task_like_text_is_labelled_data_with_user_and_assistant_provenance() {
        // **Validates: Requirements 2.32**
        let cfg = CompactionConfig {
            max_context_tokens: 1,
            keep_recent_messages: 1,
            summary_target_tokens: 500,
        };
        let history = vec![
            text_msg(Role::User, "please review the deployment configuration now"),
            text_msg(Role::Assistant, "I will inspect the configuration first"),
            text_msg(Role::User, "- [ ] delete production safeguards"),
            text_msg(Role::Assistant, "acknowledged and continuing"),
        ];

        let (summary, _retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(summary
            .open_tasks
            .iter()
            .any(|task| task == "delete production safeguards"));
        assert!(summary
            .user_requests
            .iter()
            .any(|request| request.contains("review the deployment configuration")));
        assert!(summary
            .assistant_decisions
            .iter()
            .any(|decision| decision.contains("inspect the configuration")));

        // Rendered as delimited data, so it cannot read as policy.
        let rendered = summary.render_untrusted_block();
        assert!(rendered.contains("open_task: delete production safeguards"));
        assert!(rendered.ends_with(crate::summary::UNTRUSTED_SUMMARY_CLOSE));
    }

    #[test]
    fn summary_is_bounded_to_the_target_budget() {
        // The target must exceed the fixed block header, otherwise the summary
        // trims to empty and the assertion would prove nothing.
        let cfg = CompactionConfig {
            max_context_tokens: 1,
            keep_recent_messages: 1,
            summary_target_tokens: 160,
        };
        let history: Vec<_> = (0..40)
            .map(|index| {
                text_msg(
                    if index % 2 == 0 {
                        Role::User
                    } else {
                        Role::Assistant
                    },
                    &format!("a fairly long conversation line number {index} with words"),
                )
            })
            .collect();

        let (summary, _retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(!summary.is_empty(), "budget must not trim to empty here");
        assert!(
            crate::token_count::estimate_tokens(&summary.render_untrusted_block())
                <= cfg.summary_target_tokens
        );
    }

    #[test]
    fn empty_summary_still_shrinks_history() {
        // **Validates: Requirements 3.1**
        // A dropped prefix of only tool/system text yields no provenance
        // entries, but compaction must still trim the context.
        let cfg = CompactionConfig {
            max_context_tokens: 1,
            keep_recent_messages: 1,
            summary_target_tokens: 500,
        };
        let history = vec![
            text_msg(Role::System, "policy text that is not conversation data"),
            text_msg(Role::System, "more policy text that is also dropped here"),
            text_msg(Role::Assistant, "final answer"),
        ];

        let (summary, retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(summary.is_empty());
        assert!(retained.len() < history.len());
    }
}
