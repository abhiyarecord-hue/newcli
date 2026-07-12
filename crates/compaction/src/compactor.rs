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
use crate::token_count::{estimate_history, estimate_tokens};

pub struct ThresholdCompactor;

impl Compactor for ThresholdCompactor {
    fn compact(&self, cfg: &CompactionConfig, history: &[Message]) -> (String, Vec<Message>) {
        // Under budget: nothing to do.
        if estimate_history(history) <= cfg.max_context_tokens {
            return (String::new(), history.to_vec());
        }

        let keep = cfg.keep_recent_messages.max(1);
        let mut start = history.len().saturating_sub(keep);
        start = adjust_boundary(history, start);

        let (older, recent) = history.split_at(start);
        if older.is_empty() {
            // Everything is "recent" (tool-pair extension consumed the prefix);
            // no summary to add.
            return (String::new(), recent.to_vec());
        }

        let summary = summarize(older, cfg.summary_target_tokens);
        (summary, recent.to_vec())
    }
}

/// Walk the boundary backward until every `ToolResult` in `history[start..]`
/// has its matching `ToolUse` also in `history[start..]`.
fn adjust_boundary(history: &[Message], mut start: usize) -> usize {
    loop {
        if start == 0 {
            return 0;
        }
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

/// Build the structured `[COMPACTED] ...` summary from the dropped prefix.
fn summarize(older: &[Message], target_tokens: u32) -> String {
    let mut decisions: Vec<String> = Vec::new();
    let mut files: BTreeSet<String> = BTreeSet::new();
    let mut open_tasks: Vec<String> = Vec::new();

    for m in older {
        for b in &m.content {
            match b {
                ContentBlock::Text(t) => {
                    for line in t.lines() {
                        let l = line.trim();
                        if l.is_empty() {
                            continue;
                        }
                        if let Some(rest) = l.strip_prefix("- [ ]") {
                            open_tasks.push(rest.trim().to_string());
                        } else if l.to_lowercase().contains("todo") {
                            open_tasks.push(l.to_string());
                        } else if matches!(m.role, Role::Assistant) && decisions.len() < 8 {
                            decisions.push(truncate_chars(l, 120));
                        }
                    }
                }
                ContentBlock::ToolUse { input, .. } => collect_paths(input, &mut files),
                ContentBlock::ToolResult { .. } => {}
            }
        }
    }

    let decisions_str = join_or_none(&decisions);
    let files_str = join_or_none(&files.iter().cloned().collect::<Vec<_>>());
    let tasks_str = join_or_none(&open_tasks);

    let summary = format!(
        "[COMPACTED] decisions: {decisions_str}; files touched: {files_str}; open tasks: {tasks_str}"
    );
    truncate_to_tokens(summary, target_tokens)
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

fn join_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
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

fn truncate_to_tokens(s: String, target_tokens: u32) -> String {
    if estimate_tokens(&s) <= target_tokens {
        return s;
    }
    // Roughly 4 chars per token; keep a little headroom for the ellipsis.
    let max_chars = (target_tokens as usize).saturating_mul(4);
    truncate_chars(&s, max_chars)
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
        let history = vec![text_msg(Role::User, "hi"), text_msg(Role::Assistant, "hello")];
        let (summary, retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(summary.is_empty());
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
            let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
            history.push(text_msg(role, &format!("message number {i} with some words here")));
        }
        let (summary, retained) = ThresholdCompactor.compact(&cfg, &history);
        assert!(summary.starts_with("[COMPACTED]"));
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
        assert!(summary.contains("src/lib.rs"), "summary: {summary}");
    }
}
