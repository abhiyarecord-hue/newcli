//! Deterministic token estimation — no tokenizer dependency.
//!
//! Heuristic per TASK-1.3: `(chars / 4).max(words)`. This is intentionally
//! model-agnostic and reproducible so compaction is a pure function.

use agent_types::{ContentBlock, Message};

/// Estimate the token count of an arbitrary string.
pub fn estimate_tokens(s: &str) -> u32 {
    let chars = s.chars().count() as u32;
    let words = s.split_whitespace().count() as u32;
    (chars / 4).max(words)
}

/// Estimate the token cost of a single message by flattening its content
/// blocks into representative text (tool names, serialized inputs, and outputs
/// all count toward the budget).
pub fn estimate_message(m: &Message) -> u32 {
    let mut total = 0u32;
    for block in &m.content {
        match block {
            ContentBlock::Text(t) => total += estimate_tokens(t),
            ContentBlock::ToolUse { name, input, .. } => {
                total += estimate_tokens(name);
                total += estimate_tokens(&input.to_string());
            }
            ContentBlock::ToolResult { output, .. } => total += estimate_tokens(output),
        }
    }
    total
}

/// Total estimated tokens across a message slice.
pub fn estimate_history(history: &[Message]) -> u32 {
    history.iter().map(estimate_message).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_is_max_of_char_quarter_and_word_count() {
        // 12 chars / 4 = 3, 2 words -> max = 3
        assert_eq!(estimate_tokens("hello world!"), 3);
        // many short words dominate: "a a a a a" = 9 chars/4=2, 5 words -> 5
        assert_eq!(estimate_tokens("a a a a a"), 5);
        assert_eq!(estimate_tokens(""), 0);
    }
}
