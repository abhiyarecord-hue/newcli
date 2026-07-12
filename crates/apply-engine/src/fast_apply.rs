//! Apply strategies: `FastApplyStrategy` (SLM endpoint) and `FallbackStrategy`
//! (deterministic marker-based splice for offline mode).

use agent_types::{AgentError, Result};

use crate::lazy_edit::LazyEdit;

/// Trait for strategies that merge a lazy-edit snippet into original content.
#[async_trait::async_trait]
pub trait ApplyStrategy: Send + Sync {
    async fn apply(&self, original: &str, edit: &LazyEdit) -> Result<String>;
}

/// POST to a configurable Fast Apply SLM endpoint (e.g. Morph).
pub struct FastApplyStrategy {
    endpoint: String,
    client: reqwest::Client,
}

impl FastApplyStrategy {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl ApplyStrategy for FastApplyStrategy {
    async fn apply(&self, original: &str, edit: &LazyEdit) -> Result<String> {
        let body = serde_json::json!({
            "original": original,
            "snippet": edit.snippet,
        });

        let resp = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Tool {
                name: "fast_apply".into(),
                reason: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(AgentError::Tool {
                name: "fast_apply".into(),
                reason: format!("http {}", resp.status()),
            });
        }

        let merged: String = resp.text().await.map_err(|e| AgentError::Tool {
            name: "fast_apply".into(),
            reason: e.to_string(),
        })?;

        Ok(merged)
    }
}

/// Deterministic, offline marker-splice strategy. Finds anchor lines around
/// each `... existing code ...` marker, preserves the matching original lines,
/// and stitches new lines from the snippet.
pub struct FallbackStrategy;

#[async_trait::async_trait]
impl ApplyStrategy for FallbackStrategy {
    async fn apply(&self, original: &str, edit: &LazyEdit) -> Result<String> {
        let marker_re = LazyEdit::marker_regex();
        let snippet_lines: Vec<&str> = edit.snippet.lines().collect();
        let original_lines: Vec<&str> = original.lines().collect();

        let mut result: Vec<String> = Vec::new();
        let mut orig_cursor = 0usize;
        let mut i = 0;

        while i < snippet_lines.len() {
            let line = snippet_lines[i];
            if marker_re.is_match(line) {
                // Find the anchor line after the marker in the snippet.
                let next_anchor = snippet_lines.get(i + 1);

                if let Some(anchor) = next_anchor {
                    // Find where this anchor appears in the original, starting from orig_cursor.
                    let anchor_pos = find_anchor(&original_lines, orig_cursor, anchor);
                    if let Some(pos) = anchor_pos {
                        // Copy original lines from cursor to the anchor position.
                        for j in orig_cursor..pos {
                            result.push(original_lines[j].to_string());
                        }
                        orig_cursor = pos;
                    } else {
                        // No anchor found: copy remaining original from cursor.
                        for j in orig_cursor..original_lines.len() {
                            result.push(original_lines[j].to_string());
                        }
                        orig_cursor = original_lines.len();
                    }
                } else {
                    // Marker is last line of snippet: copy remaining original.
                    for j in orig_cursor..original_lines.len() {
                        result.push(original_lines[j].to_string());
                    }
                    orig_cursor = original_lines.len();
                }
                i += 1;
            } else {
                // Non-marker line from snippet: emit it and advance orig past matching line.
                result.push(line.to_string());
                // Try to advance orig_cursor past a matching original line.
                if orig_cursor < original_lines.len()
                    && original_lines[orig_cursor].trim_end() == line.trim_end()
                {
                    orig_cursor += 1;
                }
                i += 1;
            }
        }

        Ok(result.join("\n"))
    }
}

/// Find the first line in `original_lines[start..]` that matches `anchor`
/// using trim_end comparison (trailing whitespace tolerance, TASK-5.1 guard).
fn find_anchor(original_lines: &[&str], start: usize, anchor: &str) -> Option<usize> {
    let target = anchor.trim_end();
    for (i, line) in original_lines[start..].iter().enumerate() {
        if line.trim_end() == target {
            return Some(start + i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fallback_preserves_original_around_markers() {
        let original = "line1\nline2\nline3\nline4\nline5";
        let snippet = "line1\nnew_code()\n// ... existing code ...\nline4\nline5";
        let edit = LazyEdit::new(snippet).unwrap();
        let merged = FallbackStrategy.apply(original, &edit).await.unwrap();

        // Must contain both new code and preserved original lines.
        assert!(merged.contains("new_code()"));
        assert!(merged.contains("line2"));
        assert!(merged.contains("line3"));
        assert!(merged.contains("line4"));
        assert!(merged.contains("line5"));
    }

    #[tokio::test]
    async fn fallback_trailing_whitespace_tolerance() {
        let original = "fn hello() {  \n    body\n}";
        let snippet = "fn hello() {\n// ... existing code ...\n}";
        let edit = LazyEdit::new(snippet).unwrap();
        let result = FallbackStrategy.apply(original, &edit).await.unwrap();
        assert!(result.contains("body"), "original body preserved: {result}");
    }
}
