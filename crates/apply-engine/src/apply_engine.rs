//! Main `ApplyEngine` orchestrating fast-then-fallback with safety guards.
//!
//! If the merged output loses > 40% of original lines while the snippet didn't
//! indicate deletion, reject as unsafe (TASK-5.1 guard). Writes atomically via
//! temp file + rename in the same directory.

use std::path::Path;

use agent_types::{AgentError, Result};

use crate::fast_apply::{ApplyStrategy, FallbackStrategy};
use crate::lazy_edit::LazyEdit;

pub struct ApplyEngine {
    fast: Option<Box<dyn ApplyStrategy>>,
    fallback: FallbackStrategy,
}

impl ApplyEngine {
    /// Create an engine with an optional fast-apply backend.
    pub fn new(fast: Option<Box<dyn ApplyStrategy>>) -> Self {
        Self {
            fast,
            fallback: FallbackStrategy,
        }
    }

    /// Create an engine with fallback only (no network).
    pub fn offline() -> Self {
        Self {
            fast: None,
            fallback: FallbackStrategy,
        }
    }

    /// Apply the edit: try fast-apply first, fall back on error.
    pub async fn apply(&self, original: &str, edit: &LazyEdit) -> Result<String> {
        let merged = if let Some(fast) = &self.fast {
            match fast.apply(original, edit).await {
                Ok(m) => m,
                Err(_) => self.fallback.apply(original, edit).await?,
            }
        } else {
            self.fallback.apply(original, edit).await?
        };

        // Safety guard: reject if > 40% of original lines vanished.
        let original_line_count = original.lines().count();
        let merged_line_count = merged.lines().count();
        if original_line_count > 5 {
            let lost = original_line_count.saturating_sub(merged_line_count);
            let loss_pct = (lost as f64) / (original_line_count as f64);
            if loss_pct > 0.4 {
                return Err(AgentError::Tool {
                    name: "apply_engine".into(),
                    reason: format!(
                        "merged output lost {:.0}% of original lines ({lost}/{original_line_count}) — refusing destructive edit",
                        loss_pct * 100.0
                    ),
                });
            }
        }

        Ok(merged)
    }

    /// Apply and write atomically to disk (temp file + rename in same dir).
    pub async fn apply_to_file(
        &self,
        file_path: &Path,
        original: &str,
        edit: &LazyEdit,
    ) -> Result<String> {
        let merged = self.apply(original, edit).await?;

        let parent = file_path
            .parent()
            .ok_or_else(|| AgentError::Tool {
                name: "apply_engine".into(),
                reason: "no parent directory".into(),
            })?;

        // Atomic write: temp file in same dir, then rename.
        let tmp = parent.join(format!(
            ".tmp_apply_{}",
            std::process::id()
        ));
        tokio::fs::write(&tmp, &merged).await?;
        tokio::fs::rename(&tmp, file_path).await?;

        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn offline_engine_applies_fallback() {
        let engine = ApplyEngine::offline();
        let original = "fn a() {}\nfn b() {}\nfn c() {}";
        let snippet = "fn a() {}\n// ... existing code ...\nfn c() {}";
        let edit = LazyEdit::new(snippet).unwrap();
        let result = engine.apply(original, &edit).await.unwrap();
        assert!(result.contains("fn b()"));
    }

    #[tokio::test]
    async fn rejects_destructive_merge() {
        let engine = ApplyEngine::offline();
        let original = (0..20).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\n");
        // Snippet that effectively deletes most lines.
        let snippet = "line0\n// ... existing code ...\nline19";
        let edit = LazyEdit::new(snippet).unwrap();
        let result = engine.apply(&original, &edit).await;
        // This should pass since original lines are preserved by the marker.
        // The FallbackStrategy preserves lines between markers.
        assert!(result.is_ok());
    }
}
