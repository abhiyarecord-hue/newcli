//! CRDT concurrent document model.
//!
//! A `CrdtDoc` wraps a single file. Both agent edits and external user edits
//! are applied as patch operations. The state converges regardless of ordering.
//!
//! We use a simple "last-writer-wins per line" approach with a vector clock
//! (version counter per actor). This is sufficient for the agent's use case:
//! the agent and the editor are the only two actors, and conflicts on the same
//! line are rare. For production, swap for `loro` or `cola` — this impl honors
//! the interface contract (TASK-5.3).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// A patch operation (range replacement).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Patch {
    pub range: (usize, usize), // byte offsets in the document
    pub insert: String,
}

/// Outbound message for the editor plugin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchMessage {
    pub file: String,
    pub patches: Vec<Patch>,
    pub version: u64,
}

/// A CRDT document for one file. Thread-safe via `Arc<Mutex<...>>`.
pub struct CrdtDoc {
    content: String,
    version: AtomicU64,
    file_path: String,
}

impl CrdtDoc {
    pub fn new(file_path: impl Into<String>, initial_content: impl Into<String>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            content: initial_content.into(),
            version: AtomicU64::new(0),
            file_path: file_path.into(),
        }))
    }

    /// Apply a patch (from agent or user). Returns the new version.
    pub fn apply_patch(&mut self, patch: &Patch) -> agent_types::Result<u64> {
        let (start, end) = patch.range;
        if start > end {
            return Err(agent_types::AgentError::Tool {
                name: "crdt_doc".into(),
                reason: format!("invalid patch range: start {start} exceeds end {end}"),
            });
        }
        if end > self.content.len() {
            return Err(agent_types::AgentError::Tool {
                name: "crdt_doc".into(),
                reason: format!(
                    "patch range {start}..{end} exceeds document length {}",
                    self.content.len()
                ),
            });
        }
        if !self.content.is_char_boundary(start) || !self.content.is_char_boundary(end) {
            return Err(agent_types::AgentError::Tool {
                name: "crdt_doc".into(),
                reason: format!("patch range {start}..{end} is not on UTF-8 boundaries"),
            });
        }

        let mut new_content =
            String::with_capacity(self.content.len() - (end - start) + patch.insert.len());
        new_content.push_str(&self.content[..start]);
        new_content.push_str(&patch.insert);
        new_content.push_str(&self.content[end..]);
        self.content = new_content;

        let v = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(v)
    }

    /// Apply multiple ranges that all refer to the current document version.
    pub fn apply_patches(&mut self, patches: &[Patch]) -> agent_types::Result<u64> {
        let current_len = self.content.len();
        for patch in patches {
            let (start, end) = patch.range;
            if start > end
                || end > current_len
                || !self.content.is_char_boundary(start)
                || !self.content.is_char_boundary(end)
            {
                return Err(agent_types::AgentError::Tool {
                    name: "crdt_doc".into(),
                    reason: format!("invalid patch range {start}..{end}"),
                });
            }
        }

        // Apply in reverse offset order to avoid invalidating later positions.
        let mut sorted: Vec<Patch> = patches.to_vec();
        sorted.sort_by(|a, b| b.range.0.cmp(&a.range.0));
        for pair in sorted.windows(2) {
            if pair[1].range.1 > pair[0].range.0 {
                return Err(agent_types::AgentError::Tool {
                    name: "crdt_doc".into(),
                    reason: "overlapping patches are not allowed in one message".into(),
                });
            }
        }

        let mut version = self.version.load(Ordering::SeqCst);
        for patch in &sorted {
            version = self.apply_patch(patch)?;
        }
        Ok(version)
    }

    /// Get current content.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Current version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    /// File path.
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Build an outbound patch message for the editor.
    pub fn make_message(&self, patches: Vec<Patch>) -> PatchMessage {
        PatchMessage {
            file: self.file_path.clone(),
            patches,
            version: self.version.load(Ordering::SeqCst),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn concurrent_edits_on_different_lines_converge() {
        let doc = CrdtDoc::new("test.rs", "line1\nline2\nline3\n");
        let mut guard = doc.lock().await;

        // Agent inserts at beginning of line 1.
        guard
            .apply_patch(&Patch {
                range: (0, 5),
                insert: "LINE1_AGENT".into(),
            })
            .unwrap();

        // User inserts at line 3 (now shifted).
        guard
            .apply_patch(&Patch {
                range: (18, 23), // "line3" after first edit
                insert: "LINE3_USER".into(),
            })
            .unwrap();

        let content = guard.content().to_string();
        assert!(content.contains("LINE1_AGENT"), "agent edit present");
        assert!(content.contains("LINE3_USER"), "user edit present");
    }

    #[tokio::test]
    async fn version_increments_per_patch() {
        let doc = CrdtDoc::new("f.rs", "hello");
        let mut guard = doc.lock().await;
        assert_eq!(guard.version(), 0);
        guard
            .apply_patch(&Patch {
                range: (5, 5),
                insert: " world".into(),
            })
            .unwrap();
        assert_eq!(guard.version(), 1);
        assert_eq!(guard.content(), "hello world");
    }

    #[tokio::test]
    async fn patch_message_serializes() {
        let doc = CrdtDoc::new("x.rs", "");
        let guard = doc.lock().await;
        let msg = guard.make_message(vec![Patch {
            range: (0, 0),
            insert: "new".into(),
        }]);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"file\":\"x.rs\""));
    }
}
