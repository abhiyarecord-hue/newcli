//! Chat history persistence — JSONL format (one JSON message per line).
//!
//! Stores conversation history at `.agent/HISTORY.jsonl`. On startup, the last
//! N messages are loaded back into the orchestrator so context carries over
//! across sessions (like Kiro's workspace memory).

use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use agent_types::{Message, Result, AgentError};

/// Maximum messages to retain in the history file. Older entries are pruned on
/// load to keep the file from growing unbounded.
const MAX_HISTORY_MESSAGES: usize = 200;

pub struct ChatHistory {
    path: PathBuf,
}

impl ChatHistory {
    /// Open (or create) the history file at `<project_root>/.agent/HISTORY.jsonl`.
    pub fn open(project_root: &Path) -> Result<Self> {
        let agent_dir = project_root.join(".agent");
        fs::create_dir_all(&agent_dir)?;
        let path = agent_dir.join("HISTORY.jsonl");
        Ok(Self { path })
    }

    /// Load the most recent messages from the history file. Returns an empty
    /// vec if the file doesn't exist or is empty.
    pub fn load(&self, max_messages: Option<usize>) -> Result<Vec<Message>> {
        let max = max_messages.unwrap_or(MAX_HISTORY_MESSAGES);

        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.path)
            .map_err(|e| AgentError::Storage(format!("open history: {e}")))?;
        let reader = std::io::BufReader::new(file);

        let mut messages: Vec<Message> = Vec::new();
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Message>(trimmed) {
                Ok(msg) => messages.push(msg),
                Err(_) => continue, // skip malformed lines gracefully
            }
        }

        // Keep only the most recent messages.
        if messages.len() > max {
            messages = messages.split_off(messages.len() - max);
        }

        Ok(messages)
    }

    /// Append one or more messages to the history file.
    pub fn append(&self, messages: &[Message]) -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| AgentError::Storage(format!("open history for append: {e}")))?;

        for msg in messages {
            let json = serde_json::to_string(msg)
                .map_err(|e| AgentError::Storage(format!("serialize message: {e}")))?;
            writeln!(file, "{json}")
                .map_err(|e| AgentError::Storage(format!("write history: {e}")))?;
        }

        Ok(())
    }

    /// Compact: rewrite the file keeping only the last `max` messages.
    /// Called periodically or on startup to prevent unbounded growth.
    pub fn compact(&self, max: usize) -> Result<()> {
        let messages = self.load(Some(max))?;
        // Rewrite the entire file with only retained messages.
        let mut file = fs::File::create(&self.path)
            .map_err(|e| AgentError::Storage(format!("compact history: {e}")))?;
        for msg in &messages {
            let json = serde_json::to_string(msg)
                .map_err(|e| AgentError::Storage(format!("serialize: {e}")))?;
            writeln!(file, "{json}")
                .map_err(|e| AgentError::Storage(format!("write: {e}")))?;
        }
        Ok(())
    }

    /// Clear all history (e.g. user command `/clear`).
    pub fn clear(&self) -> Result<()> {
        if self.path.exists() {
            fs::write(&self.path, "")
                .map_err(|e| AgentError::Storage(format!("clear history: {e}")))?;
        }
        Ok(())
    }

    /// Get the file path (for display/debug).
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_types::{ContentBlock, Role};

    fn sample_messages(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| Message {
                role: if i % 2 == 0 { Role::User } else { Role::Assistant },
                content: vec![ContentBlock::Text(format!("message {i}"))],
                token_estimate: 10,
            })
            .collect()
    }

    #[test]
    fn round_trip_append_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let msgs = sample_messages(5);
        history.append(&msgs).unwrap();

        let loaded = history.load(None).unwrap();
        assert_eq!(loaded.len(), 5);
    }

    #[test]
    fn load_caps_at_max() {
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let msgs = sample_messages(50);
        history.append(&msgs).unwrap();

        let loaded = history.load(Some(10)).unwrap();
        assert_eq!(loaded.len(), 10);
        // Should be the LAST 10
        if let ContentBlock::Text(t) = &loaded[0].content[0] {
            assert_eq!(t, "message 40");
        }
    }

    #[test]
    fn compact_prunes_old_messages() {
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        history.append(&sample_messages(100)).unwrap();
        history.compact(20).unwrap();

        let loaded = history.load(None).unwrap();
        assert_eq!(loaded.len(), 20);
    }

    #[test]
    fn clear_empties_file() {
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        history.append(&sample_messages(10)).unwrap();
        history.clear().unwrap();

        let loaded = history.load(None).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();
        let loaded = history.load(None).unwrap();
        assert_eq!(loaded.len(), 0);
    }
}
