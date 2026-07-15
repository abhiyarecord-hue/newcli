//! Persistent state mapped to on-disk markdown files:
//! - SOUL.md: persona + policies (contains `language:` line)
//! - HEARTBEAT.md: task list with `- [ ]` / `- [x]` checkboxes
//! - MEMORY.md: append-only memory with ISO-8601 timestamps

use std::fs;
use std::path::{Path, PathBuf};

use agent_types::{AgentError, LanguageMode, Result};

pub struct PersistentState {
    root: PathBuf,
}

impl PersistentState {
    /// Load (or create) the persistent state directory at `root/.agent/`.
    /// Also ensures the core markdown files exist (empty) so the agent folder
    /// is not empty on first run.
    pub fn load(root: &Path) -> Result<Self> {
        let state = Self {
            root: root.to_path_buf(),
        };
        let dir = state.agent_dir();
        fs::create_dir_all(&dir)?;

        // Touch core files if missing.
        for path in [state.soul_path(), state.heartbeat_path(), state.memory_path()] {
            if !path.exists() {
                let _ = fs::write(&path, "");
            }
        }

        Ok(state)
    }

    fn agent_dir(&self) -> PathBuf {
        self.root.join(".agent")
    }

    fn soul_path(&self) -> PathBuf {
        self.agent_dir().join("SOUL.md")
    }

    fn heartbeat_path(&self) -> PathBuf {
        self.agent_dir().join("HEARTBEAT.md")
    }

    fn memory_path(&self) -> PathBuf {
        self.agent_dir().join("MEMORY.md")
    }

    // === Language Mode ===

    /// Read `language:` from SOUL.md Policies section.
    pub fn language_mode(&self) -> LanguageMode {
        let soul = fs::read_to_string(self.soul_path()).unwrap_or_default();
        for line in soul.lines() {
            let trimmed = line.trim();
            if let Some(val) = trimmed.strip_prefix("language:") {
                let val = val.trim();
                return match val {
                    "hinglish" => LanguageMode::Hinglish,
                    "en" => LanguageMode::En,
                    _ => LanguageMode::En,
                };
            }
        }
        LanguageMode::En
    }

    /// Set language mode with a surgical single-line edit.
    pub fn set_language_mode(&self, mode: LanguageMode) -> Result<()> {
        let path = self.soul_path();
        let mode_str = match mode {
            LanguageMode::En => "en",
            LanguageMode::Hinglish => "hinglish",
        };

        let content = fs::read_to_string(&path).unwrap_or_default();
        let mut found = false;
        let new_content: String = content
            .lines()
            .map(|line| {
                if line.trim().starts_with("language:") {
                    found = true;
                    format!("language: {mode_str}")
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let final_content = if found {
            new_content
        } else {
            // Append under ## Policies section.
            if content.contains("## Policies") {
                let mut result = String::new();
                for line in content.lines() {
                    result.push_str(line);
                    result.push('\n');
                    if line.trim() == "## Policies" {
                        result.push_str(&format!("language: {mode_str}\n"));
                    }
                }
                result
            } else {
                format!("{content}\n## Policies\nlanguage: {mode_str}\n")
            }
        };

        fs::write(&path, &final_content)?;
        Ok(())
    }

    // === Heartbeat (Tasks) ===

    /// Get tasks as (done, description) pairs.
    pub fn heartbeat_tasks(&self) -> Vec<(bool, String)> {
        let content = fs::read_to_string(self.heartbeat_path()).unwrap_or_default();
        content
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix("- [x]") {
                    Some((true, rest.trim().to_string()))
                } else if let Some(rest) = trimmed.strip_prefix("- [ ]") {
                    Some((false, rest.trim().to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Set a task done by index (0-based). Surgical line edit.
    pub fn set_task_done(&self, idx: usize) -> Result<()> {
        let path = self.heartbeat_path();
        let content = fs::read_to_string(&path).unwrap_or_default();
        let mut task_idx = 0usize;
        let new_content: String = content
            .lines()
            .map(|line| {
                let trimmed = line.trim();
                if trimmed.starts_with("- [ ]") || trimmed.starts_with("- [x]") {
                    if task_idx == idx && trimmed.starts_with("- [ ]") {
                        task_idx += 1;
                        return line.replace("- [ ]", "- [x]");
                    }
                    task_idx += 1;
                }
                line.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, &new_content)?;
        Ok(())
    }

    // === Memory ===

    /// Append a timestamped memory entry.
    pub fn append_memory(&self, entry: &str) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let path = self.memory_path();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
        let line = format!("\n[{timestamp}] {entry}\n");

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        file.write_all(line.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_mode_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let state = PersistentState::load(dir.path()).unwrap();
        // Default is En.
        assert_eq!(state.language_mode(), LanguageMode::En);
        // Set to Hinglish.
        state.set_language_mode(LanguageMode::Hinglish).unwrap();
        assert_eq!(state.language_mode(), LanguageMode::Hinglish);
        // Set back to En.
        state.set_language_mode(LanguageMode::En).unwrap();
        assert_eq!(state.language_mode(), LanguageMode::En);
    }

    #[test]
    fn set_task_done_flips_checkbox() {
        let dir = tempfile::tempdir().unwrap();
        let state = PersistentState::load(dir.path()).unwrap();
        let hb = "# Tasks\n- [ ] first task\n- [ ] second task\n- [x] done task\n";
        fs::write(state.heartbeat_path(), hb).unwrap();

        state.set_task_done(0).unwrap();
        let tasks = state.heartbeat_tasks();
        assert_eq!(tasks[0], (true, "first task".to_string()));
        assert_eq!(tasks[1], (false, "second task".to_string()));
        assert_eq!(tasks[2], (true, "done task".to_string()));
    }

    #[test]
    fn set_task_done_preserves_other_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let state = PersistentState::load(dir.path()).unwrap();
        let hb = "# My Custom Header\n\nSome notes here.\n\n- [ ] task one\n- [ ] task two\n";
        fs::write(state.heartbeat_path(), hb).unwrap();
        state.set_task_done(0).unwrap();
        let content = fs::read_to_string(state.heartbeat_path()).unwrap();
        assert!(content.contains("# My Custom Header"));
        assert!(content.contains("Some notes here."));
        assert!(content.contains("- [x] task one"));
    }

    #[test]
    fn append_memory_adds_timestamped_entry() {
        let dir = tempfile::tempdir().unwrap();
        let state = PersistentState::load(dir.path()).unwrap();
        state.append_memory("learned something new").unwrap();
        let content = fs::read_to_string(state.memory_path()).unwrap();
        assert!(content.contains("learned something new"));
        assert!(content.contains("[20")); // ISO timestamp prefix
    }
}
