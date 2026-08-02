//! Chat history persistence.
//!
//! The authoritative format is a versioned snapshot at `.agent/HISTORY.v2.json`
//! that is atomically replaced after a completed turn or compaction. Each
//! persisted message carries a monotonic ID, so a consumer cursor cannot drift
//! when compaction shortens the in-memory history.
//!
//! The legacy append-only `.agent/HISTORY.jsonl` remains readable. It is
//! imported once, with malformed lines reported by line number and incomplete
//! tool groups trimmed, and the source is retained as a migration backup. A
//! failed migration is non-destructive: the legacy file is left in place.

use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use agent_types::{apply_bounded_window, trim_to_complete_groups, AgentError, Message, Result};
use compaction::ConversationSummary;
use runtime_core::{atomic_replace, atomic_replace_blocking, AtomicWriteOptions};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::persistent::block_on;

/// Maximum messages to retain in the history file. Older entries are pruned on
/// load to keep the file from growing unbounded.
const MAX_HISTORY_MESSAGES: usize = 200;

/// Schema version of [`ConversationSnapshotV2`].
pub const CONVERSATION_SCHEMA_VERSION: u32 = 2;

/// One persisted message with its stable monotonic identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedMessage {
    pub id: u64,
    pub message: Message,
}

/// Versioned conversation snapshot.
///
/// `generation` increases on every successful replacement so a reader can tell
/// which snapshot it observed. `next_message_id` is monotonic and never reused,
/// even after compaction drops earlier messages.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversationSnapshotV2 {
    pub schema_version: u32,
    pub generation: u64,
    pub next_message_id: u64,
    /// Structured compacted summary. An empty summary is meaningful, so this
    /// field may be omitted. Stored typed rather than as prose so a restore can
    /// hand it back to the orchestrator without re-parsing rendered text.
    #[serde(default)]
    pub compacted_summary: ConversationSummary,
    /// Required: a missing `messages` field is a corrupt snapshot, not an empty
    /// conversation, and must never load as a silent reset.
    pub messages: Vec<PersistedMessage>,
}

impl Default for ConversationSnapshotV2 {
    fn default() -> Self {
        Self {
            schema_version: CONVERSATION_SCHEMA_VERSION,
            generation: 0,
            next_message_id: 1,
            compacted_summary: ConversationSummary::default(),
            messages: Vec::new(),
        }
    }
}

impl ConversationSnapshotV2 {
    /// Messages in persisted order, without their IDs.
    pub fn plain_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .map(|persisted| persisted.message.clone())
            .collect()
    }

    /// Replace the persisted conversation with `messages`.
    ///
    /// IDs are drawn from the monotonic counter and are never reused, so a
    /// full-state replacement cannot resurrect an earlier identity. Callers use
    /// this instead of an index cursor, which cannot survive compaction
    /// truncating the history from the front.
    pub fn replace_messages(&mut self, messages: &[Message]) {
        self.messages.clear();
        self.extend(messages);
    }

    /// Append `messages`, assigning each the next monotonic ID.
    pub fn extend(&mut self, messages: &[Message]) {
        for message in messages {
            self.messages.push(PersistedMessage {
                id: self.next_message_id,
                message: message.clone(),
            });
            self.next_message_id = self.next_message_id.saturating_add(1);
        }
    }
}

/// Outcome of clearing active conversation history.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClearReport {
    /// Active stores that existed and were removed.
    pub removed: Vec<PathBuf>,
    /// Migration backups left on disk; never restored automatically.
    pub retained_backups: Vec<PathBuf>,
}

/// Diagnostics produced by a one-time legacy import.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MigrationReport {
    /// Messages imported after trimming.
    pub imported: usize,
    /// 1-based line numbers that could not be parsed.
    pub malformed_lines: Vec<usize>,
    /// Trailing messages dropped because their tool group was incomplete.
    pub trimmed_incomplete: usize,
    /// Leading messages dropped because they referenced an absent `ToolUse`.
    pub trimmed_leading_orphans: usize,
    /// Path the legacy file was retained at.
    pub backup_path: Option<PathBuf>,
}

pub struct ChatHistory {
    path: PathBuf,
    snapshot_path: PathBuf,
}

impl ChatHistory {
    /// Open (or create) the history directory at `<project_root>/.agent`.
    pub fn open(project_root: &Path) -> Result<Self> {
        let agent_dir = project_root.join(".agent");
        fs::create_dir_all(&agent_dir)?;
        Ok(Self {
            path: agent_dir.join("HISTORY.jsonl"),
            snapshot_path: agent_dir.join("HISTORY.v2.json"),
        })
    }

    /// Path of the authoritative versioned snapshot.
    pub fn snapshot_path(&self) -> &Path {
        &self.snapshot_path
    }

    /// Load the authoritative snapshot, importing legacy history once.
    ///
    /// A present v2 snapshot always wins. Otherwise a legacy file is imported,
    /// written as v2, and retained as a backup. Import never deletes the source.
    /// The returned [`MigrationReport`] is `Some` only when an import ran, so a
    /// caller can surface its line-number diagnostics.
    pub fn load_snapshot(
        &self,
        max_messages: Option<usize>,
    ) -> Result<(ConversationSnapshotV2, Option<MigrationReport>)> {
        let max = max_messages.unwrap_or(MAX_HISTORY_MESSAGES);

        if let Some(snapshot) = self.read_snapshot()? {
            return Ok((self.bounded_snapshot(snapshot, max), None));
        }

        // A fresh project has neither store, so no import ran and there is
        // nothing to report.
        if !self.path.exists() {
            return Ok((ConversationSnapshotV2::default(), None));
        }

        let (snapshot, report) = self.import_legacy(max)?;
        Ok((snapshot, Some(report)))
    }

    /// Apply the restore cap to a loaded snapshot without cutting through a
    /// tool transaction. Message IDs are preserved, never renumbered.
    fn bounded_snapshot(
        &self,
        snapshot: ConversationSnapshotV2,
        max: usize,
    ) -> ConversationSnapshotV2 {
        let mut plain = snapshot.plain_messages();
        let original_len = plain.len();
        apply_bounded_window(&mut plain, max);
        let dropped = original_len - plain.len();
        if dropped == 0 {
            return snapshot;
        }

        let mut bounded = snapshot;
        bounded.messages.drain(..dropped);
        bounded
    }

    /// Atomically replace the v2 snapshot, advancing its generation.
    ///
    /// A defensive `trim_to_complete_groups` is applied before serialization so
    /// an interrupted turn can never persist an orphaned `ToolResult` or
    /// unresolved `ToolUse` that would be rejected on restore.
    pub fn save_snapshot(
        &self,
        snapshot: &ConversationSnapshotV2,
    ) -> Result<ConversationSnapshotV2> {
        let mut next = snapshot.clone();
        next.schema_version = CONVERSATION_SCHEMA_VERSION;
        next.generation = next.generation.saturating_add(1);

        // Trim without renumbering IDs: only drop from ends.
        let mut plain = next.plain_messages();
        let (leading, trailing) = trim_to_complete_groups(&mut plain);
        if leading + trailing > 0 {
            if leading > 0 {
                next.messages.drain(..leading);
            }
            if trailing > 0 {
                let keep = next.messages.len() - trailing;
                next.messages.truncate(keep);
            }
        }

        let bytes = serde_json::to_vec_pretty(&next)
            .map_err(|error| AgentError::Storage(format!("serialize snapshot: {error}")))?;
        block_on(atomic_replace(
            &self.snapshot_path,
            &bytes,
            AtomicWriteOptions::default(),
            &CancellationToken::new(),
        ))?;
        Ok(next)
    }

    /// Read the v2 snapshot if present.
    ///
    /// A malformed or future-version snapshot is an error rather than a silent
    /// reset, so conversation data is never discarded implicitly.
    fn read_snapshot(&self) -> Result<Option<ConversationSnapshotV2>> {
        let raw = match fs::read_to_string(&self.snapshot_path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(AgentError::Storage(format!("read snapshot: {error}")));
            }
        };

        let snapshot: ConversationSnapshotV2 = serde_json::from_str(&raw).map_err(|error| {
            AgentError::Storage(format!(
                "conversation snapshot at {} is malformed: {error}",
                self.snapshot_path.display()
            ))
        })?;

        // Only the exact supported version is accepted. An unknown version is
        // never silently restamped as v2 by the next write.
        if snapshot.schema_version != CONVERSATION_SCHEMA_VERSION {
            return Err(AgentError::Storage(format!(
                "conversation snapshot schema version {} is not the supported version {}",
                snapshot.schema_version, CONVERSATION_SCHEMA_VERSION
            )));
        }

        let mut seen: Vec<u64> = Vec::with_capacity(snapshot.messages.len());
        for persisted in &snapshot.messages {
            if seen.contains(&persisted.id) {
                return Err(AgentError::Storage(format!(
                    "conversation snapshot reuses message id {}",
                    persisted.id
                )));
            }
            if persisted.id >= snapshot.next_message_id {
                return Err(AgentError::Storage(format!(
                    "conversation snapshot message id {} is not below next_message_id {}",
                    persisted.id, snapshot.next_message_id
                )));
            }
            seen.push(persisted.id);
        }
        Ok(Some(snapshot))
    }

    /// Import legacy JSONL once and write it as a v2 snapshot.
    pub fn import_legacy(
        &self,
        max_messages: usize,
    ) -> Result<(ConversationSnapshotV2, MigrationReport)> {
        let mut report = MigrationReport::default();

        let file = match fs::File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok((ConversationSnapshotV2::default(), report));
            }
            Err(error) => {
                return Err(AgentError::Storage(format!("open legacy history: {error}")));
            }
        };

        let mut messages: Vec<Message> = Vec::new();
        for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line_number = index + 1;
            let line =
                line.map_err(|error| AgentError::Storage(format!("read legacy history: {error}")))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Message>(trimmed) {
                Ok(message) => messages.push(message),
                Err(_) => report.malformed_lines.push(line_number),
            }
        }

        // Never hand a provider an orphan result or an unresolved tool call.
        let (leading, trailing) = trim_to_complete_groups(&mut messages);
        report.trimmed_leading_orphans = leading;
        report.trimmed_incomplete = trailing;

        let before_window = messages.len();
        apply_bounded_window(&mut messages, max_messages);
        report.trimmed_leading_orphans += before_window
            .saturating_sub(messages.len())
            .saturating_sub(before_window.saturating_sub(max_messages));

        let mut snapshot = ConversationSnapshotV2::default();
        snapshot.extend(&messages);
        report.imported = snapshot.messages.len();

        // Write v2 first: if this fails the legacy file is untouched.
        let saved = self.save_snapshot(&snapshot)?;

        // Retain the source under a unique name; it is never deleted.
        report.backup_path = Some(self.retain_legacy_backup()?);
        Ok((saved, report))
    }

    /// Rename the legacy file to a unique retained backup.
    fn retain_legacy_backup(&self) -> Result<PathBuf> {
        let base = self.path.with_extension("jsonl.pre-v2-backup");
        // Claim the name by exclusive creation rather than checking `exists()`
        // and then renaming. That check-then-act window let a name appear in
        // between, and `rename` would have silently replaced it, destroying an
        // earlier backup. Creating an empty placeholder first means only this
        // process owns the name it is about to move onto.
        for attempt in 0..=1000u32 {
            let candidate = if attempt == 0 {
                base.clone()
            } else {
                base.with_extension(format!("pre-v2-backup.{attempt}"))
            };
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(placeholder) => {
                    // Close the handle before renaming onto it; Windows can
                    // refuse to replace a file that is still open.
                    drop(placeholder);
                    // Replacing our own placeholder is safe and race-free.
                    match fs::rename(&self.path, &candidate) {
                        Ok(()) => return Ok(candidate),
                        Err(error) => {
                            // Do not leave an empty file at a retained-backup
                            // name: `retained_backups()` would report it as a
                            // preserved backup that holds nothing.
                            let _ = fs::remove_file(&candidate);
                            return Err(AgentError::Storage(format!(
                                "retain legacy history: {error}"
                            )));
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(AgentError::Storage(format!(
                        "claim legacy history backup '{}': {error}",
                        candidate.display()
                    )));
                }
            }
        }
        Err(AgentError::Storage(
            "cannot find a unique legacy history backup name".into(),
        ))
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

        // Keep only the most recent messages, without cutting through a tool
        // transaction. A raw count cut can leave a leading `ToolResult` whose
        // `ToolUse` was dropped, which a provider rejects on the next turn.
        apply_bounded_window(&mut messages, max);

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
    ///
    /// The retained messages are serialized in full before anything is
    /// replaced. Truncating the destination first, as this previously did, left
    /// a partially rewritten history if the process stopped mid-loop.
    pub fn compact(&self, max: usize) -> Result<()> {
        let messages = self.load(Some(max))?;
        let mut buffer = String::new();
        for msg in &messages {
            let json = serde_json::to_string(msg)
                .map_err(|e| AgentError::Storage(format!("serialize: {e}")))?;
            buffer.push_str(&json);
            buffer.push('\n');
        }
        atomic_replace_blocking(
            &self.path,
            buffer.as_bytes(),
            AtomicWriteOptions::default(),
            &CancellationToken::new(),
        )
        .map_err(|error| AgentError::Storage(format!("compact history: {error}")))
    }

    /// Clear all active conversation history (e.g. user command `/clear`).
    ///
    /// Both the v2 snapshot and the legacy JSONL store are removed. Long-term
    /// memory (`MEMORY.md`, `SOUL.md`, `HEARTBEAT.md`) is a separate contract and
    /// is never touched here.
    ///
    /// Cross-file removal cannot be a single atomic operation, so the order is
    /// chosen to fail safe: the legacy store is removed first, because deleting
    /// the authoritative v2 snapshot first would let a surviving legacy file be
    /// re-imported as "restored" conversation data. Any failure propagates and
    /// success is reported only after every active store is gone.
    pub fn clear(&self) -> Result<ClearReport> {
        let mut report = ClearReport::default();

        // Legacy first: see the fail-safe ordering note above.
        for (path, context) in [
            (&self.path, "clear legacy history"),
            (&self.snapshot_path, "clear conversation snapshot"),
        ] {
            match remove_if_present(path, context) {
                Ok(true) => report.removed.push(path.clone()),
                Ok(false) => {}
                // Surface what was already removed so a caller never sees a
                // partial clear described as if nothing had changed.
                Err(error) => return Err(partial_clear_error(error, &report)),
            }
        }

        // Confirm rather than assume: report success only when nothing active
        // remains readable.
        for path in [&self.path, &self.snapshot_path] {
            let still_present = path
                .try_exists()
                .map_err(|error| AgentError::Storage(format!("verify clear: {error}")))?;
            if still_present {
                return Err(partial_clear_error(
                    AgentError::Storage(format!("{} still present after clear", path.display())),
                    &report,
                ));
            }
        }

        report.retained_backups = self.retained_backups();
        Ok(report)
    }

    /// Migration backups retained by a previous legacy import.
    ///
    /// These are never read back by [`load_snapshot`](Self::load_snapshot), so
    /// they cannot restore cleared conversation data, but they remain on disk
    /// and are reported so a caller can disclose them.
    fn retained_backups(&self) -> Vec<PathBuf> {
        let Some(parent) = self.path.parent() else {
            return Vec::new();
        };
        let Ok(entries) = fs::read_dir(parent) else {
            return Vec::new();
        };
        let mut backups: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("HISTORY.jsonl.pre-v2-backup")
                            || name.starts_with("HISTORY.pre-v2-backup")
                    })
            })
            .collect();
        backups.sort();
        backups
    }

    /// Get the file path (for display/debug).
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Describe a failed clear without hiding what was already removed.
///
/// A caller must be able to tell "nothing changed" apart from "one store is
/// already gone", because the second case needs a retry rather than a retry-safe
/// no-op assumption.
fn partial_clear_error(error: AgentError, report: &ClearReport) -> AgentError {
    if report.removed.is_empty() {
        return error;
    }
    let removed: Vec<String> = report
        .removed
        .iter()
        .map(|path| path.display().to_string())
        .collect();
    AgentError::Storage(format!(
        "{error}; clear was partial and already removed: {}",
        removed.join(", ")
    ))
}

/// Remove a file, treating absence as success and any other error as failure.
///
/// Returns whether a file was actually removed.
fn remove_if_present(path: &Path, context: &str) -> Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(AgentError::Storage(format!("{context}: {error}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_types::{ContentBlock, Role};

    fn sample_messages(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| Message {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
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
    fn clear_removes_both_active_stores_and_prevents_restore() {
        // **Validates: Requirements 2.30**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        history.append(&sample_messages(4)).unwrap();
        let mut snapshot = ConversationSnapshotV2::default();
        snapshot.extend(&sample_messages(2));
        snapshot.compacted_summary = ConversationSummary {
            open_tasks: vec!["must not survive".into()],
            ..ConversationSummary::default()
        };
        history.save_snapshot(&snapshot).unwrap();

        let report = history.clear().unwrap();
        assert!(report.removed.contains(&history.path().to_path_buf()));
        assert!(report
            .removed
            .contains(&history.snapshot_path().to_path_buf()));
        assert!(!history.path().exists());
        assert!(!history.snapshot_path().exists());

        // Nothing is restored, including the compacted summary.
        let (restored, migration) = history.load_snapshot(None).unwrap();
        assert!(migration.is_none());
        assert!(restored.messages.is_empty());
        assert!(restored.compacted_summary.is_empty());
        assert!(history.load(None).unwrap().is_empty());
    }

    #[test]
    fn clear_on_empty_workspace_succeeds_without_creating_files() {
        // **Validates: Requirements 2.30**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let report = history.clear().unwrap();
        assert!(report.removed.is_empty());
        assert!(!history.path().exists());
        assert!(!history.snapshot_path().exists());
    }

    #[test]
    fn clear_leaves_long_term_memory_untouched() {
        // **Validates: Requirements 2.30, 3.12**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();
        let agent_dir = dir.path().join(".agent");

        let memory_files = ["MEMORY.md", "SOUL.md", "HEARTBEAT.md"];
        for name in memory_files {
            fs::write(agent_dir.join(name), format!("durable {name}")).unwrap();
        }
        history.append(&sample_messages(2)).unwrap();

        history.clear().unwrap();

        for name in memory_files {
            let path = agent_dir.join(name);
            assert!(path.exists(), "{name} must survive /clear");
            assert_eq!(
                fs::read_to_string(&path).unwrap(),
                format!("durable {name}")
            );
        }
    }

    #[test]
    fn clear_propagates_a_non_missing_io_error_instead_of_reporting_success() {
        // **Validates: Requirements 2.30**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        // A directory at the snapshot path cannot be removed with `remove_file`,
        // which is a non-NotFound error and must not be treated as cleared.
        fs::create_dir(history.snapshot_path()).unwrap();
        history.append(&sample_messages(2)).unwrap();

        let error = history.clear().unwrap_err().to_string();
        assert!(history.snapshot_path().exists());
        // The legacy store was already removed, so the error must say so.
        assert!(
            error.contains("partial"),
            "partial clear must be disclosed: {error}"
        );
    }

    #[test]
    fn clear_reports_a_retained_migration_backup_that_is_never_restored() {
        // **Validates: Requirements 2.30**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let legacy = serde_json::to_string(&text("pre-migration")).unwrap();
        fs::write(history.path(), format!("{legacy}\n")).unwrap();
        let (_snapshot, report) = history.load_snapshot(None).unwrap();
        let backup = report.unwrap().backup_path.unwrap();

        let cleared = history.clear().unwrap();
        assert!(cleared.retained_backups.contains(&backup));
        assert!(backup.exists());

        // The backup is not an active store, so it cannot restore the data.
        let (restored, migration) = history.load_snapshot(None).unwrap();
        assert!(migration.is_none());
        assert!(restored.messages.is_empty());
    }

    #[test]
    fn missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();
        let loaded = history.load(None).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    fn tool_use(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "a.rs"}),
                provider_metadata: None,
            }],
            token_estimate: 1,
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
            token_estimate: 1,
        }
    }

    #[test]
    fn snapshot_assigns_monotonic_ids_that_survive_compaction() {
        // **Validates: Requirements 2.31**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let mut snapshot = ConversationSnapshotV2::default();
        snapshot.extend(&sample_messages(3));
        let saved = history.save_snapshot(&snapshot).unwrap();
        assert_eq!(saved.generation, 1);
        assert_eq!(
            saved.messages.iter().map(|m| m.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        // Compaction drops earlier messages; later IDs must not be reused.
        let mut compacted = saved.clone();
        compacted.messages.drain(..2);
        compacted.compacted_summary = ConversationSummary {
            open_tasks: vec!["UNTRUSTED_SUMMARY".into()],
            ..ConversationSummary::default()
        };
        compacted.extend(&sample_messages(1));
        let saved = history.save_snapshot(&compacted).unwrap();

        let (reloaded, migration) = history.load_snapshot(None).unwrap();
        assert!(migration.is_none());
        assert_eq!(reloaded.generation, saved.generation);
        assert_eq!(reloaded.generation, 2);
        assert_eq!(
            reloaded.messages.iter().map(|m| m.id).collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert_eq!(reloaded.next_message_id, 5);
        assert_eq!(
            reloaded.compacted_summary.open_tasks,
            vec!["UNTRUSTED_SUMMARY".to_string()]
        );
    }

    #[test]
    fn legacy_import_reports_malformed_lines_and_retains_a_backup() {
        // **Validates: Requirements 2.31, 2.39, 3.12**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let valid = serde_json::to_string(&sample_messages(1)[0]).unwrap();
        let legacy = format!("{valid}\n{{not json\n\n{valid}\n");
        fs::write(history.path(), legacy).unwrap();

        let (snapshot, report) = history.import_legacy(MAX_HISTORY_MESSAGES).unwrap();
        assert_eq!(report.imported, 2);
        assert_eq!(report.malformed_lines, vec![2]);
        assert_eq!(snapshot.schema_version, CONVERSATION_SCHEMA_VERSION);

        // The source is retained, never deleted.
        let backup = report.backup_path.unwrap();
        assert!(backup.exists());
        assert!(!history.path().exists());
        assert!(history.snapshot_path().exists());
    }

    #[test]
    fn legacy_import_trims_only_incomplete_trailing_tool_groups() {
        // **Validates: Requirements 2.39**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let messages = [
            sample_messages(1).remove(0),
            tool_use("call-1"),
            tool_result("call-1"),
            tool_use("call-2"), // unresolved: must be trimmed
        ];
        let legacy: String = messages
            .iter()
            .map(|message| format!("{}\n", serde_json::to_string(message).unwrap()))
            .collect();
        fs::write(history.path(), legacy).unwrap();

        let (snapshot, report) = history.import_legacy(MAX_HISTORY_MESSAGES).unwrap();
        assert_eq!(report.trimmed_incomplete, 1);
        assert_eq!(snapshot.messages.len(), 3);
        assert!(!snapshot.plain_messages().iter().any(|message| {
            message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { id, .. } if id == "call-2"))
        }));
    }

    #[test]
    fn present_snapshot_wins_and_legacy_is_not_reimported() {
        // **Validates: Requirements 2.31**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let mut snapshot = ConversationSnapshotV2::default();
        snapshot.extend(&[text("from-v2")]);
        history.save_snapshot(&snapshot).unwrap();

        let legacy = serde_json::to_string(&text("from-legacy")).unwrap();
        fs::write(history.path(), format!("{legacy}\n")).unwrap();

        let (loaded, migration) = history.load_snapshot(None).unwrap();
        assert!(migration.is_none());
        assert_eq!(loaded.messages.len(), 1);
        assert!(matches!(
            &loaded.messages[0].message.content[0],
            ContentBlock::Text(value) if value == "from-v2"
        ));
        // The legacy file is left untouched because it was not imported.
        assert!(history.path().exists());
    }

    #[test]
    fn malformed_or_future_snapshot_is_an_error_not_a_silent_reset() {
        // **Validates: Requirements 2.31**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        fs::write(history.snapshot_path(), "{not json").unwrap();
        assert!(history.load_snapshot(None).is_err());

        for version in [CONVERSATION_SCHEMA_VERSION + 1, 1, 0] {
            fs::write(
                history.snapshot_path(),
                serde_json::json!({
                    "schema_version": version,
                    "generation": 9,
                    "next_message_id": 9,
                    "compacted_summary": "",
                    "messages": []
                })
                .to_string(),
            )
            .unwrap();
            assert!(
                history.load_snapshot(None).is_err(),
                "schema version {version} must not load as v2"
            );
        }

        // A structured summary field round-trips and defaults when absent.
        let typed = ConversationSummary {
            user_requests: vec!["ask".into()],
            ..ConversationSummary::default()
        };
        let snapshot = ConversationSnapshotV2 {
            compacted_summary: typed.clone(),
            ..ConversationSnapshotV2::default()
        };
        history.save_snapshot(&snapshot).unwrap();
        assert_eq!(
            history.load_snapshot(None).unwrap().0.compacted_summary,
            typed
        );

        // A snapshot missing `messages` is corrupt, not an empty conversation.
        fs::write(
            history.snapshot_path(),
            serde_json::json!({
                "schema_version": CONVERSATION_SCHEMA_VERSION,
                "generation": 3,
                "next_message_id": 4
            })
            .to_string(),
        )
        .unwrap();
        assert!(history.load_snapshot(None).is_err());

        // Reused or out-of-range IDs are rejected rather than silently extended.
        for ids in [vec![1, 1], vec![1, 9]] {
            let messages: Vec<_> = ids
                .iter()
                .map(|id| serde_json::json!({"id": id, "message": text("x")}))
                .collect();
            fs::write(
                history.snapshot_path(),
                serde_json::json!({
                    "schema_version": CONVERSATION_SCHEMA_VERSION,
                    "generation": 1,
                    "next_message_id": 3,
                    "messages": messages
                })
                .to_string(),
            )
            .unwrap();
            assert!(
                history.load_snapshot(None).is_err(),
                "ids {ids:?} must be rejected"
            );
        }
    }

    #[test]
    fn fresh_project_reports_no_migration() {
        // **Validates: Requirements 2.31**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let (snapshot, migration) = history.load_snapshot(None).unwrap();
        assert!(migration.is_none());
        assert!(snapshot.messages.is_empty());
        assert_eq!(snapshot.generation, 0);
        // A read must not create either store.
        assert!(!history.snapshot_path().exists());
        assert!(!history.path().exists());
    }

    #[test]
    fn bounded_restore_never_exposes_an_orphan_tool_result() {
        // **Validates: Requirements 2.39, 3.12**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        let mut snapshot = ConversationSnapshotV2::default();
        snapshot.extend(&[text("earlier"), tool_use("call-1"), tool_result("call-1")]);
        history.save_snapshot(&snapshot).unwrap();

        // A cap of one would cut through the tool transaction.
        let (bounded, _) = history.load_snapshot(Some(1)).unwrap();
        assert!(!bounded.plain_messages().iter().any(|message| {
            message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
        }));
        // Retained IDs are never renumbered.
        for persisted in &bounded.messages {
            assert!(persisted.id < bounded.next_message_id);
        }
    }

    #[test]
    fn legacy_import_drops_a_leading_orphan_tool_result() {
        // **Validates: Requirements 2.39**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        // The legacy file begins mid-transaction: the ToolUse was pruned earlier.
        let messages = [
            tool_result("call-0"),
            text("later"),
            tool_use("call-1"),
            tool_result("call-1"),
        ];
        let legacy: String = messages
            .iter()
            .map(|message| format!("{}\n", serde_json::to_string(message).unwrap()))
            .collect();
        fs::write(history.path(), legacy).unwrap();

        let (snapshot, report) = history.import_legacy(MAX_HISTORY_MESSAGES).unwrap();
        assert_eq!(report.trimmed_leading_orphans, 1);
        assert_eq!(snapshot.messages.len(), 3);
        assert!(matches!(
            &snapshot.messages[0].message.content[0],
            ContentBlock::Text(value) if value == "later"
        ));
    }

    fn text(value: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text(value.into())],
            token_estimate: 1,
        }
    }

    #[test]
    fn legacy_load_append_and_clear_remain_available() {
        // **Validates: Requirements 3.12**
        let dir = tempfile::tempdir().unwrap();
        let history = ChatHistory::open(dir.path()).unwrap();

        history.append(&sample_messages(3)).unwrap();
        assert_eq!(history.load(None).unwrap().len(), 3);
        history.compact(2).unwrap();
        assert_eq!(history.load(None).unwrap().len(), 2);
        history.clear().unwrap();
        assert_eq!(history.load(None).unwrap().len(), 0);
    }
}
