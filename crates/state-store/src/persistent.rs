//! Persistent state mapped to on-disk markdown files:
//! - SOUL.md: persona + policies (contains `language:` line)
//! - HEARTBEAT.md: task list with `- [ ]` / `- [x]` checkboxes
//! - MEMORY.md: append-only memory with ISO-8601 timestamps

use std::fs;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread;

use agent_types::{AgentError, LanguageMode, Result};
use runtime_core::{atomic_replace, AtomicWriteOptions};
use tokio_util::sync::CancellationToken;

struct ThreadWaker(thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Drive a future to completion on the current thread.
///
/// Shared with conversation snapshot writes so both persistence paths use the
/// same shared atomic writer from a synchronous API.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}

fn read_optional_utf8(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(content) => Ok(Some(content)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn line_body_and_ending(line: &str) -> (&str, &str) {
    if let Some(body) = line.strip_suffix("\r\n") {
        (body, "\r\n")
    } else if let Some(body) = line.strip_suffix('\n') {
        (body, "\n")
    } else {
        (line, "")
    }
}

fn newline_style(content: &str) -> &'static str {
    if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

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
        for path in [
            state.soul_path(),
            state.heartbeat_path(),
            state.memory_path(),
        ] {
            // Claim the name exclusively instead of testing `exists()` and then
            // writing. The old sequence could truncate a file that another
            // process created in the window between the two calls, and an
            // empty SOUL.md/HEARTBEAT.md/MEMORY.md is indistinguishable from a
            // deliberately cleared one. `AlreadyExists` is the normal case on
            // every run after the first, so only that error is ignored.
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
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

    fn atomic_write(&self, path: &Path, content: &str, options: AtomicWriteOptions) -> Result<()> {
        block_on(atomic_replace(
            path,
            content.as_bytes(),
            options,
            &CancellationToken::new(),
        ))
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
        let content = read_optional_utf8(&path)?.unwrap_or_default();
        let mode = match mode {
            LanguageMode::En => "en",
            LanguageMode::Hinglish => "hinglish",
        };
        let ending = newline_style(&content);
        let mut found = false;
        let mut updated = String::with_capacity(content.len() + 32);

        for line in content.split_inclusive('\n') {
            let (body, line_ending) = line_body_and_ending(line);
            let trimmed = body.trim_start();
            if trimmed.starts_with("language:") {
                found = true;
                let indentation = &body[..body.len() - trimmed.len()];
                updated.push_str(indentation);
                updated.push_str("language: ");
                updated.push_str(mode);
                updated.push_str(line_ending);
            } else {
                updated.push_str(line);
            }
        }

        if !found {
            updated.clear();
            let mut inserted = false;
            for line in content.split_inclusive('\n') {
                let (body, line_ending) = line_body_and_ending(line);
                updated.push_str(line);
                if !inserted && body.trim() == "## Policies" {
                    if line_ending.is_empty() {
                        updated.push_str(ending);
                    }
                    updated.push_str("language: ");
                    updated.push_str(mode);
                    updated.push_str(ending);
                    inserted = true;
                }
            }

            if !inserted {
                updated = content;
                if !updated.is_empty() && !updated.ends_with(['\n', '\r']) {
                    updated.push_str(ending);
                }
                updated.push_str("## Policies");
                updated.push_str(ending);
                updated.push_str("language: ");
                updated.push_str(mode);
                updated.push_str(ending);
            }
        }

        self.atomic_write(&path, &updated, AtomicWriteOptions::default())
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
                } else {
                    trimmed
                        .strip_prefix("- [ ]")
                        .map(|rest| (false, rest.trim().to_string()))
                }
            })
            .collect()
    }

    /// Set a task done by index (0-based). Surgical line edit.
    pub fn set_task_done(&self, idx: usize) -> Result<()> {
        self.set_task_done_with_options(idx, AtomicWriteOptions::default())
    }

    fn set_task_done_with_options(&self, idx: usize, options: AtomicWriteOptions) -> Result<()> {
        let path = self.heartbeat_path();
        let content = read_optional_utf8(&path)?.unwrap_or_default();
        let mut task_idx = 0usize;
        let mut found = false;
        let mut changed = false;
        let mut updated = String::with_capacity(content.len());

        for line in content.split_inclusive('\n') {
            let (body, line_ending) = line_body_and_ending(line);
            let trimmed = body.trim_start();
            let is_open = trimmed.starts_with("- [ ]");
            let is_done = trimmed.starts_with("- [x]");

            if is_open || is_done {
                if task_idx == idx {
                    found = true;
                    if is_open {
                        let marker = body.find("- [ ]").expect("matched checkbox marker");
                        updated.push_str(&body[..marker]);
                        updated.push_str("- [x]");
                        updated.push_str(&body[marker + 5..]);
                        updated.push_str(line_ending);
                        changed = true;
                    } else {
                        updated.push_str(line);
                    }
                } else {
                    updated.push_str(line);
                }
                task_idx += 1;
            } else {
                updated.push_str(line);
            }
        }

        if !found {
            return Err(AgentError::Storage(format!(
                "task index {idx} not found in {}",
                path.display()
            )));
        }
        if !changed {
            return Ok(());
        }

        self.atomic_write(&path, &updated, options)
    }

    // === Memory ===

    /// Append a timestamped memory entry.
    pub fn append_memory(&self, entry: &str) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let path = self.memory_path();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
        let line = format!("\n[{timestamp}] {entry}\n");

        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
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

    // **Validates: Requirements 2.15, 3.12**
    #[test]
    fn missing_files_default_only_when_the_requested_update_is_valid() {
        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        fs::remove_file(state.soul_path()).expect("remove SOUL.md");
        fs::remove_file(state.heartbeat_path()).expect("remove HEARTBEAT.md");

        state
            .set_language_mode(LanguageMode::Hinglish)
            .expect("a missing SOUL.md may be initialized");
        assert_eq!(
            fs::read_to_string(state.soul_path()).expect("created SOUL.md"),
            "## Policies\nlanguage: hinglish\n"
        );

        let result = state.set_task_done(0);
        assert!(result.is_err(), "a missing task list has no index zero");
        assert!(
            !state.heartbeat_path().exists(),
            "an absent task index must be rejected before writing"
        );
    }

    // **Validates: Requirement 2.15**
    #[test]
    fn malformed_utf8_is_propagated_without_overwriting_either_file() {
        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        let malformed = [0xff, 0xfe, 0xfd];

        fs::write(state.soul_path(), malformed).expect("malformed SOUL.md");
        let soul_result = state.set_language_mode(LanguageMode::Hinglish);
        assert!(matches!(
            soul_result,
            Err(AgentError::Io(ref error)) if error.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(
            fs::read(state.soul_path()).expect("SOUL.md bytes"),
            malformed
        );

        fs::write(state.heartbeat_path(), malformed).expect("malformed HEARTBEAT.md");
        let heartbeat_result = state.set_task_done(0);
        assert!(matches!(
            heartbeat_result,
            Err(AgentError::Io(ref error)) if error.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(
            fs::read(state.heartbeat_path()).expect("HEARTBEAT.md bytes"),
            malformed
        );
    }

    // **Validates: Requirement 2.15**
    #[test]
    fn non_not_found_io_errors_are_propagated_without_replacing_the_path() {
        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        let soul = state.soul_path();
        fs::remove_file(&soul).expect("remove SOUL.md");
        fs::create_dir(&soul).expect("directory at SOUL.md path");

        let result = state.set_language_mode(LanguageMode::Hinglish);
        assert!(matches!(result, Err(AgentError::Io(_))));
        assert!(soul.is_dir(), "the failing path must remain untouched");
    }

    #[cfg(unix)]
    // **Validates: Requirement 2.15**
    #[test]
    fn unreadable_files_are_not_treated_as_empty_on_unix() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        let soul = b"## Policies\nlanguage: en\n";
        let heartbeat = b"# Tasks\n- [ ] keep me\n";
        fs::write(state.soul_path(), soul).expect("SOUL.md fixture");
        fs::write(state.heartbeat_path(), heartbeat).expect("HEARTBEAT.md fixture");
        fs::set_permissions(state.soul_path(), fs::Permissions::from_mode(0o000))
            .expect("make SOUL.md unreadable");
        fs::set_permissions(state.heartbeat_path(), fs::Permissions::from_mode(0o000))
            .expect("make HEARTBEAT.md unreadable");

        let soul_result = state.set_language_mode(LanguageMode::Hinglish);
        let heartbeat_result = state.set_task_done(0);

        fs::set_permissions(state.soul_path(), fs::Permissions::from_mode(0o600))
            .expect("restore SOUL.md permissions");
        fs::set_permissions(state.heartbeat_path(), fs::Permissions::from_mode(0o600))
            .expect("restore HEARTBEAT.md permissions");
        assert!(matches!(soul_result, Err(AgentError::Io(_))));
        assert!(matches!(heartbeat_result, Err(AgentError::Io(_))));
        assert_eq!(fs::read(state.soul_path()).expect("SOUL.md bytes"), soul);
        assert_eq!(
            fs::read(state.heartbeat_path()).expect("HEARTBEAT.md bytes"),
            heartbeat
        );
    }

    #[cfg(windows)]
    // **Validates: Requirement 2.15**
    #[test]
    fn unreadable_files_are_not_treated_as_empty_on_windows() {
        use std::os::windows::fs::OpenOptionsExt;

        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        let soul = b"## Policies\r\nlanguage: en\r\n";
        let heartbeat = b"# Tasks\r\n- [ ] keep me\r\n";
        fs::write(state.soul_path(), soul).expect("SOUL.md fixture");
        fs::write(state.heartbeat_path(), heartbeat).expect("HEARTBEAT.md fixture");
        let soul_lock = fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(state.soul_path())
            .expect("lock SOUL.md against reads");
        let heartbeat_lock = fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(state.heartbeat_path())
            .expect("lock HEARTBEAT.md against reads");

        let soul_result = state.set_language_mode(LanguageMode::Hinglish);
        let heartbeat_result = state.set_task_done(0);
        assert!(matches!(soul_result, Err(AgentError::Io(_))));
        assert!(matches!(heartbeat_result, Err(AgentError::Io(_))));

        drop(soul_lock);
        drop(heartbeat_lock);
        assert_eq!(fs::read(state.soul_path()).expect("SOUL.md bytes"), soul);
        assert_eq!(
            fs::read(state.heartbeat_path()).expect("HEARTBEAT.md bytes"),
            heartbeat
        );
    }

    // **Validates: Requirements 2.15, 3.12**
    #[test]
    fn absent_and_present_task_indexes_have_explicit_write_behavior() {
        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        let original = "# Tasks\n- [ ] only task";
        fs::write(state.heartbeat_path(), original).expect("HEARTBEAT.md fixture");

        assert!(state.set_task_done(1).is_err());
        assert_eq!(
            fs::read_to_string(state.heartbeat_path()).expect("unchanged HEARTBEAT.md"),
            original
        );

        state.set_task_done(0).expect("present task index");
        assert_eq!(
            fs::read_to_string(state.heartbeat_path()).expect("updated HEARTBEAT.md"),
            "# Tasks\n- [x] only task"
        );
    }

    // **Validates: Requirement 2.15**
    #[test]
    fn atomic_writer_failure_preserves_the_destination() {
        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        let original = "# Tasks\n- [ ] destination survives\n";
        fs::write(state.heartbeat_path(), original).expect("HEARTBEAT.md fixture");
        let invalid_options = AtomicWriteOptions {
            replace_attempts: 0,
            ..AtomicWriteOptions::default()
        };

        let result = state.set_task_done_with_options(0, invalid_options);
        assert!(matches!(
            result,
            Err(AgentError::Io(ref error)) if error.kind() == std::io::ErrorKind::InvalidInput
        ));
        assert_eq!(
            fs::read_to_string(state.heartbeat_path()).expect("preserved HEARTBEAT.md"),
            original
        );
    }

    // **Validates: Requirements 2.15, 3.12**
    #[test]
    fn setters_change_only_the_target_markdown_tokens() {
        let dir = tempfile::tempdir().expect("temporary workspace");
        let state = PersistentState::load(dir.path()).expect("persistent state");
        let soul = "# Persona\r\nKeep **Markdown**.\r\n## Policies\r\n  language: en\r\n- preserve `code`\r\n";
        let heartbeat =
            "# Tasks\r\n\r\nNotes *stay*.\r\n  - [ ] first **bold**\r\n\t- [x] done `code`\r\n";
        fs::write(state.soul_path(), soul).expect("SOUL.md fixture");
        fs::write(state.heartbeat_path(), heartbeat).expect("HEARTBEAT.md fixture");

        state
            .set_language_mode(LanguageMode::Hinglish)
            .expect("set language");
        state.set_task_done(0).expect("complete task");

        assert_eq!(
            fs::read_to_string(state.soul_path()).expect("SOUL.md result"),
            "# Persona\r\nKeep **Markdown**.\r\n## Policies\r\n  language: hinglish\r\n- preserve `code`\r\n"
        );
        assert_eq!(
            fs::read_to_string(state.heartbeat_path()).expect("HEARTBEAT.md result"),
            "# Tasks\r\n\r\nNotes *stay*.\r\n  - [x] first **bold**\r\n\t- [x] done `code`\r\n"
        );
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
