//! Evaluation run identity.
//!
//! Two runs started in the same second must never share a results file. A run
//! ID therefore combines a timestamp with random entropy, and the results file
//! is created exclusively so a collision is detected rather than silently
//! merged. Appending is only possible through the [`RunHandle`] that created the
//! file, which removes implicit append/resume of some other run's results.
//!
//! Legacy timestamp-only result files stay readable; only creation is stricter.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_types::{AgentError, Result};

use crate::swebench::EvalOutcome;

/// Prefix every generated run ID carries.
pub const RUN_ID_PREFIX: &str = "run-";

/// Extension of a results file.
pub const RESULTS_EXTENSION: &str = "jsonl";

/// How many distinct IDs to try before giving up on a collision.
const MAX_ID_ATTEMPTS: usize = 16;

/// Distinguishes IDs generated within the same process and nanosecond.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Random, non-sequential suffix for a run ID.
///
/// `RandomState` is seeded per process by the OS, so two concurrently started
/// processes do not produce the same suffix even at the same timestamp. This is
/// collision resistance, not a security boundary.
fn random_suffix() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let mut hasher = RandomState::new().build_hasher();
    std::process::id().hash(&mut hasher);
    SEQUENCE.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Build a fresh run ID from the current time plus random entropy.
pub fn new_run_id() -> String {
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.6fZ");
    format!("{RUN_ID_PREFIX}{stamp}-{}", random_suffix())
}

/// Reject a run ID that could escape the results directory or name a file the
/// caller did not intend.
///
/// Applied when resolving a user-supplied ID, so `--run-a ../../secrets` cannot
/// read outside the results directory.
pub fn validate_run_id(run_id: &str) -> Result<()> {
    let invalid = |reason: &str| {
        Err(AgentError::Tool {
            name: "evals".into(),
            reason: format!("invalid run id '{run_id}': {reason}"),
        })
    };

    if run_id.is_empty() {
        return invalid("empty");
    }
    if run_id.len() > 128 {
        return invalid("longer than 128 characters");
    }
    if run_id == "." || run_id == ".." || run_id.contains("..") {
        return invalid("contains a parent-directory segment");
    }
    if !run_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return invalid("contains characters outside [A-Za-z0-9._-]");
    }
    // Windows resolves these names to devices regardless of extension, so a run
    // ID of `CON` would open a console handle instead of a results file. Reject
    // them on every platform so behavior does not diverge.
    const RESERVED: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    let stem = run_id.split('.').next().unwrap_or(run_id);
    if RESERVED
        .iter()
        .any(|reserved| stem.eq_ignore_ascii_case(reserved))
    {
        return invalid("is a reserved device name");
    }
    Ok(())
}

/// Resolve the results file for an existing run ID.
pub fn results_path(results_dir: &Path, run_id: &str) -> Result<PathBuf> {
    validate_run_id(run_id)?;
    Ok(results_dir.join(format!("{run_id}.{RESULTS_EXTENSION}")))
}

/// An exclusively created results file owned by one run.
#[derive(Clone, Debug)]
pub struct RunHandle {
    run_id: String,
    path: PathBuf,
}

impl RunHandle {
    /// Identity of this run.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Results file this run owns.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one outcome to this run's own results file.
    ///
    /// The file is opened for append without `create`, because this handle
    /// already created it. A missing file therefore surfaces as an error instead
    /// of silently starting a second file under the same identity.
    pub fn append(&self, outcome: &EvalOutcome) -> Result<()> {
        let line = serde_json::to_string(outcome).map_err(|error| AgentError::Tool {
            name: "evals".into(),
            reason: format!("serialize outcome: {error}"),
        })?;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|error| AgentError::Tool {
                name: "evals".into(),
                reason: format!("open results {}: {error}", self.path.display()),
            })?;
        writeln!(file, "{line}").map_err(|error| AgentError::Tool {
            name: "evals".into(),
            reason: format!("write results {}: {error}", self.path.display()),
        })?;
        Ok(())
    }
}

/// Create a new run with an exclusively created, empty results file.
///
/// A losing race on the filename produces a new ID and retries rather than
/// appending to the winner's file, so two runs can never merge.
pub fn create_exclusive_run(results_dir: &Path) -> Result<RunHandle> {
    std::fs::create_dir_all(results_dir).map_err(|error| AgentError::Tool {
        name: "evals".into(),
        reason: format!("create results dir {}: {error}", results_dir.display()),
    })?;

    let mut last_conflict = None;
    for _ in 0..MAX_ID_ATTEMPTS {
        let run_id = new_run_id();
        let path = results_dir.join(format!("{run_id}.{RESULTS_EXTENSION}"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_file) => return Ok(RunHandle { run_id, path }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_conflict = Some(path);
                continue;
            }
            Err(error) => {
                return Err(AgentError::Tool {
                    name: "evals".into(),
                    reason: format!("create results {}: {error}", path.display()),
                });
            }
        }
    }

    Err(AgentError::Tool {
        name: "evals".into(),
        reason: format!(
            "could not create a unique results file after {MAX_ID_ATTEMPTS} attempts (last conflict: {})",
            last_conflict
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "unknown".into())
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(case_id: &str) -> EvalOutcome {
        EvalOutcome {
            case_id: case_id.into(),
            passed: true,
            turns: 0,
            tool_calls: 0,
            tokens_in: 0,
            tokens_out: 0,
            wall_time_ms: 1,
            error: None,
        }
    }

    #[test]
    fn generated_ids_are_unique_within_the_same_second() {
        // **Validates: Requirements 2.17**
        let ids: std::collections::BTreeSet<String> = (0..256).map(|_| new_run_id()).collect();
        assert_eq!(ids.len(), 256, "timestamp-only IDs would collide here");
        assert!(ids.iter().all(|id| id.starts_with(RUN_ID_PREFIX)));
        assert!(ids.iter().all(|id| validate_run_id(id).is_ok()));
    }

    #[test]
    fn concurrent_runs_never_share_a_results_file() {
        // **Validates: Requirements 2.17**
        let dir = tempfile::tempdir().unwrap();
        let handles: Vec<RunHandle> = (0..8)
            .map(|_| create_exclusive_run(dir.path()).unwrap())
            .collect();

        let paths: std::collections::BTreeSet<&Path> =
            handles.iter().map(|handle| handle.path()).collect();
        assert_eq!(paths.len(), 8, "each run must own a distinct file");

        for (index, handle) in handles.iter().enumerate() {
            handle.append(&outcome(&format!("case-{index}"))).unwrap();
        }
        for handle in &handles {
            let content = std::fs::read_to_string(handle.path()).unwrap();
            assert_eq!(
                content.lines().count(),
                1,
                "a run's file must contain only its own outcomes"
            );
        }
    }

    #[test]
    fn appending_without_an_owned_file_is_refused() {
        // **Validates: Requirements 2.17**
        // Implicit resume is what let two runs merge, so a handle whose file is
        // gone must fail rather than recreate it.
        let dir = tempfile::tempdir().unwrap();
        let handle = create_exclusive_run(dir.path()).unwrap();
        std::fs::remove_file(handle.path()).unwrap();
        assert!(handle.append(&outcome("orphan")).is_err());
        assert!(!handle.path().exists());
    }

    #[test]
    fn run_ids_that_could_escape_the_results_directory_are_rejected() {
        // **Validates: Requirements 2.17**
        for candidate in [
            "",
            ".",
            "..",
            "../secrets",
            "..\\secrets",
            "nested/run",
            "nested\\run",
            "run id",
            "run\0id",
            "CON",
            "nul",
            "Com1.jsonl",
        ] {
            assert!(
                validate_run_id(candidate).is_err(),
                "must reject {candidate:?}"
            );
            assert!(results_path(Path::new("results"), candidate).is_err());
        }

        // A legacy timestamp-only ID stays readable.
        assert!(validate_run_id("run-1750000000").is_ok());
        assert_eq!(
            results_path(Path::new("results"), "run-1750000000").unwrap(),
            Path::new("results").join("run-1750000000.jsonl")
        );
    }
}
