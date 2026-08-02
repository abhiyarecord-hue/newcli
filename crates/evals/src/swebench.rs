//! SWE-bench-lite scripted runner.
//!
//! Loads eval cases from `.agent/evals/cases/*.toml`, runs each in isolation,
//! appends outcomes as JSONL.

use std::path::{Path, PathBuf};

use agent_types::{AgentError, Result};
use serde::{Deserialize, Serialize};

/// Suite selector meaning "every loaded case".
pub const ALL_SUITES: &str = "all";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCase {
    pub id: String,
    pub prompt: String,
    pub repo_fixture: PathBuf,
    pub check_cmd: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Suite this case belongs to.
    ///
    /// Omitted in most case files, where the file stem is the suite name. An
    /// explicit value wins so several suites can share one file if needed.
    #[serde(default)]
    pub suite: Option<String>,
}

fn default_timeout() -> u64 {
    120
}

/// Suite a case file belongs to: its explicit `suite`, else the file stem.
fn suite_of(path: &Path, case: &EvalCase) -> String {
    if let Some(suite) = case.suite.as_deref() {
        if !suite.trim().is_empty() {
            return suite.to_string();
        }
    }
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Load only the cases belonging to `suite`.
///
/// Filtering happens before anything is scheduled, so a case outside the
/// selected suite is never loaded into a run, counted, or executed. The
/// [`ALL_SUITES`] selector keeps every case.
pub fn load_suite(dir: &Path, suite: &str) -> Result<Vec<EvalCase>> {
    let loaded = load_cases_with_suites(dir)?;
    if suite == ALL_SUITES {
        return Ok(loaded.into_iter().map(|(_, case)| case).collect());
    }
    Ok(loaded
        .into_iter()
        .filter(|(case_suite, _)| case_suite == suite)
        .map(|(_, case)| case)
        .collect())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalOutcome {
    pub case_id: String,
    pub passed: bool,
    pub turns: u32,
    pub tool_calls: u32,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub wall_time_ms: u64,
    pub error: Option<String>,
}

/// Load every eval case from a directory of TOML files.
pub fn load_cases(dir: &Path) -> Result<Vec<EvalCase>> {
    Ok(load_cases_with_suites(dir)?
        .into_iter()
        .map(|(_, case)| case)
        .collect())
}

/// Load every case paired with the suite it belongs to.
fn load_cases_with_suites(dir: &Path) -> Result<Vec<(String, EvalCase)>> {
    if !dir.is_dir() {
        return Err(AgentError::Tool {
            name: "evals".into(),
            reason: format!("cases dir not found: {}", dir.display()),
        });
    }

    let mut cases = Vec::new();
    for entry in std::fs::read_dir(dir)
        .map_err(|e| AgentError::Tool {
            name: "evals".into(),
            reason: format!("read dir: {e}"),
        })?
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            let content = std::fs::read_to_string(&path).map_err(|e| AgentError::Tool {
                name: "evals".into(),
                reason: format!("read {}: {e}", path.display()),
            })?;
            let case: EvalCase = toml::from_str(&content).map_err(|e| AgentError::Tool {
                name: "evals".into(),
                reason: format!("parse {}: {e}", path.display()),
            })?;
            let suite = suite_of(&path, &case);
            cases.push((suite, case));
        }
    }
    // Directory order is not stable across platforms; sort so a run's case
    // order and its results file are reproducible.
    cases.sort_by(|(left_suite, left), (right_suite, right)| {
        (left_suite, &left.id).cmp(&(right_suite, &right.id))
    });
    Ok(cases)
}

/// Append one outcome as a JSONL line to an arbitrary results file.
///
/// **Never used for run output.** It creates the file if absent and appends to
/// whatever is already there, so pointing a run at it would let two runs merge —
/// the defect Task 23.2 fixed. A run claims its file through
/// [`crate::run::create_exclusive_run`] and writes via
/// [`crate::run::RunHandle::append`], which refuses to create a file it does not
/// own.
///
/// Retained as test-only rather than deleted so the JSONL line format stays
/// covered, and gated so it cannot be reached from production code again.
#[cfg(test)]
pub(crate) fn append_outcome(results_path: &Path, outcome: &EvalOutcome) -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;

    if let Some(parent) = results_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let line = serde_json::to_string(outcome).map_err(|e| AgentError::Tool {
        name: "evals".into(),
        reason: e.to_string(),
    })?;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(results_path)?;
    writeln!(file, "{line}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_case_serde_round_trip() {
        let toml_str = r#"
id = "test-001"
prompt = "Fix the bug in main.rs"
repo_fixture = "fixtures/test-repo"
check_cmd = "cargo test"
timeout_secs = 60
"#;
        let case: EvalCase = toml::from_str(toml_str).unwrap();
        assert_eq!(case.id, "test-001");
        assert_eq!(case.timeout_secs, 60);
    }

    fn write_case(dir: &Path, file_stem: &str, id: &str, extra: &str) {
        std::fs::write(
            dir.join(format!("{file_stem}.toml")),
            format!(
                "id = {id:?}\nprompt = \"p\"\nrepo_fixture = \".\"\ncheck_cmd = \"true\"\n{extra}"
            ),
        )
        .unwrap();
    }

    #[test]
    fn load_suite_filters_by_file_stem_before_scheduling() {
        // **Validates: Requirements 2.7**
        let dir = tempfile::tempdir().unwrap();
        write_case(dir.path(), "selected", "selected-case", "");
        write_case(dir.path(), "other", "other-suite-case", "");

        let selected = load_suite(dir.path(), "selected").unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "selected-case");

        // An unknown suite selects nothing rather than falling back to all.
        assert!(load_suite(dir.path(), "absent").unwrap().is_empty());

        // Every case remains reachable through the explicit "all" selector.
        let all = load_suite(dir.path(), ALL_SUITES).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(load_cases(dir.path()).unwrap().len(), 2);
    }

    #[test]
    fn explicit_suite_field_overrides_the_file_stem() {
        // **Validates: Requirements 2.7**
        let dir = tempfile::tempdir().unwrap();
        write_case(dir.path(), "stem", "moved-case", "suite = \"declared\"\n");
        assert!(load_suite(dir.path(), "stem").unwrap().is_empty());
        assert_eq!(load_suite(dir.path(), "declared").unwrap().len(), 1);
    }

    #[test]
    fn loaded_case_order_is_deterministic() {
        // **Validates: Requirements 2.7, 2.17**
        // Results are appended in case order, so load order must not depend on
        // platform directory iteration.
        let dir = tempfile::tempdir().unwrap();
        write_case(dir.path(), "zeta", "z-case", "");
        write_case(dir.path(), "alpha", "a-case", "");
        let ids: Vec<String> = load_suite(dir.path(), ALL_SUITES)
            .unwrap()
            .into_iter()
            .map(|case| case.id)
            .collect();
        assert_eq!(ids, vec!["a-case".to_string(), "z-case".to_string()]);
    }

    #[test]
    fn append_outcome_creates_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results").join("run.jsonl");
        let outcome = EvalOutcome {
            case_id: "t1".into(),
            passed: true,
            turns: 3,
            tool_calls: 5,
            tokens_in: 1000,
            tokens_out: 500,
            wall_time_ms: 2500,
            error: None,
        };
        append_outcome(&path, &outcome).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"case_id\":\"t1\""));
        assert!(content.contains("\"passed\":true"));
    }
}
