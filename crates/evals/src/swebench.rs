//! SWE-bench-lite scripted runner.
//!
//! Loads eval cases from `.agent/evals/cases/*.toml`, runs each in isolation,
//! appends outcomes as JSONL.

use std::path::{Path, PathBuf};

use agent_types::{AgentError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCase {
    pub id: String,
    pub prompt: String,
    pub repo_fixture: PathBuf,
    pub check_cmd: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_timeout() -> u64 {
    120
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

/// Load eval cases from a directory of TOML files.
pub fn load_cases(dir: &Path) -> Result<Vec<EvalCase>> {
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
            cases.push(case);
        }
    }
    Ok(cases)
}

/// Append one outcome as a JSONL line to a results file.
pub fn append_outcome(results_path: &Path, outcome: &EvalOutcome) -> Result<()> {
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
