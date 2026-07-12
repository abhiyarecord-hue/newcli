//! Trajectory recording: subscribe to EventBus and record full trajectories
//! as append-only JSONL files.

use std::path::{Path, PathBuf};

use agent_types::{AgentEvent, Result, AgentError};
use serde_json;

/// Records agent events and tool call/result data to JSONL.
pub struct TrajectoryRecorder {
    output_path: PathBuf,
}

impl TrajectoryRecorder {
    pub fn new(output_dir: &Path, case_id: &str, run_id: &str) -> Self {
        let output_path = output_dir
            .join(case_id)
            .join(format!("{run_id}.jsonl"));
        Self { output_path }
    }

    /// Record a single event.
    pub fn record_event(&self, event: &AgentEvent) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        if let Some(parent) = self.output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let line = serde_json::to_string(event).map_err(|e| AgentError::Tool {
            name: "trajectory".into(),
            reason: e.to_string(),
        })?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    pub fn output_path(&self) -> &Path {
        &self.output_path
    }
}

/// Compare two runs: detect hard regressions (pass→fail) and soft regressions.
pub fn diff_runs(
    baseline: &[crate::swebench::EvalOutcome],
    current: &[crate::swebench::EvalOutcome],
) -> DiffResult {
    let mut hard_regressions = Vec::new();
    let mut soft_regressions = Vec::new();

    for curr in current {
        if let Some(base) = baseline.iter().find(|b| b.case_id == curr.case_id) {
            // Hard regression: pass → fail.
            if base.passed && !curr.passed {
                hard_regressions.push(curr.case_id.clone());
            }
            // Soft regression: turns +20% or tokens +30%.
            if curr.turns as f64 > base.turns as f64 * 1.2 {
                soft_regressions.push(format!("{}: turns +{:.0}%", curr.case_id,
                    ((curr.turns as f64 / base.turns.max(1) as f64) - 1.0) * 100.0));
            }
            if curr.tokens_in > (base.tokens_in as f64 * 1.3) as u64 {
                soft_regressions.push(format!("{}: tokens_in +{:.0}%", curr.case_id,
                    ((curr.tokens_in as f64 / base.tokens_in.max(1) as f64) - 1.0) * 100.0));
            }
        }
    }

    DiffResult {
        hard_regressions,
        soft_regressions,
    }
}

pub struct DiffResult {
    pub hard_regressions: Vec<String>,
    pub soft_regressions: Vec<String>,
}

impl DiffResult {
    pub fn has_hard_regression(&self) -> bool {
        !self.hard_regressions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swebench::EvalOutcome;

    #[test]
    fn hard_regression_detected() {
        let baseline = vec![EvalOutcome {
            case_id: "t1".into(),
            passed: true,
            turns: 3,
            tool_calls: 2,
            tokens_in: 1000,
            tokens_out: 500,
            wall_time_ms: 2000,
            error: None,
        }];
        let current = vec![EvalOutcome {
            case_id: "t1".into(),
            passed: false,
            turns: 4,
            tool_calls: 3,
            tokens_in: 1200,
            tokens_out: 600,
            wall_time_ms: 3000,
            error: Some("failed".into()),
        }];
        let diff = diff_runs(&baseline, &current);
        assert!(diff.has_hard_regression());
        assert_eq!(diff.hard_regressions[0], "t1");
    }

    #[test]
    fn identical_runs_no_regression() {
        let outcomes = vec![EvalOutcome {
            case_id: "t1".into(),
            passed: true,
            turns: 3,
            tool_calls: 2,
            tokens_in: 1000,
            tokens_out: 500,
            wall_time_ms: 2000,
            error: None,
        }];
        let diff = diff_runs(&outcomes, &outcomes);
        assert!(!diff.has_hard_regression());
        assert!(diff.soft_regressions.is_empty());
    }
}
