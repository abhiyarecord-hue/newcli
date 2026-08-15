//! Trajectory recording: subscribe to EventBus and record full trajectories
//! as append-only JSONL files.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use agent_types::{AgentError, AgentEvent, Result};
use serde_json;

/// Records agent events and tool call/result data to JSONL.
pub struct TrajectoryRecorder {
    output_path: PathBuf,
}

impl TrajectoryRecorder {
    pub fn new(output_dir: &Path, case_id: &str, run_id: &str) -> Self {
        let output_path = output_dir.join(case_id).join(format!("{run_id}.jsonl"));
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

/// Relative increase in turns that counts as a soft regression.
const TURN_REGRESSION_RATIO: f64 = 1.2;

/// Relative increase in input tokens that counts as a soft regression.
const TOKEN_REGRESSION_RATIO: f64 = 1.3;

/// Rendered when a metric grew from a zero baseline.
///
/// A percentage against zero is undefined; printing `+0%` for a real increase
/// contradicts the change it is describing, so the growth is stated instead.
pub const ZERO_BASELINE_LABEL: &str = "N/A (baseline zero)";

/// Index outcomes by case ID, separating IDs that appear more than once.
///
/// A duplicated ID makes "the" outcome for that case ambiguous, so it is
/// reported rather than silently resolved to whichever entry was seen last.
fn index_by_case(
    outcomes: &[crate::swebench::EvalOutcome],
) -> (
    BTreeMap<&str, &crate::swebench::EvalOutcome>,
    BTreeSet<&str>,
) {
    let mut indexed: BTreeMap<&str, &crate::swebench::EvalOutcome> = BTreeMap::new();
    let mut duplicates: BTreeSet<&str> = BTreeSet::new();
    for outcome in outcomes {
        if indexed.insert(outcome.case_id.as_str(), outcome).is_some() {
            duplicates.insert(outcome.case_id.as_str());
        }
    }
    (indexed, duplicates)
}

/// Soft-regression entry for one metric, or `None` when it did not regress.
///
/// Percent thresholds apply only to a positive baseline. Growth from zero is
/// always reported, but as an explicit zero-baseline note rather than a percent.
fn metric_regression(
    case_id: &str,
    metric: &str,
    baseline: f64,
    current: f64,
    ratio: f64,
) -> Option<String> {
    if baseline > 0.0 {
        if current > baseline * ratio {
            let percent = ((current / baseline) - 1.0) * 100.0;
            return Some(format!("{case_id}: {metric} +{percent:.0}%"));
        }
        return None;
    }
    if current > 0.0 {
        return Some(format!(
            "{case_id}: {metric} rose from 0 to {current:.0} — {ZERO_BASELINE_LABEL}"
        ));
    }
    None
}

/// Compare two runs over the union of their case IDs.
///
/// Every case is classified, so a case dropped from the current run cannot
/// disappear from the report. A previously passing case that is now absent is a
/// hard regression: the evidence that it passed is gone.
pub fn diff_runs(
    baseline: &[crate::swebench::EvalOutcome],
    current: &[crate::swebench::EvalOutcome],
) -> DiffResult {
    let (baseline_index, baseline_duplicates) = index_by_case(baseline);
    let (current_index, current_duplicates) = index_by_case(current);

    let mut result = DiffResult::default();

    // The union, in a stable order so two identical inputs always report
    // identically.
    let case_ids: BTreeSet<&str> = baseline_index
        .keys()
        .chain(current_index.keys())
        .copied()
        .collect();

    for case_id in case_ids {
        let duplicated =
            baseline_duplicates.contains(case_id) || current_duplicates.contains(case_id);
        if duplicated {
            result.duplicates.push(case_id.to_string());
            result.errors.push(format!(
                "{case_id}: listed more than once in a run; comparison skipped"
            ));
            continue;
        }

        match (baseline_index.get(case_id), current_index.get(case_id)) {
            (Some(base), Some(curr)) => {
                result.compared.push(case_id.to_string());
                if base.passed && !curr.passed {
                    result.hard_regressions.push(case_id.to_string());
                }
                if let Some(entry) = metric_regression(
                    case_id,
                    "turns",
                    base.turns as f64,
                    curr.turns as f64,
                    TURN_REGRESSION_RATIO,
                ) {
                    result.soft_regressions.push(entry);
                }
                if let Some(entry) = metric_regression(
                    case_id,
                    "tokens_in",
                    base.tokens_in as f64,
                    curr.tokens_in as f64,
                    TOKEN_REGRESSION_RATIO,
                ) {
                    result.soft_regressions.push(entry);
                }
            }
            (Some(base), None) => {
                result.removed.push(case_id.to_string());
                if base.passed {
                    // A passing case that vanished is not neutral: its evidence
                    // is gone, so it fails the gate like a pass to fail.
                    result.hard_regressions.push(case_id.to_string());
                } else {
                    result
                        .soft_regressions
                        .push(format!("{case_id}: removed from the current run"));
                }
            }
            (None, Some(_)) => result.added.push(case_id.to_string()),
            (None, None) => {}
        }
    }

    result
}

#[derive(Clone, Debug, Default)]
pub struct DiffResult {
    pub hard_regressions: Vec<String>,
    pub soft_regressions: Vec<String>,
    /// Cases present in both runs and actually compared.
    pub compared: Vec<String>,
    /// Cases only in the current run.
    pub added: Vec<String>,
    /// Cases only in the baseline run.
    pub removed: Vec<String>,
    /// Cases listed more than once, so not comparable.
    pub duplicates: Vec<String>,
    /// Why a case could not be compared.
    pub errors: Vec<String>,
}

impl DiffResult {
    pub fn has_hard_regression(&self) -> bool {
        !self.hard_regressions.is_empty()
    }

    /// Whether a gate should refuse this comparison.
    ///
    /// Broader than [`has_hard_regression`](Self::has_hard_regression): a case
    /// that could not be compared is not evidence of passing, so an ambiguous
    /// result must not exit green. Kept separate so existing consumers of the
    /// hard-regression signal are unchanged.
    pub fn blocks_gate(&self) -> bool {
        self.has_hard_regression() || !self.errors.is_empty()
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

    fn case(case_id: &str, passed: bool, turns: u32, tokens_in: u64) -> EvalOutcome {
        EvalOutcome {
            case_id: case_id.into(),
            passed,
            turns,
            tool_calls: 0,
            tokens_in,
            tokens_out: 0,
            wall_time_ms: 1,
            error: None,
        }
    }

    #[test]
    fn removed_passing_case_is_a_hard_regression() {
        // **Validates: Requirements 2.18**
        let diff = diff_runs(&[case("gone", true, 1, 1)], &[]);
        assert_eq!(diff.removed, vec!["gone".to_string()]);
        assert!(diff.hard_regressions.contains(&"gone".to_string()));
        assert!(diff.compared.is_empty());
        assert!(diff.has_hard_regression());
    }

    #[test]
    fn removed_failing_case_is_reported_without_failing_the_gate() {
        // **Validates: Requirements 2.18**
        // Nothing was proven about a case that was already failing, so its
        // removal is disclosed but does not gate.
        let diff = diff_runs(&[case("gone", false, 1, 1)], &[]);
        assert_eq!(diff.removed, vec!["gone".to_string()]);
        assert!(!diff.has_hard_regression());
        assert!(diff
            .soft_regressions
            .iter()
            .any(|entry| entry.contains("gone")));
    }

    #[test]
    fn added_case_is_classified_and_never_a_regression() {
        // **Validates: Requirements 2.18**
        let diff = diff_runs(&[], &[case("fresh", true, 9, 9)]);
        assert_eq!(diff.added, vec!["fresh".to_string()]);
        assert!(diff.hard_regressions.is_empty());
        assert!(diff.soft_regressions.is_empty());
    }

    #[test]
    fn growth_from_a_zero_baseline_is_never_rendered_as_a_percentage() {
        // **Validates: Requirements 2.18**
        let diff = diff_runs(&[case("zero", true, 0, 0)], &[case("zero", true, 4, 7)]);
        assert!(
            diff.soft_regressions
                .iter()
                .all(|entry| !entry.contains('%')),
            "a zero baseline has no meaningful percentage: {:?}",
            diff.soft_regressions
        );
        assert_eq!(diff.soft_regressions.len(), 2, "both metrics grew");
        assert!(diff
            .soft_regressions
            .iter()
            .all(|entry| entry.contains(ZERO_BASELINE_LABEL)));
        assert!(!diff.has_hard_regression());
    }

    #[test]
    fn zero_baseline_that_stays_zero_is_not_a_regression() {
        // **Validates: Requirements 2.18**
        let diff = diff_runs(&[case("zero", true, 0, 0)], &[case("zero", true, 0, 0)]);
        assert!(diff.soft_regressions.is_empty());
        assert_eq!(diff.compared, vec!["zero".to_string()]);
    }

    #[test]
    fn percent_thresholds_apply_only_above_the_ratio() {
        // **Validates: Requirements 2.18**
        // 10 -> 11 is under the 1.2 turn ratio; 10 -> 13 is over it.
        let under = diff_runs(&[case("c", true, 10, 100)], &[case("c", true, 11, 100)]);
        assert!(under.soft_regressions.is_empty());

        let over = diff_runs(&[case("c", true, 10, 100)], &[case("c", true, 13, 100)]);
        assert_eq!(over.soft_regressions.len(), 1);
        assert!(over.soft_regressions[0].contains("turns +30%"));
    }

    #[test]
    fn duplicate_case_ids_are_reported_instead_of_silently_resolved() {
        // **Validates: Requirements 2.18**
        let diff = diff_runs(
            &[case("dup", true, 1, 1), case("dup", false, 9, 9)],
            &[case("dup", false, 9, 9)],
        );
        assert_eq!(diff.duplicates, vec!["dup".to_string()]);
        assert!(diff.errors.iter().any(|entry| entry.contains("dup")));
        // Ambiguous input must not be silently scored either way.
        assert!(diff.compared.is_empty());
        assert!(diff.hard_regressions.is_empty());
        // But it must not pass the gate either: nothing was verified.
        assert!(diff.blocks_gate());
    }

    #[test]
    fn a_clean_comparison_does_not_block_the_gate() {
        // **Validates: Requirements 2.18**
        let diff = diff_runs(&[case("c", true, 1, 1)], &[case("c", true, 1, 1)]);
        assert!(!diff.blocks_gate());
        assert!(diff.errors.is_empty());
    }

    #[test]
    fn every_union_case_is_classified_exactly_once() {
        // **Validates: Requirements 2.18**
        let baseline = [case("both", true, 1, 1), case("only_base", true, 1, 1)];
        let current = [case("both", true, 1, 1), case("only_curr", true, 1, 1)];
        let diff = diff_runs(&baseline, &current);
        assert_eq!(diff.compared, vec!["both".to_string()]);
        assert_eq!(diff.removed, vec!["only_base".to_string()]);
        assert_eq!(diff.added, vec!["only_curr".to_string()]);
        assert_eq!(
            diff.compared.len() + diff.removed.len() + diff.added.len() + diff.duplicates.len(),
            3,
            "the union must be fully partitioned"
        );
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
