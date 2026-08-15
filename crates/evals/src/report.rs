//! Evaluation reporting: pass-rate tables, per-case breakdowns.

use crate::swebench::EvalOutcome;

pub struct PassRate {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
}

impl PassRate {
    pub fn from_outcomes(outcomes: &[EvalOutcome]) -> Self {
        let passed = outcomes.iter().filter(|o| o.passed).count();
        Self {
            total: outcomes.len(),
            passed,
            failed: outcomes.len() - passed,
        }
    }

    pub fn percentage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.passed as f64 / self.total as f64) * 100.0
    }
}

/// How a run produced its outcomes.
///
/// The runner executes each case's `check_cmd` and nothing else, so agent-style
/// measurements do not exist. Recording the mode keeps the report from implying
/// a capability the runner does not have.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Only the configured check command ran. No agent turns, tool calls, or
    /// token counts were measured.
    #[default]
    CheckOnly,
}

impl ExecutionMode {
    /// Stable machine-readable label used in reports.
    pub fn label(&self) -> &'static str {
        match self {
            Self::CheckOnly => "check_only",
        }
    }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

pub struct EvalReport {
    pub outcomes: Vec<EvalOutcome>,
    pub execution_mode: ExecutionMode,
}

impl EvalReport {
    pub fn new(outcomes: Vec<EvalOutcome>) -> Self {
        Self {
            outcomes,
            execution_mode: ExecutionMode::CheckOnly,
        }
    }

    pub fn pass_rate(&self) -> PassRate {
        PassRate::from_outcomes(&self.outcomes)
    }

    /// Print a summary table.
    ///
    /// Agent columns are deliberately omitted: the runner never measures turns,
    /// tool calls, or tokens, so printing them would fabricate measurements.
    /// Pass/fail and wall time are the only real observations.
    pub fn print_summary(&self) {
        let rate = self.pass_rate();
        println!("=== Evaluation Results ===");
        println!("execution_mode: {}", self.execution_mode);
        println!(
            "full-agent execution: unsupported (turns, tool calls, and token counts \
             are not measured in this mode and are omitted rather than reported as zero)"
        );
        println!(
            "Pass rate: {}/{} ({:.1}%)",
            rate.passed,
            rate.total,
            rate.percentage()
        );
        println!("{:-<50}", "");
        println!("{:<20} {:>6} {:>8}", "Case", "Pass", "Time(ms)");
        println!("{:-<50}", "");
        for o in &self.outcomes {
            let status = if o.passed { "✓" } else { "✗" };
            println!("{:<20} {:>6} {:>8}", o.case_id, status, o.wall_time_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swebench::EvalOutcome;

    #[test]
    fn pass_rate_calculation() {
        let outcomes = vec![
            EvalOutcome {
                case_id: "a".into(),
                passed: true,
                turns: 1,
                tool_calls: 1,
                tokens_in: 0,
                tokens_out: 0,
                wall_time_ms: 100,
                error: None,
            },
            EvalOutcome {
                case_id: "b".into(),
                passed: false,
                turns: 2,
                tool_calls: 3,
                tokens_in: 0,
                tokens_out: 0,
                wall_time_ms: 200,
                error: Some("timeout".into()),
            },
        ];
        let rate = PassRate::from_outcomes(&outcomes);
        assert_eq!(rate.passed, 1);
        assert_eq!(rate.failed, 1);
        assert_eq!(rate.percentage(), 50.0);
    }
}
