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

pub struct EvalReport {
    pub outcomes: Vec<EvalOutcome>,
}

impl EvalReport {
    pub fn new(outcomes: Vec<EvalOutcome>) -> Self {
        Self { outcomes }
    }

    pub fn pass_rate(&self) -> PassRate {
        PassRate::from_outcomes(&self.outcomes)
    }

    /// Print a summary table.
    pub fn print_summary(&self) {
        let rate = self.pass_rate();
        println!("=== Evaluation Results ===");
        println!(
            "Pass rate: {}/{} ({:.1}%)",
            rate.passed,
            rate.total,
            rate.percentage()
        );
        println!("{:-<50}", "");
        println!("{:<20} {:>6} {:>6} {:>8}", "Case", "Pass", "Turns", "Time(ms)");
        println!("{:-<50}", "");
        for o in &self.outcomes {
            let status = if o.passed { "✓" } else { "✗" };
            println!(
                "{:<20} {:>6} {:>6} {:>8}",
                o.case_id, status, o.turns, o.wall_time_ms
            );
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
