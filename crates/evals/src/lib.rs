//! `evals` (L5): evaluation harness — SWE-bench-lite runner, trajectory
//! recorder, regression scoring.

pub mod report;
pub mod swebench;
pub mod trajectory;

pub use report::{EvalReport, PassRate};
pub use swebench::{EvalCase, EvalOutcome};
pub use trajectory::TrajectoryRecorder;
