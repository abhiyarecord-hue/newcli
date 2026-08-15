//! `evals` (L5): evaluation harness — SWE-bench-lite runner, trajectory
//! recorder, regression scoring.

pub mod report;
pub mod run;
pub mod swebench;
pub mod trajectory;

pub use report::{EvalReport, PassRate};
pub use run::{create_exclusive_run, new_run_id, results_path, validate_run_id, RunHandle};
pub use swebench::{EvalCase, EvalOutcome};
pub use trajectory::TrajectoryRecorder;
