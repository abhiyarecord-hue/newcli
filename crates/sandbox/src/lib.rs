//! `sandbox` (L2): Path jail, process executor, SSRF-filtered network.
//!
//! Despite the crate name there is **no MicroVM and no hard network isolation**.
//! Command execution is a best-effort process policy, not a security boundary; a
//! determined command can still reach the host. Path confinement and the SSRF
//! filter are enforced boundaries for the paths and URLs that flow through them.
//!
//! - [`path_jail`]: [`PathJail`] — canonicalized root boundary for all file access.
//! - [`executor`]: [`SandboxExecutor`] trait + [`ProcessFallback`] (OS processes).
//! - [`net_guard`]: [`NetGuard`] — SSRF-filtered HTTPS-only client.

pub mod bounded_output;
pub mod executor;
pub mod net_guard;
pub mod path_jail;
pub mod process_tree;

pub use executor::{ProcessFallback, SandboxExecutor};
pub use net_guard::NetGuard;
pub use path_jail::PathJail;
pub use process_tree::{
    ExecResult, ProcessConfig, ProcessSupervisor, SupervisedChild, SupervisedStdioChild,
    Termination, DEFAULT_OUTPUT_LIMIT, DEFAULT_TERMINATION_GRACE,
};
