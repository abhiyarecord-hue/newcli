//! `sandbox` (L2): Path jail, MicroVM executor, SSRF-filtered network.
//!
//! - [`path_jail`]: [`PathJail`] — canonicalized root boundary for all file access.
//! - [`executor`]: [`SandboxExecutor`] trait + [`ProcessFallback`].
//! - [`net_guard`]: [`NetGuard`] — SSRF-filtered HTTPS-only client.

pub mod executor;
pub mod net_guard;
pub mod path_jail;

pub use executor::{ExecResult, ProcessFallback, SandboxExecutor};
pub use net_guard::NetGuard;
pub use path_jail::PathJail;
