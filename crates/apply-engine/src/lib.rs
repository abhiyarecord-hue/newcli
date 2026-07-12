//! `apply-engine` (L2): Fast-Apply SLM bridge, semantic diff, CRDT doc.
//!
//! - [`lazy_edit`]: Validate snippets containing `... existing code ...` markers.
//! - [`fast_apply`]: `ApplyStrategy` trait + `FastApplyStrategy` / `FallbackStrategy`.
//! - [`apply_engine`]: Main `ApplyEngine` orchestrating fast-then-fallback.
//! - `semantic_diff`: Entity-level diff (TASK-5.2).
//! - `crdt_doc` / `ipc`: CRDT concurrent documents (TASK-5.3).

pub mod apply_engine;
pub mod crdt_doc;
pub mod fast_apply;
pub mod ipc;
pub mod lazy_edit;
pub mod semantic_diff;

pub use apply_engine::ApplyEngine;
pub use crdt_doc::{CrdtDoc, Patch, PatchMessage};
pub use fast_apply::{ApplyStrategy, FallbackStrategy, FastApplyStrategy};
pub use ipc::IpcServer;
pub use lazy_edit::LazyEdit;
pub use semantic_diff::{semantic_diff, ChangeKind, EntityChange};
