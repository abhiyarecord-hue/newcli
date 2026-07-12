//! `lsp-client` (L2): JSON-RPC 2.0 over stdio, Content-Length framing.
//!
//! - [`client`]: [`LspClient`] — spawn, initialize, request/notify, shutdown.
//! - [`diagnostics`]: buffered diagnostics, definitions, references.

pub mod client;
pub mod diagnostics;

pub use client::LspClient;
pub use diagnostics::{
    byte_offset_to_utf16_col, diagnostics_for, did_change, did_open, find_references,
    goto_definition, language_id_from_path, utf16_col_to_byte_offset, Diagnostic,
    DiagnosticsStore, Location,
};
