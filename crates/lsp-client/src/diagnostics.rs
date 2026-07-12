//! Diagnostics buffering + goto_definition / find_references / diagnostics_for.
//!
//! `publishDiagnostics` notifications are buffered into a `DashMap<String, Vec<Diagnostic>>`.
//! `diagnostics_for` waits until no new diagnostics arrive for a `settle` window.
//! UTF-16 column conversion from byte offsets included.

use std::path::Path;
use std::time::Duration;

use agent_types::{AgentError, Result};
use dashmap::DashMap;
use serde_json::{json, Value};

use crate::client::LspClient;

/// A simplified diagnostic (from the LSP spec).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    pub severity: u32, // 1=Error, 2=Warning, 3=Info, 4=Hint
    pub message: String,
    pub range_start_line: u32,
    pub range_start_col: u32,
    pub range_end_line: u32,
    pub range_end_col: u32,
}

/// A location (from goto_definition / find_references).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Location {
    pub uri: String,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

/// Diagnostics buffer.
pub struct DiagnosticsStore {
    pub store: DashMap<String, Vec<Diagnostic>>,
}

impl Default for DiagnosticsStore {
    fn default() -> Self {
        Self {
            store: DashMap::new(),
        }
    }
}

impl DiagnosticsStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a `textDocument/publishDiagnostics` notification payload.
    pub fn on_publish_diagnostics(&self, params: &Value) {
        let uri = params
            .get("uri")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let diags: Vec<Diagnostic> = params
            .get("diagnostics")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|d| parse_diagnostic(d))
                    .collect()
            })
            .unwrap_or_default();
        self.store.insert(uri, diags);
    }

    /// Get current diagnostics for a file.
    pub fn get(&self, uri: &str) -> Vec<Diagnostic> {
        self.store.get(uri).map(|v| v.clone()).unwrap_or_default()
    }

    /// Clear diagnostics for a file.
    pub fn clear(&self, uri: &str) {
        self.store.remove(uri);
    }
}

fn parse_diagnostic(v: &Value) -> Option<Diagnostic> {
    let range = v.get("range")?;
    let start = range.get("start")?;
    let end = range.get("end")?;
    Some(Diagnostic {
        severity: v
            .get("severity")
            .and_then(Value::as_u64)
            .unwrap_or(1) as u32,
        message: v
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        range_start_line: start.get("line").and_then(Value::as_u64).unwrap_or(0) as u32,
        range_start_col: start.get("character").and_then(Value::as_u64).unwrap_or(0) as u32,
        range_end_line: end.get("line").and_then(Value::as_u64).unwrap_or(0) as u32,
        range_end_col: end.get("character").and_then(Value::as_u64).unwrap_or(0) as u32,
    })
}

/// Send `textDocument/didOpen` notification.
pub async fn did_open(client: &LspClient, uri: &str, language_id: &str, text: &str) -> Result<()> {
    client
        .notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text
                }
            }),
        )
        .await
}

/// Send `textDocument/didChange` notification (full-sync mode).
pub async fn did_change(client: &LspClient, uri: &str, version: i32, text: &str) -> Result<()> {
    client
        .notify(
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": version },
                "contentChanges": [{ "text": text }]
            }),
        )
        .await
}

/// Request `textDocument/definition`.
pub async fn goto_definition(
    client: &mut LspClient,
    uri: &str,
    line: u32,
    col: u32,
) -> Result<Vec<Location>> {
    let result = client
        .request(
            "textDocument/definition",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": col }
            }),
        )
        .await?;
    Ok(parse_locations(&result))
}

/// Request `textDocument/references`.
pub async fn find_references(
    client: &mut LspClient,
    uri: &str,
    line: u32,
    col: u32,
) -> Result<Vec<Location>> {
    let result = client
        .request(
            "textDocument/references",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": col },
                "context": { "includeDeclaration": true }
            }),
        )
        .await?;
    Ok(parse_locations(&result))
}

/// Wait until no new diagnostics arrive for the `settle` window, then return them.
pub async fn diagnostics_for(
    diag_store: &DiagnosticsStore,
    uri: &str,
    settle: Duration,
) -> Vec<Diagnostic> {
    // Poll until stable (no change for `settle` duration).
    let mut prev_count = 0usize;
    let mut stable_since = tokio::time::Instant::now();

    loop {
        let current = diag_store.get(uri);
        let cur_count = current.len();

        if cur_count != prev_count {
            prev_count = cur_count;
            stable_since = tokio::time::Instant::now();
        } else if stable_since.elapsed() >= settle {
            return current;
        }

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Hard cap: don't wait forever.
        if stable_since.elapsed() > Duration::from_secs(10) {
            return diag_store.get(uri);
        }
    }
}

/// Convert byte offset in source to UTF-16 code units (LSP column).
pub fn byte_offset_to_utf16_col(source: &str, line: u32, byte_col: usize) -> u32 {
    let line_start = source
        .lines()
        .take(line as usize)
        .map(|l| l.len() + 1) // +1 for \n
        .sum::<usize>();
    let slice = source
        .get(line_start..line_start + byte_col)
        .unwrap_or("");
    slice.encode_utf16().count() as u32
}

/// Convert LSP position (UTF-16 col) to byte offset.
pub fn utf16_col_to_byte_offset(line_text: &str, utf16_col: u32) -> usize {
    let mut utf16_count = 0u32;
    let mut byte_offset = 0usize;
    for ch in line_text.chars() {
        if utf16_count >= utf16_col {
            break;
        }
        utf16_count += ch.len_utf16() as u32;
        byte_offset += ch.len_utf8();
    }
    byte_offset
}

fn parse_locations(value: &Value) -> Vec<Location> {
    match value {
        Value::Array(arr) => arr.iter().filter_map(parse_location).collect(),
        Value::Object(_) => parse_location(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

fn parse_location(v: &Value) -> Option<Location> {
    let uri = v.get("uri").and_then(Value::as_str)?.to_string();
    let range = v.get("range")?;
    let start = range.get("start")?;
    let end = range.get("end")?;
    Some(Location {
        uri,
        start_line: start.get("line").and_then(Value::as_u64)? as u32,
        start_col: start.get("character").and_then(Value::as_u64)? as u32,
        end_line: end.get("line").and_then(Value::as_u64)? as u32,
        end_col: end.get("character").and_then(Value::as_u64)? as u32,
    })
}

/// Detect language ID from file path (for didOpen).
pub fn language_id_from_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py" | "pyi") => "python",
        Some("ts" | "tsx") => "typescript",
        Some("js" | "jsx") => "javascript",
        Some("json") => "json",
        Some("toml") => "toml",
        Some("md") => "markdown",
        _ => "plaintext",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_store_buffers_and_retrieves() {
        let store = DiagnosticsStore::new();
        let params = json!({
            "uri": "file:///src/main.rs",
            "diagnostics": [
                {
                    "range": {
                        "start": {"line": 5, "character": 0},
                        "end": {"line": 5, "character": 10}
                    },
                    "severity": 1,
                    "message": "type error"
                }
            ]
        });
        store.on_publish_diagnostics(&params);
        let diags = store.get("file:///src/main.rs");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, 1);
        assert_eq!(diags[0].message, "type error");
        assert_eq!(diags[0].range_start_line, 5);
    }

    #[test]
    fn utf16_col_conversion_ascii() {
        // For pure ASCII, byte offset == UTF-16 offset.
        let line = "fn hello() {}";
        assert_eq!(utf16_col_to_byte_offset(line, 3), 3);
        assert_eq!(byte_offset_to_utf16_col("fn hello() {}\n", 0, 3), 3);
    }

    #[test]
    fn utf16_col_conversion_multibyte() {
        // "á" is 2 bytes UTF-8 but 1 UTF-16 code unit.
        let line = "ábc";
        assert_eq!(utf16_col_to_byte_offset(line, 1), 2); // after 'á'
        assert_eq!(utf16_col_to_byte_offset(line, 2), 3); // after 'b'
    }

    #[test]
    fn language_id_detection() {
        use std::path::PathBuf;
        assert_eq!(language_id_from_path(&PathBuf::from("x.rs")), "rust");
        assert_eq!(language_id_from_path(&PathBuf::from("y.py")), "python");
        assert_eq!(language_id_from_path(&PathBuf::from("z.ts")), "typescript");
        assert_eq!(language_id_from_path(&PathBuf::from("a.txt")), "plaintext");
    }

    #[test]
    fn parse_diagnostic_from_json() {
        let v = json!({
            "range": {
                "start": {"line": 3, "character": 2},
                "end": {"line": 3, "character": 8}
            },
            "severity": 2,
            "message": "unused import"
        });
        let d = parse_diagnostic(&v).unwrap();
        assert_eq!(d.severity, 2);
        assert_eq!(d.message, "unused import");
        assert_eq!(d.range_start_line, 3);
        assert_eq!(d.range_start_col, 2);
    }
}
