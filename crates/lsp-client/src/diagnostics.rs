//! Diagnostics buffering + goto_definition / find_references / diagnostics_for.
//!
//! `publishDiagnostics` notifications are buffered with a normalized content/version
//! fingerprint. `diagnostics_for` waits for a quiet window but never extends its
//! independent absolute deadline. UTF-16 conversion scans the source's real line
//! terminators instead of assuming every line ends with one `\n` byte.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use agent_types::{AgentError, Result};
use dashmap::DashMap;
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::client::LspClient;

const MAX_DIAGNOSTIC_WAIT: Duration = Duration::from_secs(10);
const DIAGNOSTIC_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// A simplified diagnostic (from the LSP spec).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DiagnosticFingerprint {
    version: Option<i64>,
    normalized_content: Vec<String>,
}

#[derive(Clone, Debug)]
struct DiagnosticSnapshot {
    diagnostics: Vec<Diagnostic>,
    fingerprint: DiagnosticFingerprint,
}

/// Diagnostics buffer.
pub struct DiagnosticsStore {
    /// Compatibility view of the latest parsed diagnostics.
    pub store: DashMap<String, Vec<Diagnostic>>,
    snapshots: DashMap<String, DiagnosticSnapshot>,
    update_revision: AtomicU64,
    updates: watch::Sender<u64>,
}

impl Default for DiagnosticsStore {
    fn default() -> Self {
        let (updates, _receiver) = watch::channel(0);
        Self {
            store: DashMap::new(),
            snapshots: DashMap::new(),
            update_revision: AtomicU64::new(0),
            updates,
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
        let raw_diagnostics = params
            .get("diagnostics")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let diagnostics: Vec<Diagnostic> = raw_diagnostics
            .iter()
            .filter_map(parse_diagnostic)
            .collect();
        let fingerprint = DiagnosticFingerprint {
            version: params.get("version").and_then(Value::as_i64),
            normalized_content: normalize_diagnostic_content(raw_diagnostics),
        };

        // The snapshot is authoritative for settling and is inserted first so
        // fingerprint and returned diagnostics always come from one publication.
        self.snapshots.insert(
            uri.clone(),
            DiagnosticSnapshot {
                diagnostics: diagnostics.clone(),
                fingerprint,
            },
        );
        self.store.insert(uri, diagnostics);
        self.notify_update();
    }

    /// Get current diagnostics for a file.
    pub fn get(&self, uri: &str) -> Vec<Diagnostic> {
        self.store
            .get(uri)
            .map(|diagnostics| diagnostics.clone())
            .unwrap_or_default()
    }

    /// Clear diagnostics for a file.
    pub fn clear(&self, uri: &str) {
        self.snapshots.remove(uri);
        self.store.remove(uri);
        self.notify_update();
    }

    fn snapshot(&self, uri: &str) -> DiagnosticSnapshot {
        let diagnostics = self.get(uri);
        if let Some(snapshot) = self.snapshots.get(uri) {
            if snapshot.diagnostics == diagnostics {
                return snapshot.clone();
            }
        }

        // Preserve the historical public `store` field: if an embedder writes
        // through it directly, observe those values instead of shadowing them
        // with stale private metadata. Such writes have no document version.
        snapshot_from_parsed(diagnostics)
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        self.updates.subscribe()
    }

    fn notify_update(&self) {
        let revision = self.update_revision.fetch_add(1, Ordering::AcqRel) + 1;
        self.updates.send_replace(revision);
    }
}

fn snapshot_from_parsed(diagnostics: Vec<Diagnostic>) -> DiagnosticSnapshot {
    let mut normalized_content: Vec<String> = diagnostics
        .iter()
        .map(|diagnostic| serde_json::to_string(diagnostic).unwrap_or_default())
        .collect();
    normalized_content.sort();
    DiagnosticSnapshot {
        diagnostics,
        fingerprint: DiagnosticFingerprint {
            version: None,
            normalized_content,
        },
    }
}

fn normalize_diagnostic_content(diagnostics: &[Value]) -> Vec<String> {
    let mut normalized: Vec<String> = diagnostics
        .iter()
        .map(|diagnostic| serde_json::to_string(diagnostic).unwrap_or_default())
        .collect();
    // Diagnostic order is not semantically meaningful. Sorting prevents a
    // server reorder from extending the quiet window while retaining every
    // diagnostic field (including fields outside the simplified public type).
    normalized.sort();
    normalized
}

fn parse_diagnostic(v: &Value) -> Option<Diagnostic> {
    let range = v.get("range")?;
    let start = range.get("start")?;
    let end = range.get("end")?;
    Some(Diagnostic {
        severity: v.get("severity").and_then(Value::as_u64).unwrap_or(1) as u32,
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

/// Build a standards-compliant `file:` URI for a platform-native path.
pub(crate) fn file_uri_from_path(path: &Path) -> Result<String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| AgentError::Lsp(format!("resolve LSP root directory: {error}")))?
            .join(path)
    };

    url::Url::from_file_path(&absolute)
        .map(String::from)
        .map_err(|()| AgentError::Lsp(format!("cannot convert path to file URI: {absolute:?}")))
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

/// Wait until no diagnostic content/version change arrives for `settle`.
///
/// The absolute deadline is fixed at ten seconds and is never reset by a new
/// publication. Use [`diagnostics_for_with_deadline`] to inject a smaller cap.
pub async fn diagnostics_for(
    diag_store: &DiagnosticsStore,
    uri: &str,
    settle: Duration,
) -> Vec<Diagnostic> {
    diagnostics_for_with_deadline(diag_store, uri, settle, MAX_DIAGNOSTIC_WAIT).await
}

/// Wait for diagnostic stability with an absolute deadline configurable only
/// downward from the ten-second production ceiling.
pub async fn diagnostics_for_with_deadline(
    diag_store: &DiagnosticsStore,
    uri: &str,
    settle: Duration,
    absolute_deadline: Duration,
) -> Vec<Diagnostic> {
    let absolute_deadline = absolute_deadline.min(MAX_DIAGNOSTIC_WAIT);
    let started_at = tokio::time::Instant::now();
    let absolute_end = started_at + absolute_deadline;
    let mut quiet_end = started_at + settle;
    let mut updates = diag_store.subscribe();
    let initial = diag_store.snapshot(uri);
    let mut previous_fingerprint = initial.fingerprint;

    loop {
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(absolute_end) => {
                // The cap is absolute and cannot be starved by a busy server:
                // return the newest available publication without waiting for
                // another quiet window.
                return diag_store.snapshot(uri).diagnostics;
            }
            changed = updates.changed() => {
                // `DiagnosticsStore` owns the sender, so closure is not expected.
                // A closed channel still falls through to the final snapshot.
                let _ = changed;
            }
            _ = tokio::time::sleep_until(quiet_end) => {
                // A compatibility write through the public DashMap does not
                // emit a watch revision. Re-snapshot at the decision boundary
                // before declaring the quiet period complete.
                let latest = diag_store.snapshot(uri);
                if latest.fingerprint != previous_fingerprint {
                    previous_fingerprint = latest.fingerprint.clone();
                    quiet_end = tokio::time::Instant::now() + settle;
                    continue;
                }
                return latest.diagnostics;
            }
            // Retain compatibility with embedders that mutate the historical
            // public DashMap directly instead of calling on_publish_diagnostics.
            _ = tokio::time::sleep(DIAGNOSTIC_POLL_INTERVAL) => {}
        }

        let current = diag_store.snapshot(uri);
        if current.fingerprint != previous_fingerprint {
            previous_fingerprint = current.fingerprint.clone();
            quiet_end = tokio::time::Instant::now() + settle;
        }
    }
}

/// Convert a byte offset within a zero-based source line to UTF-16 code units.
pub fn byte_offset_to_utf16_col(source: &str, line: u32, byte_col: usize) -> u32 {
    let Some((line_start, line_end)) = source_line_range(source, line) else {
        return 0;
    };
    let Some(slice_end) = line_start.checked_add(byte_col) else {
        return 0;
    };
    if slice_end > line_end {
        return 0;
    }
    source
        .get(line_start..slice_end)
        .map(|slice| slice.encode_utf16().count() as u32)
        .unwrap_or(0)
}

fn source_line_range(source: &str, target_line: u32) -> Option<(usize, usize)> {
    let bytes = source.as_bytes();
    let mut line = 0u32;
    let mut line_start = 0usize;
    let mut index = 0usize;

    while index < bytes.len() {
        let terminator_len = match bytes[index] {
            b'\r' if bytes.get(index + 1) == Some(&b'\n') => 2,
            b'\r' | b'\n' => 1,
            _ => {
                index += 1;
                continue;
            }
        };

        if line == target_line {
            return Some((line_start, index));
        }
        index += terminator_len;
        line += 1;
        line_start = index;
    }

    (line == target_line).then_some((line_start, bytes.len()))
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
    use std::sync::Arc;

    use super::*;

    fn diagnostic(uri: &str, version: i64, message: &str) -> Value {
        json!({
            "uri": uri,
            "version": version,
            "diagnostics": [{
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 1}
                },
                "severity": 1,
                "message": message,
                "source": "test"
            }]
        })
    }

    #[test]
    fn diagnostics_store_buffers_and_retrieves() {
        let store = DiagnosticsStore::new();
        let params = json!({
            "uri": "file:///src/main.rs",
            "version": 7,
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
        let diagnostics = store.get("file:///src/main.rs");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, 1);
        assert_eq!(diagnostics[0].message, "type error");
        assert_eq!(diagnostics[0].range_start_line, 5);
        assert_eq!(
            store.snapshot("file:///src/main.rs").fingerprint.version,
            Some(7)
        );

        let direct = Diagnostic {
            severity: 2,
            message: "direct compatibility write".into(),
            range_start_line: 0,
            range_start_col: 0,
            range_end_line: 0,
            range_end_col: 1,
        };
        store
            .store
            .insert("file:///src/main.rs".into(), vec![direct.clone()]);
        assert_eq!(store.get("file:///src/main.rs"), vec![direct.clone()]);
        assert_eq!(
            store.snapshot("file:///src/main.rs").diagnostics,
            vec![direct]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn equal_count_content_and_version_changes_reset_only_quiet_time() {
        let store = Arc::new(DiagnosticsStore::new());
        let uri = "file:///changed.rs";
        store.on_publish_diagnostics(&diagnostic(uri, 1, "same"));
        let updater = store.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            updater.on_publish_diagnostics(&diagnostic(uri, 2, "same"));
            tokio::time::sleep(Duration::from_millis(25)).await;
            updater.on_publish_diagnostics(&diagnostic(uri, 2, "changed"));
        });

        let started = tokio::time::Instant::now();
        let result = diagnostics_for_with_deadline(
            &store,
            uri,
            Duration::from_millis(40),
            Duration::from_millis(200),
        )
        .await;
        assert_eq!(result[0].message, "changed");
        assert_eq!(started.elapsed(), Duration::from_millis(90));
    }

    #[tokio::test(start_paused = true)]
    async fn continuous_changes_cannot_extend_absolute_deadline() {
        let store = Arc::new(DiagnosticsStore::new());
        let uri = "file:///busy.rs";
        let updater = store.clone();
        tokio::spawn(async move {
            for version in 1..=20 {
                tokio::time::sleep(Duration::from_millis(15)).await;
                updater.on_publish_diagnostics(&diagnostic(uri, version, "changing"));
            }
        });

        let started = tokio::time::Instant::now();
        let _ = diagnostics_for_with_deadline(
            &store,
            uri,
            Duration::from_millis(60),
            Duration::from_millis(100),
        )
        .await;
        assert_eq!(started.elapsed(), Duration::from_millis(100));
    }

    #[test]
    fn diagnostic_reordering_has_the_same_normalized_fingerprint() {
        let left = json!([
            {"message":"b", "range":{"start":{"line":1}, "end":{"line":1}}},
            {"message":"a", "range":{"start":{"line":0}, "end":{"line":0}}}
        ]);
        let right = json!([
            {"range":{"end":{"line":0}, "start":{"line":0}}, "message":"a"},
            {"range":{"end":{"line":1}, "start":{"line":1}}, "message":"b"}
        ]);
        assert_eq!(
            normalize_diagnostic_content(left.as_array().unwrap()),
            normalize_diagnostic_content(right.as_array().unwrap())
        );
    }

    #[test]
    fn file_uri_escapes_reserved_and_unicode_path_bytes() {
        let uri = file_uri_from_path(Path::new("space # percent% é")).unwrap();
        assert!(uri.contains("%20"));
        assert!(uri.contains("%23"));
        assert!(uri.contains("%25"));
        assert!(uri.contains("%C3%A9"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_drive_and_unc_file_uris_are_canonical() {
        assert_eq!(
            file_uri_from_path(Path::new(r"C:\workspace #\é.rs")).unwrap(),
            "file:///C:/workspace%20%23/%C3%A9.rs"
        );
        assert_eq!(
            file_uri_from_path(Path::new(r"\\server\share name\a#b.rs")).unwrap(),
            "file://server/share%20name/a%23b.rs"
        );
    }

    #[test]
    fn utf16_columns_use_real_line_terminators_and_astral_units() {
        for terminator in ["\n", "\r\n", "\r"] {
            let source = format!("a{terminator}ह😀x");
            assert_eq!(
                byte_offset_to_utf16_col(&source, 1, "ह😀".len()),
                "ह😀".encode_utf16().count() as u32
            );
        }
        assert_eq!(byte_offset_to_utf16_col("a\r\n😀x", 1, "😀".len()), 2);
        assert_eq!(byte_offset_to_utf16_col("a\r\n", 1, 0), 0);
        assert_eq!(byte_offset_to_utf16_col("a", 2, 0), 0);
    }

    #[test]
    fn utf16_col_conversion_ascii_and_multibyte() {
        assert_eq!(utf16_col_to_byte_offset("fn hello() {}", 3), 3);
        assert_eq!(byte_offset_to_utf16_col("fn hello() {}\n", 0, 3), 3);
        assert_eq!(utf16_col_to_byte_offset("ábc", 1), 2);
        assert_eq!(utf16_col_to_byte_offset("ábc", 2), 3);
    }

    #[test]
    fn language_id_detection() {
        assert_eq!(language_id_from_path(Path::new("x.rs")), "rust");
        assert_eq!(language_id_from_path(Path::new("y.py")), "python");
        assert_eq!(language_id_from_path(Path::new("z.ts")), "typescript");
        assert_eq!(language_id_from_path(Path::new("a.txt")), "plaintext");
    }

    #[test]
    fn parse_diagnostic_from_json() {
        let value = json!({
            "range": {
                "start": {"line": 3, "character": 2},
                "end": {"line": 3, "character": 8}
            },
            "severity": 2,
            "message": "unused import"
        });
        let diagnostic = parse_diagnostic(&value).unwrap();
        assert_eq!(diagnostic.severity, 2);
        assert_eq!(diagnostic.message, "unused import");
        assert_eq!(diagnostic.range_start_line, 3);
        assert_eq!(diagnostic.range_start_col, 2);
    }
}
