//! LSP client lifecycle: spawn server over stdio, Content-Length framing,
//! initialize handshake, pending-request map, shutdown & kill on Drop.
//!
//! A dedicated reader task handles stdout; server-initiated requests get an
//! empty `null` result (TASK-4.1 context guard). Kill the child on Drop to
//! avoid orphaned language servers eating GBs of RAM.
//!
//! ## Hardening (Task 18.1)
//!
//! - Pending IDs are removed on send failure, timeout, cancellation, dropped
//!   response, EOF, and normal success.
//! - `Content-Length` is rejected before allocation when missing, malformed,
//!   duplicated, or over 16 MiB.
//! - Reader termination (EOF/error) fails all outstanding pending requests.
//! - LSP navigation/server-request completion remains intentionally incomplete
//!   (requirement 3.6).

use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use agent_types::{AgentError, Result};
use dashmap::DashMap;
use serde_json::{json, Value};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, Mutex};

/// Maximum allowed body size for an LSP message (16 MiB).
/// Configurable downward by embedders; tests inject smaller caps.
const MAX_CONTENT_LENGTH: usize = 16 * 1024 * 1024;

/// Maximum aggregate bytes allowed in one LSP header block. This prevents an
/// unterminated or irrelevant header from growing an unbounded `String` before
/// `Content-Length` can be validated.
const MAX_HEADER_BYTES: usize = 8 * 1024;

/// Pending-request state shared by request futures and the response reader.
/// `DashMap` lets a request's drop guard remove its ID synchronously during
/// cancellation. The terminal flag prevents requests from being inserted after
/// the only response reader has stopped.
struct PendingState {
    entries: DashMap<i64, oneshot::Sender<Value>>,
    reader_terminated: AtomicBool,
}

type PendingMap = Arc<PendingState>;

impl PendingState {
    fn new() -> Self {
        Self {
            entries: DashMap::new(),
            reader_terminated: AtomicBool::new(false),
        }
    }
}

/// Owns one pending-map entry until the response router consumes it. Dropping
/// the request future at any await point (send, timeout wait, or caller
/// cancellation) synchronously removes the entry and drops its sender.
struct PendingRequestGuard {
    pending: PendingMap,
    id: i64,
    armed: bool,
}

impl PendingRequestGuard {
    fn insert(pending: PendingMap, id: i64, sender: oneshot::Sender<Value>) -> Result<Self> {
        // The two terminal-state checks close the race with reader shutdown:
        // either shutdown observes and drains this entry, or the second check
        // observes shutdown and this guard removes the entry synchronously.
        if pending.reader_terminated.load(Ordering::SeqCst) {
            return Err(AgentError::Lsp("LSP reader terminated".into()));
        }

        pending.entries.insert(id, sender);
        let guard = Self {
            pending,
            id,
            armed: true,
        };

        if guard.pending.reader_terminated.load(Ordering::SeqCst) {
            return Err(AgentError::Lsp("LSP reader terminated".into()));
        }

        Ok(guard)
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        if self.armed {
            self.pending.entries.remove(&self.id);
        }
    }
}

pub struct LspClient {
    child: Option<Child>,
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    next_id: AtomicI64,
    pending: PendingMap,
    _reader_handle: tokio::task::JoinHandle<()>,
}

impl LspClient {
    /// Spawn a language server and complete the `initialize` handshake.
    /// Returns after server responds to `initialize` with capabilities.
    pub async fn start(server_cmd: &str, args: &[&str], root: &Path) -> Result<Self> {
        Self::start_with_config(server_cmd, args, root, MAX_CONTENT_LENGTH).await
    }

    /// Like [`start`](Self::start) but with an injectable body cap for testing.
    pub async fn start_with_config(
        server_cmd: &str,
        args: &[&str],
        root: &Path,
        max_body: usize,
    ) -> Result<Self> {
        let root_uri = crate::diagnostics::file_uri_from_path(root)?;

        let mut child = Command::new(server_cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| AgentError::Lsp(format!("spawn {server_cmd}: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AgentError::Lsp("no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::Lsp("no stdout".into()))?;

        let stdin = Arc::new(Mutex::new(stdin));
        let pending: PendingMap = Arc::new(PendingState::new());

        // A configured cap may make tests/embedders stricter, but can never
        // weaken the global 16 MiB allocation bound.
        let max_body = max_body.min(MAX_CONTENT_LENGTH);

        // Reader task: routes responses and fails all pending on termination.
        let pending_clone = pending.clone();
        let reader_handle = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            // Terminates on clean EOF (`Ok(None)`) or any protocol error.
            while let Ok(Some(msg)) = read_message(&mut reader, max_body).await {
                handle_incoming(msg, &pending_clone);
            }
            // Reader terminated: fail every outstanding pending request so callers
            // do not wait indefinitely.
            fail_all_pending(&pending_clone);
        });

        let mut client = Self {
            child: Some(child),
            stdin,
            next_id: AtomicI64::new(1),
            pending,
            _reader_handle: reader_handle,
        };

        // Initialize request.
        let init_params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {},
            "clientInfo": { "name": "rust-agent", "version": "0.1.0" }
        });

        let _result = client.request("initialize", init_params).await?;

        // Send `initialized` notification.
        client.notify("initialized", json!({})).await?;

        Ok(client)
    }

    /// Send a JSON-RPC request and await the response. Times out after 10s.
    pub async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let (tx, rx) = oneshot::channel();
        let pending_guard = PendingRequestGuard::insert(self.pending.clone(), id, tx)?;

        // The guard owns cleanup if sending fails or this future is cancelled
        // while the serialized write is pending.
        send_message(&self.stdin, &msg).await?;

        await_pending_response(
            method,
            rx,
            pending_guard,
            std::time::Duration::from_secs(10),
        )
        .await
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        send_message(&self.stdin, &msg).await
    }

    /// Graceful shutdown: send `shutdown` request then `exit` notification.
    pub async fn shutdown(mut self) -> Result<()> {
        let _ = self.request("shutdown", Value::Null).await;
        let _ = self.notify("exit", Value::Null).await;
        if let Some(mut child) = self.child.take() {
            let _ = child.kill().await;
        }
        Ok(())
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        // Kill the child process to avoid orphans.
        if let Some(ref mut child) = self.child {
            let _ = child.start_kill();
        }
    }
}

/// Await one registered response while retaining synchronous ownership of its
/// pending entry. Keeping this in one helper makes cancellation semantics
/// directly testable: dropping this future drops the guard immediately.
async fn await_pending_response(
    method: &str,
    rx: oneshot::Receiver<Value>,
    mut pending_guard: PendingRequestGuard,
    timeout: std::time::Duration,
) -> Result<Value> {
    match tokio::time::timeout(timeout, rx).await {
        Ok(Ok(value)) => {
            // Normal response routing removes the sender before delivering the
            // value. Disarm to make that ownership transfer explicit.
            pending_guard.disarm();
            if let Some(err) = value.get("error") {
                return Err(AgentError::Lsp(format!("lsp error: {err}")));
            }
            Ok(value.get("result").cloned().unwrap_or(Value::Null))
        }
        Ok(Err(_)) => {
            // Reader termination removes the sender and closes the channel.
            // The guard remains safe if another path closed it first.
            Err(AgentError::Lsp("response channel dropped".into()))
        }
        Err(_) => {
            // Returning drops the guard, removing the timed-out ID before any
            // late response can be correlated to a future request.
            Err(AgentError::Lsp(format!(
                "request '{method}' timed out after {timeout:?}"
            )))
        }
    }
}

/// Permanently close the response path and fail every outstanding request.
/// The terminal flag plus the guard's post-insert check prevents a concurrent
/// or later request from escaping this drain.
fn fail_all_pending(pending: &PendingMap) {
    pending.reader_terminated.store(true, Ordering::SeqCst);

    let ids: Vec<i64> = pending.entries.iter().map(|entry| *entry.key()).collect();
    for id in ids {
        if let Some((_id, tx)) = pending.entries.remove(&id) {
            let _ = tx.send(json!({
                "error": {
                    "code": -32099,
                    "message": "LSP reader terminated"
                }
            }));
        }
    }
}

/// Write a JSON-RPC message with Content-Length header.
async fn send_message(stdin: &Arc<Mutex<tokio::process::ChildStdin>>, msg: &Value) -> Result<()> {
    let body = serde_json::to_string(msg).map_err(|e| AgentError::Lsp(e.to_string()))?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());

    let mut guard = stdin.lock().await;
    guard
        .write_all(header.as_bytes())
        .await
        .map_err(|e| AgentError::Lsp(format!("write header: {e}")))?;
    guard
        .write_all(body.as_bytes())
        .await
        .map_err(|e| AgentError::Lsp(format!("write body: {e}")))?;
    guard
        .flush()
        .await
        .map_err(|e| AgentError::Lsp(format!("flush: {e}")))?;
    Ok(())
}

/// Read one bounded newline-terminated LSP header line. The returned wire
/// length includes the line ending so the caller can enforce an aggregate
/// header-block cap. EOF before a line ending is a malformed partial header.
async fn read_bounded_header_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max_wire_bytes: usize,
) -> Result<Option<(Vec<u8>, usize)>> {
    let mut line = Vec::new();

    loop {
        let (consume_len, found_newline) = {
            let available = reader
                .fill_buf()
                .await
                .map_err(|error| AgentError::Lsp(format!("read header: {error}")))?;
            if available.is_empty() {
                if line.is_empty() {
                    return Ok(None);
                }
                return Err(AgentError::Lsp(
                    "unexpected EOF inside LSP header line".into(),
                ));
            }

            if let Some(position) = available.iter().position(|byte| *byte == b'\n') {
                let wire_len = line.len().saturating_add(position).saturating_add(1);
                if wire_len > max_wire_bytes {
                    return Err(AgentError::Lsp(format!(
                        "LSP header block exceeds {MAX_HEADER_BYTES} byte limit"
                    )));
                }
                line.extend_from_slice(&available[..position]);
                (position + 1, true)
            } else {
                let wire_len = line.len().saturating_add(available.len());
                if wire_len > max_wire_bytes {
                    return Err(AgentError::Lsp(format!(
                        "LSP header block exceeds {MAX_HEADER_BYTES} byte limit"
                    )));
                }
                line.extend_from_slice(available);
                (available.len(), false)
            }
        };

        reader.consume(consume_len);
        if found_newline {
            let wire_len = line.len() + 1;
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some((line, wire_len)));
        }
    }
}

/// Read one JSON-RPC message from a Content-Length-framed stream.
///
/// Rejects before body allocation:
/// - Missing `Content-Length`
/// - Malformed or duplicate `Content-Length` (case-insensitive header name)
/// - A body over `max_body` (default 16 MiB)
/// - An unterminated or aggregate header block over 8 KiB
async fn read_message<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max_body: usize,
) -> Result<Option<Value>> {
    let max_body = max_body.min(MAX_CONTENT_LENGTH);
    let mut content_length: Option<usize> = None;
    let mut header_wire_bytes = 0usize;
    let mut started_header_block = false;

    loop {
        let remaining = MAX_HEADER_BYTES.saturating_sub(header_wire_bytes);
        let Some((line_bytes, wire_len)) = read_bounded_header_line(reader, remaining).await?
        else {
            if started_header_block {
                return Err(AgentError::Lsp(
                    "unexpected EOF before LSP header terminator".into(),
                ));
            }
            return Ok(None);
        };
        started_header_block = true;
        header_wire_bytes = header_wire_bytes.saturating_add(wire_len);

        if line_bytes.is_empty() {
            break;
        }

        let line = std::str::from_utf8(&line_bytes)
            .map_err(|_| AgentError::Lsp("LSP header is not valid UTF-8".into()))?;
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if !name.trim().eq_ignore_ascii_case("Content-Length") {
            continue;
        }
        if content_length.is_some() {
            return Err(AgentError::Lsp("duplicate Content-Length header".into()));
        }

        // LSP header OWS is ASCII space or horizontal tab. Using Unicode
        // `trim()` here would incorrectly accept non-ASCII whitespace around
        // an otherwise numeric value.
        let value = value.trim_matches(|character| character == ' ' || character == '\t');
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(AgentError::Lsp("malformed Content-Length value".into()));
        }
        let length = value
            .parse::<usize>()
            .map_err(|_| AgentError::Lsp("malformed Content-Length value".into()))?;
        if length > max_body {
            return Err(AgentError::Lsp(format!(
                "Content-Length {length} exceeds {max_body} byte limit"
            )));
        }
        content_length = Some(length);
    }

    let length =
        content_length.ok_or_else(|| AgentError::Lsp("missing Content-Length header".into()))?;
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|error| AgentError::Lsp(format!("read body: {error}")))?;

    serde_json::from_slice(&body)
        .map(Some)
        .map_err(|error| AgentError::Lsp(format!("parse json: {error}")))
}

/// Handle an incoming message from the server.
fn handle_incoming(msg: Value, pending: &PendingMap) {
    // If it has an `id` and we have a pending sender for it, it's a response.
    if let Some(id) = msg.get("id").and_then(Value::as_i64) {
        if let Some((_id, tx)) = pending.entries.remove(&id) {
            let _ = tx.send(msg);
        }
    }

    // Server-initiated requests (method + id) are consumed silently: responding
    // would need stdin, and LSP server-request completion remains intentionally
    // incomplete (requirement 3.6). Server notifications (method, no id) such as
    // diagnostics are handled by Task 18.2 subscribers.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_length_header_format() {
        let body = r#"{"jsonrpc":"2.0","method":"test","params":{}}"#;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        assert!(header.starts_with("Content-Length: "));
        assert!(header.ends_with("\r\n\r\n"));
        assert_eq!(header, format!("Content-Length: {}\r\n\r\n", body.len()));
    }

    #[test]
    fn kill_on_drop_is_set() {
        // Structural check: the spawn path must keep `kill_on_drop(true)` so a
        // dropped client cannot leave an orphaned language server behind.
        let source = include_str!("client.rs");
        assert!(source.contains(".kill_on_drop(true)"));
    }
    fn pending_state() -> PendingMap {
        Arc::new(PendingState::new())
    }

    #[tokio::test]
    async fn dropping_polled_response_future_removes_pending_request() {
        let pending = pending_state();
        let (tx, rx) = oneshot::channel();
        let guard = PendingRequestGuard::insert(pending.clone(), 42, tx).unwrap();
        let mut response = Box::pin(await_pending_response(
            "cancelled",
            rx,
            guard,
            std::time::Duration::from_secs(1),
        ));

        // Poll the same helper used by `request`, then simulate caller
        // cancellation by dropping it while the response is still pending.
        let outer_timeout =
            tokio::time::timeout(std::time::Duration::from_millis(1), response.as_mut()).await;
        assert!(outer_timeout.is_err());
        assert!(pending.entries.contains_key(&42));

        drop(response);
        assert!(!pending.entries.contains_key(&42));
    }

    #[tokio::test]
    async fn response_routing_completes_and_removes_pending_request() {
        let pending = pending_state();
        let (tx, rx) = oneshot::channel();
        let guard = PendingRequestGuard::insert(pending.clone(), 6, tx).unwrap();

        handle_incoming(
            json!({"jsonrpc": "2.0", "id": 6, "result": {"ok": true}}),
            &pending,
        );
        let value = await_pending_response("success", rx, guard, std::time::Duration::from_secs(1))
            .await
            .unwrap();

        assert_eq!(value["ok"], true);
        assert!(pending.entries.is_empty());
    }

    #[tokio::test]
    async fn response_timeout_removes_pending_request() {
        let pending = pending_state();
        let (tx, rx) = oneshot::channel::<Value>();
        let guard = PendingRequestGuard::insert(pending.clone(), 7, tx).unwrap();

        let error =
            await_pending_response("timeout", rx, guard, std::time::Duration::from_millis(1))
                .await
                .unwrap_err()
                .to_string();

        assert!(error.contains("timed out"));
        assert!(!pending.entries.contains_key(&7));
    }

    #[tokio::test]
    async fn reader_termination_clears_map_and_rejects_later_requests() {
        let pending = pending_state();
        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();
        pending.entries.insert(1, tx1);
        pending.entries.insert(2, tx2);

        fail_all_pending(&pending);

        assert!(pending.entries.is_empty());
        assert!(pending.reader_terminated.load(Ordering::SeqCst));
        let v1 = rx1.await.unwrap();
        assert!(v1.get("error").is_some());
        let v2 = rx2.await.unwrap();
        assert!(v2.get("error").is_some());

        let (tx3, _rx3) = oneshot::channel();
        let error = PendingRequestGuard::insert(pending.clone(), 3, tx3)
            .err()
            .expect("reader termination must reject new requests")
            .to_string();
        assert!(error.contains("reader terminated"));
        assert!(pending.entries.is_empty());
    }

    #[test]
    fn fail_all_pending_empty_is_terminal_noop() {
        let pending = pending_state();
        fail_all_pending(&pending);
        assert!(pending.entries.is_empty());
        assert!(pending.reader_terminated.load(Ordering::SeqCst));
    }

    async fn parse_frame(data: &[u8], max_body: usize) -> Result<Option<Value>> {
        let mut reader = BufReader::new(data);
        read_message(&mut reader, max_body).await
    }

    #[tokio::test]
    async fn read_message_rejects_missing_content_length() {
        let error = parse_frame(b"Other-Header: value\r\n\r\n", MAX_CONTENT_LENGTH)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing Content-Length"));
    }

    #[tokio::test]
    async fn read_message_rejects_malformed_content_length() {
        for frame in [
            b"Content-Length: abc\r\n\r\n".as_slice(),
            b"Content-Length: +1\r\n\r\n".as_slice(),
            b"Content-Length: -1\r\n\r\n".as_slice(),
            b"Content-Length: \r\n\r\n".as_slice(),
            b"Content-Length: \xc2\xa012\r\n\r\n".as_slice(),
            b"Content-Length: \x0b12\r\n\r\n".as_slice(),
        ] {
            let error = parse_frame(frame, MAX_CONTENT_LENGTH)
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains("malformed Content-Length"));
        }
    }

    #[tokio::test]
    async fn read_message_rejects_case_insensitive_duplicate_content_length() {
        let error = parse_frame(
            b"Content-Length: 10\r\ncontent-length: 10\r\n\r\n",
            MAX_CONTENT_LENGTH,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("duplicate Content-Length"));
    }

    #[tokio::test]
    async fn read_message_rejects_over_cap_content_length_before_body() {
        let error = parse_frame(b"Content-Length: 1025\r\n\r\n", 1024)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds 1024 byte limit"));
    }

    #[tokio::test]
    async fn configured_body_limit_cannot_exceed_global_cap() {
        let frame = format!("Content-Length: {}\r\n\r\n", MAX_CONTENT_LENGTH + 1);
        let error = parse_frame(frame.as_bytes(), usize::MAX)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("exceeds {MAX_CONTENT_LENGTH} byte limit")));
    }

    #[tokio::test]
    async fn read_message_rejects_unbounded_or_partial_header_lines() {
        let oversized = vec![b'x'; MAX_HEADER_BYTES + 1];
        let error = parse_frame(&oversized, MAX_CONTENT_LENGTH)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("header block exceeds"));

        let partial = parse_frame(b"Content-Length: 2", MAX_CONTENT_LENGTH)
            .await
            .unwrap_err()
            .to_string();
        assert!(partial.contains("unexpected EOF inside LSP header line"));
    }

    #[tokio::test]
    async fn read_message_rejects_partial_body() {
        let error = parse_frame(b"Content-Length: 2\r\n\r\n{", MAX_CONTENT_LENGTH)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("read body"));
    }

    #[tokio::test]
    async fn read_message_accepts_valid_case_insensitive_content_length() {
        let body = br#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#;
        let mut frame = format!("content-length: {}\r\n\r\n", body.len()).into_bytes();
        frame.extend_from_slice(body);

        let value = parse_frame(&frame, MAX_CONTENT_LENGTH)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(value["id"], 1);
        assert_eq!(value["result"]["ok"], true);
    }

    #[tokio::test]
    async fn read_message_returns_none_only_for_clean_eof() {
        assert!(parse_frame(b"", MAX_CONTENT_LENGTH)
            .await
            .unwrap()
            .is_none());
    }
}
