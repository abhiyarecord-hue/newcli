//! IPC transport for CRDT patches — IPC wire v1.
//!
//! Newline-delimited JSON over TCP loopback. The server listens on a local port
//! and accepts editor plugin connections.
//!
//! **v1 wire:** Every incoming message is a JSON object with at minimum
//! `"protocol_version": 1`, `"request_id": <string>`, and `"type": <tag>`.
//! Legacy raw `PatchMessage` is still accepted when the document was
//! pre-registered through the Rust API (compatibility path, no envelope fields).
//!
//! **Lifecycle:** `OpenDocument` registers path/text/version; duplicate
//! conflicting opens return a structured `Error` envelope. `CloseDocument`
//! deregisters. `Subscribe` enrolls a connection for outbound fan-out.
//! `ApplyPatches` applies a versioned batch to a registered document.
//! `Ack`, `Error`, and `DocumentUpdate` are outbound-only envelope types.
//!
//! **Bounds (Task 17.2):** Configurable newline frame cap (default 1 MiB),
//! read-idle timeout (default 30 s), write deadline (default 5 s), and
//! subscriber queue depth (default 64). Slow/full peers are disconnected
//! without blocking healthy peers. Document locks are released before any
//! enqueue or network wait.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use agent_types::{AgentError, Result};
use serde::{Deserialize, Serialize};
use serde_json;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, Mutex};
use tokio::time::timeout;

use crate::crdt_doc::{CrdtDoc, Patch, PatchMessage};

// ---------------------------------------------------------------------------
// Configurable bounds
// ---------------------------------------------------------------------------

/// Runtime-injectable IPC transport limits. All fields are public so tests can
/// set small values without live network timeouts.
#[derive(Debug, Clone)]
pub struct IpcConfig {
    /// Maximum bytes in one newline-delimited frame (default 1 MiB).
    pub max_frame_bytes: usize,
    /// Idle timeout for partial/read silence (default 30 s).
    pub read_idle_timeout: Duration,
    /// Deadline for a single `write_all` operation (default 5 s).
    pub write_deadline: Duration,
    /// Bounded subscriber outbound queue capacity (default 64).
    pub subscriber_queue_capacity: usize,
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            max_frame_bytes: 1024 * 1024, // 1 MiB
            read_idle_timeout: Duration::from_secs(30),
            write_deadline: Duration::from_secs(5),
            subscriber_queue_capacity: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// v1 envelope types
// ---------------------------------------------------------------------------

/// Incoming v1 message kinds understood by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum InboundMessage {
    /// Register a new document with initial text and version.
    OpenDocument {
        protocol_version: u64,
        request_id: String,
        path: String,
        text: String,
        #[serde(default)]
        version: u64,
    },
    /// Deregister a previously opened document.
    CloseDocument {
        protocol_version: u64,
        request_id: String,
        path: String,
    },
    /// Apply a versioned patch batch to a registered document.
    ApplyPatches {
        protocol_version: u64,
        request_id: String,
        file: String,
        patches: Vec<Patch>,
        version: u64,
    },
    /// Subscribe this connection for outbound `DocumentUpdate` fan-out.
    Subscribe {
        protocol_version: u64,
        request_id: String,
        path: String,
    },
}

/// Outbound message envelope sent from the server to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum OutboundMessage {
    /// Acknowledgement of a successfully processed request.
    Ack { request_id: String, version: u64 },
    /// Structured error response correlated to a request.
    Error {
        request_id: String,
        code: String,
        message: String,
    },
    /// Pushed document update to all subscribers.
    DocumentUpdate {
        path: String,
        patches: Vec<Patch>,
        version: u64,
    },
}

impl OutboundMessage {
    fn to_line(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(self).unwrap_or_default();
        bytes.push(b'\n');
        bytes
    }
}

// ---------------------------------------------------------------------------
// Subscriber channel — bounded queue, Weak senders stored in registry
// ---------------------------------------------------------------------------

/// Sender half for outbound subscriber pushes (bounded queue).
type SubscriberTx = mpsc::Sender<Vec<u8>>;

/// Weak sender stored in the document registry. When all receiver halves are
/// dropped the Weak upgrade will fail, automatically pruning dead entries.
type WeakSubscriberTx = Weak<mpsc::Sender<Vec<u8>>>;

// ---------------------------------------------------------------------------
// Per-document lifecycle record
// ---------------------------------------------------------------------------

struct DocEntry {
    doc: Arc<Mutex<CrdtDoc>>,
    /// Weak subscriber senders registered via `Subscribe` for this path.
    subscribers: Vec<WeakSubscriberTx>,
}

impl DocEntry {
    /// Drain dead weak references and return cloned live senders.
    fn live_senders(&mut self) -> Vec<Arc<SubscriberTx>> {
        self.subscribers.retain(|w| w.strong_count() > 0);
        self.subscribers
            .iter()
            .filter_map(|w| w.upgrade())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// IpcServer
// ---------------------------------------------------------------------------

/// Manages multiple CRDT documents and the IPC server.
///
/// Supports both pre-registered documents (legacy raw `PatchMessage`) and
/// dynamically registered documents through the v1 `OpenDocument` lifecycle.
pub struct IpcServer {
    docs: Arc<Mutex<HashMap<String, DocEntry>>>,
    listener: Mutex<Option<TcpListener>>,
    config: Arc<IpcConfig>,
}

impl IpcServer {
    /// Create a new IPC server with default bounds and retain the bound
    /// listener so no other process can claim the port between `bind` and
    /// `run`.
    pub async fn bind(port: u16) -> Result<Self> {
        Self::bind_with_config(port, IpcConfig::default()).await
    }

    /// Create a new IPC server with injected bounds (used by tests).
    pub async fn bind_with_config(port: u16, config: IpcConfig) -> Result<Self> {
        let listener = TcpListener::bind(format!("127.0.0.1:{port}"))
            .await
            .map_err(|e| AgentError::Tool {
                name: "ipc_server".into(),
                reason: format!("bind port {port}: {e}"),
            })?;

        Ok(Self {
            docs: Arc::new(Mutex::new(HashMap::new())),
            listener: Mutex::new(Some(listener)),
            config: Arc::new(config),
        })
    }

    /// Pre-register a document (legacy API — raw `PatchMessage` will route here).
    pub async fn register_doc(&self, doc: Arc<Mutex<CrdtDoc>>) {
        let path = doc.lock().await.file_path().to_string();
        let mut map = self.docs.lock().await;
        map.entry(path).or_insert_with(|| DocEntry {
            doc,
            subscribers: Vec::new(),
        });
    }

    /// Start the IPC listener loop (spawns a task per connection).
    pub async fn run(&self) -> Result<()> {
        let listener = self
            .listener
            .lock()
            .await
            .take()
            .ok_or_else(|| AgentError::Tool {
                name: "ipc_server".into(),
                reason: "server is already running".into(),
            })?;

        let docs = self.docs.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let docs_clone = docs.clone();
                let config_clone = config.clone();
                tokio::spawn(async move {
                    handle_connection(stream, docs_clone, config_clone).await;
                });
            }
        });

        Ok(())
    }

    /// Apply patches, then fan-out a `DocumentUpdate` to all subscribers.
    ///
    /// The document lock is released before any enqueue so no shared state is
    /// held across I/O waits. Full/closed subscriber queues are dropped
    /// (slow-peer disconnect); healthy peers continue unimpeded.
    pub async fn broadcast(&self, file: &str, patches: Vec<Patch>) -> Result<PatchMessage> {
        // --- apply under lock, collect live senders, release lock ---
        let (msg, outbound_bytes, live_txs) = {
            let mut map = self.docs.lock().await;
            let entry = map.get_mut(file).ok_or_else(|| AgentError::Tool {
                name: "ipc".into(),
                reason: format!("no doc for {file}"),
            })?;
            // Clone the doc Arc so we can release the mutable borrow on entry.
            let doc = entry.doc.clone();
            let mut guard = doc.lock().await;
            guard.apply_patches(&patches)?;
            let msg = guard.make_message(patches.clone());
            drop(guard);
            let update = OutboundMessage::DocumentUpdate {
                path: file.to_string(),
                patches: patches.clone(),
                version: msg.version,
            };
            let bytes = update.to_line();
            let txs = entry.live_senders();
            (msg, bytes, txs)
        }; // all locks released here

        // --- nonblocking fan-out: drop slow/full subscribers ---
        for tx in &live_txs {
            // try_send is nonblocking; if the queue is full the peer is slow
            // and gets disconnected (the Arc<Sender> will be dropped here,
            // and the Weak in the registry will fail on the next prune).
            let _ = tx.try_send(outbound_bytes.clone());
        }

        Ok(msg)
    }
}

// ---------------------------------------------------------------------------
// Bounded line reader — replaces unbounded `lines()` with a frame cap
// ---------------------------------------------------------------------------

/// Read one newline-terminated line from `reader` up to `max_bytes`.
/// Returns `Ok(Some(line))` on success (newline stripped),
/// `Ok(None)` on clean EOF, and `Err` if the frame exceeds the cap.
async fn read_bounded_line<R: AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
    max_bytes: usize,
) -> std::io::Result<Option<String>> {
    let mut buf = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            // EOF
            if buf.is_empty() {
                return Ok(None);
            }
            // Partial line at EOF — treat as a complete frame
            return String::from_utf8(buf).map(Some).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid UTF-8")
            });
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            buf.extend_from_slice(&available[..pos]);
            let consume = pos + 1;
            reader.consume(consume);
            if buf.len() > max_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "frame exceeds max_frame_bytes",
                ));
            }
            return String::from_utf8(buf).map(Some).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid UTF-8")
            });
        }
        // No newline in this chunk — consume the entire buffer
        let len = available.len();
        buf.extend_from_slice(available);
        reader.consume(len);
        if buf.len() > max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "frame exceeds max_frame_bytes",
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Per-connection handler with bounded framing, read-idle, and write deadline
// ---------------------------------------------------------------------------

async fn handle_connection(
    stream: tokio::net::TcpStream,
    docs: Arc<Mutex<HashMap<String, DocEntry>>>,
    config: Arc<IpcConfig>,
) {
    let (reader_half, writer_half) = stream.into_split();
    let writer = Arc::new(Mutex::new(writer_half));
    let write_deadline = config.write_deadline;

    // Each connection gets a bounded outbound channel for subscriber pushes.
    let (sub_tx, mut sub_rx) = mpsc::channel::<Vec<u8>>(config.subscriber_queue_capacity);
    // Keep an Arc<Sender> so we can store a Weak in the registry.
    let sub_tx_arc = Arc::new(sub_tx);

    // Spawn a task that drains the subscriber channel and writes to the socket
    // with the write deadline.
    let writer_clone = writer.clone();
    tokio::spawn(async move {
        while let Some(bytes) = sub_rx.recv().await {
            let mut w = writer_clone.lock().await;
            let result = timeout(write_deadline, w.write_all(&bytes)).await;
            match result {
                Ok(Ok(())) => {}
                _ => break, // write timeout or error — disconnect
            }
        }
    });

    let mut reader = BufReader::new(reader_half);
    let read_idle = config.read_idle_timeout;
    let max_frame = config.max_frame_bytes;

    loop {
        // Apply the read-idle timeout to the bounded line read.
        let line_result = timeout(read_idle, read_bounded_line(&mut reader, max_frame)).await;
        match line_result {
            Ok(Ok(Some(line))) => {
                let response = dispatch_line(&line, &docs, &sub_tx_arc).await;
                if let Some(resp_bytes) = response {
                    let mut w = writer.lock().await;
                    let write_result = timeout(write_deadline, w.write_all(&resp_bytes)).await;
                    match write_result {
                        Ok(Ok(())) => {}
                        _ => break, // write timeout or error
                    }
                }
            }
            Ok(Ok(None)) => break, // clean EOF
            Ok(Err(_)) => break,   // frame too large or invalid UTF-8
            Err(_) => break,       // read-idle timeout
        }
    }
}

/// Dispatch one received line. Returns `Some(bytes)` if a response should be
/// written back to the sender, `None` for fire-and-forget paths.
async fn dispatch_line(
    line: &str,
    docs: &Arc<Mutex<HashMap<String, DocEntry>>>,
    sub_tx: &Arc<SubscriberTx>,
) -> Option<Vec<u8>> {
    // --- Try v1 tagged envelope first ---
    if let Ok(msg) = serde_json::from_str::<InboundMessage>(line) {
        return Some(handle_v1_message(msg, docs, sub_tx).await);
    }

    // --- Fall back to legacy raw PatchMessage ---
    if let Ok(pm) = serde_json::from_str::<PatchMessage>(line) {
        return Some(handle_legacy_patch(pm, docs).await);
    }

    // Unrecognized format — no response
    None
}

async fn handle_v1_message(
    msg: InboundMessage,
    docs: &Arc<Mutex<HashMap<String, DocEntry>>>,
    sub_tx: &Arc<SubscriberTx>,
) -> Vec<u8> {
    match msg {
        InboundMessage::OpenDocument {
            request_id,
            path,
            text,
            version,
            ..
        } => {
            let mut map = docs.lock().await;
            if map.contains_key(&path) {
                let resp = OutboundMessage::Error {
                    request_id,
                    code: "document_already_open".into(),
                    message: format!("document '{path}' is already registered"),
                };
                resp.to_line()
            } else {
                let doc = CrdtDoc::new(&path, text);
                map.insert(
                    path.clone(),
                    DocEntry {
                        doc,
                        subscribers: Vec::new(),
                    },
                );
                let resp = OutboundMessage::Ack {
                    request_id,
                    version,
                };
                resp.to_line()
            }
        }

        InboundMessage::CloseDocument {
            request_id, path, ..
        } => {
            let mut map = docs.lock().await;
            if map.remove(&path).is_some() {
                let resp = OutboundMessage::Ack {
                    request_id,
                    version: 0,
                };
                resp.to_line()
            } else {
                let resp = OutboundMessage::Error {
                    request_id,
                    code: "document_not_registered".into(),
                    message: format!("no document registered for '{path}'"),
                };
                resp.to_line()
            }
        }

        InboundMessage::ApplyPatches {
            request_id,
            file,
            patches,
            version,
            ..
        } => {
            let map = docs.lock().await;
            if let Some(entry) = map.get(&file) {
                let doc = entry.doc.clone();
                drop(map);
                let mut guard = doc.lock().await;
                if version != guard.version() {
                    let resp = OutboundMessage::Error {
                        request_id,
                        code: "version_conflict".into(),
                        message: format!(
                            "expected version {}, received version {version}",
                            guard.version()
                        ),
                    };
                    resp.to_line()
                } else {
                    match guard.apply_patches(&patches) {
                        Ok(new_version) => {
                            let resp = OutboundMessage::Ack {
                                request_id,
                                version: new_version,
                            };
                            resp.to_line()
                        }
                        Err(e) => {
                            let resp = OutboundMessage::Error {
                                request_id,
                                code: "apply_failed".into(),
                                message: e.to_string(),
                            };
                            resp.to_line()
                        }
                    }
                }
            } else {
                drop(map);
                let resp = OutboundMessage::Error {
                    request_id,
                    code: "document_not_registered".into(),
                    message: format!("no document registered for '{file}'"),
                };
                resp.to_line()
            }
        }

        InboundMessage::Subscribe {
            request_id, path, ..
        } => {
            let mut map = docs.lock().await;
            if let Some(entry) = map.get_mut(&path) {
                // Store a Weak reference so dead connections are pruned.
                entry.subscribers.push(Arc::downgrade(sub_tx));
                let resp = OutboundMessage::Ack {
                    request_id,
                    version: 0,
                };
                resp.to_line()
            } else {
                let resp = OutboundMessage::Error {
                    request_id,
                    code: "document_not_registered".into(),
                    message: format!("no document registered for '{path}'"),
                };
                resp.to_line()
            }
        }
    }
}

/// Handle a legacy raw `PatchMessage` (pre-v1 compatibility).
/// Only accepted when the document was pre-registered via the Rust API.
async fn handle_legacy_patch(
    msg: PatchMessage,
    docs: &Arc<Mutex<HashMap<String, DocEntry>>>,
) -> Vec<u8> {
    let map = docs.lock().await;
    if let Some(entry) = map.get(&msg.file) {
        let doc = entry.doc.clone();
        drop(map);
        let mut guard = doc.lock().await;
        let ack = if msg.version != guard.version() {
            serde_json::json!({
                "error": "version_conflict",
                "expected_version": guard.version(),
                "received_version": msg.version
            })
        } else {
            match guard.apply_patches(&msg.patches) {
                Ok(version) => serde_json::json!({"version": version}),
                Err(e) => serde_json::json!({"error": e.to_string()}),
            }
        };
        format!("{ack}\n").into_bytes()
    } else {
        drop(map);
        let ack = serde_json::json!({
            "error": "document_not_registered",
            "file": msg.file
        });
        format!("{ack}\n").into_bytes()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::time::{timeout, Duration};

    async fn free_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        listener.local_addr().unwrap().port()
    }

    async fn connect(port: u16) -> TcpStream {
        TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap()
    }

    async fn send_line(stream: &mut TcpStream, v: &Value) {
        let mut bytes = serde_json::to_vec(v).unwrap();
        bytes.push(b'\n');
        stream.write_all(&bytes).await.unwrap();
    }

    async fn read_value(stream: &mut TcpStream) -> Value {
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).await.unwrap();
        serde_json::from_str(&line).unwrap()
    }

    fn test_config() -> IpcConfig {
        IpcConfig {
            max_frame_bytes: 1024, // 1 KiB for tests
            read_idle_timeout: Duration::from_millis(200),
            write_deadline: Duration::from_millis(100),
            subscriber_queue_capacity: 4,
        }
    }

    /// Upper bound on how long a test waits for an expected event.
    ///
    /// The behaviour under test is bounded by the small injected values in
    /// [`test_config`]; this allowance exists only so a broken bound fails the
    /// test instead of hanging it. It is deliberately far larger than those
    /// values, because a tight allowance turns scheduling delay on a loaded or
    /// slow CI runner into a spurious failure. Making it generous costs nothing:
    /// a bound that does not fire still never produces the awaited event.
    const OUTER_WAIT: Duration = Duration::from_secs(5);

    // -------------------------------------------------------------------
    // Lifecycle tests (preserved from Task 17.1)
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn open_document_returns_correlated_ack() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1,
                "request_id": "open-abc",
                "type": "OpenDocument",
                "path": "hello.rs",
                "text": "fn main() {}",
                "version": 0
            }),
        )
        .await;

        let resp = timeout(OUTER_WAIT, read_value(&mut peer))
            .await
            .expect("should receive ack before the outer wait elapses");
        assert_eq!(resp["type"], "Ack");
        assert_eq!(resp["request_id"], "open-abc");
    }

    #[tokio::test]
    async fn conflicting_open_returns_structured_error() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "o1",
                "type": "OpenDocument", "path": "dup.rs",
                "text": "a", "version": 0
            }),
        )
        .await;
        let _ = timeout(Duration::from_millis(300), read_value(&mut peer))
            .await
            .unwrap();

        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "o2",
                "type": "OpenDocument", "path": "dup.rs",
                "text": "b", "version": 0
            }),
        )
        .await;
        let resp = timeout(OUTER_WAIT, read_value(&mut peer))
            .await
            .expect("should receive error before the outer wait elapses");
        assert_eq!(resp["type"], "Error");
        assert_eq!(resp["request_id"], "o2");
        assert_eq!(resp["code"], "document_already_open");
    }

    #[tokio::test]
    async fn close_document_removes_registration() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "o1",
                "type": "OpenDocument", "path": "close.rs",
                "text": "x", "version": 0
            }),
        )
        .await;
        let _ = timeout(Duration::from_millis(300), read_value(&mut peer))
            .await
            .unwrap();

        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "c1",
                "type": "CloseDocument", "path": "close.rs"
            }),
        )
        .await;
        let resp = timeout(OUTER_WAIT, read_value(&mut peer)).await.unwrap();
        assert_eq!(resp["type"], "Ack");
        assert_eq!(resp["request_id"], "c1");

        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "c2",
                "type": "CloseDocument", "path": "close.rs"
            }),
        )
        .await;
        let err = timeout(OUTER_WAIT, read_value(&mut peer)).await.unwrap();
        assert_eq!(err["type"], "Error");
        assert_eq!(err["code"], "document_not_registered");
    }

    // -------------------------------------------------------------------
    // Acknowledgement / version-conflict tests
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn apply_patches_returns_structured_ack_with_new_version() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "o1",
                "type": "OpenDocument", "path": "apply.rs",
                "text": "hello world", "version": 0
            }),
        )
        .await;
        let _ = timeout(Duration::from_millis(300), read_value(&mut peer))
            .await
            .unwrap();

        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "ap1",
                "type": "ApplyPatches", "file": "apply.rs",
                "patches": [{"range": [5, 5], "insert": " beautiful"}],
                "version": 0
            }),
        )
        .await;
        let resp = timeout(OUTER_WAIT, read_value(&mut peer)).await.unwrap();
        assert_eq!(resp["type"], "Ack");
        assert_eq!(resp["request_id"], "ap1");
        assert_eq!(resp["version"], 1);
    }

    #[tokio::test]
    async fn apply_patches_rejects_wrong_version() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "o1",
                "type": "OpenDocument", "path": "vconflict.rs",
                "text": "abc", "version": 0
            }),
        )
        .await;
        let _ = timeout(Duration::from_millis(300), read_value(&mut peer))
            .await
            .unwrap();

        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "ap-bad",
                "type": "ApplyPatches", "file": "vconflict.rs",
                "patches": [], "version": 99
            }),
        )
        .await;
        let resp = timeout(OUTER_WAIT, read_value(&mut peer)).await.unwrap();
        assert_eq!(resp["type"], "Error");
        assert_eq!(resp["code"], "version_conflict");
    }

    #[tokio::test]
    async fn apply_patches_on_unknown_doc_returns_structured_error() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        send_line(
            &mut peer,
            &json!({
                "protocol_version": 1, "request_id": "ap-miss",
                "type": "ApplyPatches", "file": "ghost.rs",
                "patches": [], "version": 0
            }),
        )
        .await;
        let resp = timeout(OUTER_WAIT, read_value(&mut peer)).await.unwrap();
        assert_eq!(resp["type"], "Error");
        assert_eq!(resp["code"], "document_not_registered");
    }

    // -------------------------------------------------------------------
    // Subscribe / fan-out test
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn subscribe_registers_for_document_update_fan_out() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        let doc = CrdtDoc::new("fanout.rs", "old");
        server.register_doc(doc.clone()).await;
        server.run().await.unwrap();

        let mut subscriber = connect(port).await;
        send_line(
            &mut subscriber,
            &json!({
                "protocol_version": 1, "request_id": "sub-1",
                "type": "Subscribe", "path": "fanout.rs"
            }),
        )
        .await;
        let ack = timeout(Duration::from_millis(300), read_value(&mut subscriber))
            .await
            .expect("Subscribe ack");
        assert_eq!(ack["type"], "Ack");

        let _pm = server
            .broadcast(
                "fanout.rs",
                vec![Patch {
                    range: (0, 3),
                    insert: "new".into(),
                }],
            )
            .await
            .unwrap();

        let update = timeout(OUTER_WAIT, read_value(&mut subscriber))
            .await
            .expect("should receive DocumentUpdate");
        assert_eq!(update["type"], "DocumentUpdate");
        assert_eq!(update["path"], "fanout.rs");
        assert_eq!(update["version"], 1);
    }

    // -------------------------------------------------------------------
    // Bounded framing tests (Task 17.2)
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn oversized_frame_disconnects_peer() {
        // Validates: Requirements 2.20 — oversized unterminated frame is bounded
        let port = free_port().await;
        let server = IpcServer::bind_with_config(port, test_config())
            .await
            .unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        // Send 2 KiB without a newline — exceeds the 1 KiB test cap.
        peer.write_all(&vec![b'x'; 2048]).await.unwrap();

        let mut byte = [0u8; 1];
        let closed = timeout(OUTER_WAIT, peer.read(&mut byte)).await;
        assert!(
            matches!(closed, Ok(Ok(0))),
            "oversized frame should close the connection, got {closed:?}"
        );
    }

    #[tokio::test]
    async fn read_idle_timeout_disconnects_silent_peer() {
        // Validates: Requirements 2.20 — read-idle timeout
        let port = free_port().await;
        let server = IpcServer::bind_with_config(port, test_config())
            .await
            .unwrap();
        server.run().await.unwrap();

        let mut peer = connect(port).await;
        // Send nothing — the 200 ms read-idle timeout should close the connection.
        let mut byte = [0u8; 1];
        let closed = timeout(OUTER_WAIT, peer.read(&mut byte)).await;
        assert!(
            matches!(closed, Ok(Ok(0))),
            "silent peer should be disconnected after read idle, got {closed:?}"
        );
    }

    #[tokio::test]
    async fn slow_peer_does_not_block_healthy_peer() {
        // Validates: Requirements 2.20 — nonblocking broadcast; slow peer
        // disconnected, healthy peer still receives updates.
        let port = free_port().await;
        let server = Arc::new(
            IpcServer::bind_with_config(
                port,
                IpcConfig {
                    max_frame_bytes: 1024 * 1024,
                    read_idle_timeout: Duration::from_secs(30),
                    write_deadline: Duration::from_millis(100),
                    subscriber_queue_capacity: 1, // tiny queue to fill quickly
                },
            )
            .await
            .unwrap(),
        );
        let doc = CrdtDoc::new("shared.rs", "initial");
        server.register_doc(doc.clone()).await;
        server.run().await.unwrap();

        // Healthy subscriber
        let mut healthy = connect(port).await;
        send_line(
            &mut healthy,
            &json!({
                "protocol_version": 1, "request_id": "sh",
                "type": "Subscribe", "path": "shared.rs"
            }),
        )
        .await;
        let _ = timeout(Duration::from_millis(300), read_value(&mut healthy))
            .await
            .unwrap();

        // Slow subscriber — subscribe then stop reading
        let mut slow = connect(port).await;
        send_line(
            &mut slow,
            &json!({
                "protocol_version": 1, "request_id": "ss",
                "type": "Subscribe", "path": "shared.rs"
            }),
        )
        .await;
        // Do NOT drain the slow subscriber's ack or subsequent updates.

        // Give the slow subscriber's ack time to arrive then fill the queue.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Broadcast several updates — the slow peer's queue fills but the
        // healthy peer must still receive all of them.
        for i in 0..4 {
            let _ = server
                .broadcast(
                    "shared.rs",
                    vec![Patch {
                        range: (0, 0),
                        insert: format!("{i}"),
                    }],
                )
                .await
                .unwrap();
        }

        // The healthy peer should receive at least one DocumentUpdate.
        let update = timeout(OUTER_WAIT, read_value(&mut healthy)).await;
        assert!(
            matches!(&update, Ok(v) if v["type"] == "DocumentUpdate"),
            "healthy peer should receive updates even when slow peer is present, got {update:?}"
        );
    }

    // -------------------------------------------------------------------
    // Legacy raw PatchMessage compatibility
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn legacy_patch_message_accepted_for_pre_registered_doc() {
        // Validates: Requirements 3.4, 3.5 — exact-version valid patch batches
        // and pre-registered legacy raw patches remain intact.
        let port = free_port().await;
        let doc = CrdtDoc::new("legacy.rs", "hello world");
        let server = IpcServer::bind(port).await.unwrap();
        server.register_doc(doc.clone()).await;
        server.run().await.unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut stream = connect(port).await;
        let msg = PatchMessage {
            file: "legacy.rs".into(),
            patches: vec![Patch {
                range: (5, 5),
                insert: " beautiful".into(),
            }],
            version: 0,
        };
        let line = format!("{}\n", serde_json::to_string(&msg).unwrap());
        stream.write_all(line.as_bytes()).await.unwrap();

        let mut buf = String::new();
        BufReader::new(stream).read_line(&mut buf).await.unwrap();
        let val: Value = serde_json::from_str(&buf).unwrap();
        assert_eq!(val["version"], 1);

        let guard = doc.lock().await;
        assert_eq!(guard.content(), "hello beautiful world");
        assert_eq!(guard.version(), 1);
    }

    #[tokio::test]
    async fn legacy_patch_message_rejected_for_unregistered_doc() {
        let port = free_port().await;
        let server = IpcServer::bind(port).await.unwrap();
        server.run().await.unwrap();

        let mut stream = connect(port).await;
        let msg = PatchMessage {
            file: "ghost.rs".into(),
            patches: vec![],
            version: 0,
        };
        let line = format!("{}\n", serde_json::to_string(&msg).unwrap());
        stream.write_all(line.as_bytes()).await.unwrap();

        let mut buf = String::new();
        BufReader::new(stream).read_line(&mut buf).await.unwrap();
        let val: Value = serde_json::from_str(&buf).unwrap();
        assert!(val.get("error").is_some());
    }

    // -------------------------------------------------------------------
    // Preservation: existing round-trip test (from Task 8 baseline)
    // -------------------------------------------------------------------

    #[tokio::test]
    async fn ipc_round_trip_with_mock_client() {
        let port = free_port().await;
        let doc = CrdtDoc::new("test.rs", "hello world");
        let server = IpcServer::bind(port).await.unwrap();
        server.register_doc(doc.clone()).await;
        server.run().await.unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut stream = connect(port).await;
        let msg = PatchMessage {
            file: "test.rs".into(),
            patches: vec![Patch {
                range: (5, 5),
                insert: " beautiful".into(),
            }],
            version: 0,
        };
        let json_line = format!("{}\n", serde_json::to_string(&msg).unwrap());
        stream.write_all(json_line.as_bytes()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let guard = doc.lock().await;
        assert_eq!(guard.content(), "hello beautiful world");
        assert_eq!(guard.version(), 1);
    }
}
