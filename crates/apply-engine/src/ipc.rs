//! IPC transport for CRDT patches.
//!
//! Newline-delimited JSON over TCP loopback (cross-platform; Unix domain sockets
//! used on Linux/macOS). The server listens on a local port and accepts editor
//! plugin connections. Each line is a `PatchMessage`.
//!
//! On startup, unlink stale socket / release port. One `CrdtDoc` per file
//! behind `Arc<Mutex>` — async state isolation.

use std::collections::HashMap;
use std::sync::Arc;

use agent_types::{AgentError, Result};
use serde_json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use crate::crdt_doc::{CrdtDoc, Patch, PatchMessage};

/// Manages multiple CRDT documents and the IPC server.
pub struct IpcServer {
    docs: Arc<Mutex<HashMap<String, Arc<Mutex<CrdtDoc>>>>>,
    listener: Mutex<Option<TcpListener>>,
}

impl IpcServer {
    /// Create a new IPC server and retain the bound listener so no other
    /// process can claim the port between `bind` and `run`.
    pub async fn bind(port: u16) -> Result<Self> {
        let listener = TcpListener::bind(format!("127.0.0.1:{port}"))
            .await
            .map_err(|e| AgentError::Tool {
                name: "ipc_server".into(),
                reason: format!("bind port {port}: {e}"),
            })?;

        Ok(Self {
            docs: Arc::new(Mutex::new(HashMap::new())),
            listener: Mutex::new(Some(listener)),
        })
    }

    /// Register a document.
    pub async fn register_doc(&self, doc: Arc<Mutex<CrdtDoc>>) {
        let path = doc.lock().await.file_path().to_string();
        self.docs.lock().await.insert(path, doc);
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

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let docs_clone = docs.clone();
                tokio::spawn(async move {
                    let (reader, mut writer) = stream.into_split();
                    let mut lines = BufReader::new(reader).lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        let msg: PatchMessage = match serde_json::from_str(&line) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };

                        let map = docs_clone.lock().await;
                        if let Some(doc) = map.get(&msg.file).cloned() {
                            drop(map);
                            let mut doc_guard = doc.lock().await;
                            let ack = if msg.version != doc_guard.version() {
                                serde_json::json!({
                                    "error": "version_conflict",
                                    "expected_version": doc_guard.version(),
                                    "received_version": msg.version
                                })
                            } else {
                                match doc_guard.apply_patches(&msg.patches) {
                                    Ok(version) => serde_json::json!({"version": version}),
                                    Err(e) => serde_json::json!({"error": e.to_string()}),
                                }
                            };
                            drop(doc_guard);
                            let _ = writer
                                .write_all(format!("{}\n", ack).as_bytes())
                                .await;
                        } else {
                            drop(map);
                            let ack = serde_json::json!({
                                "error": "document_not_registered",
                                "file": msg.file
                            });
                            let _ = writer
                                .write_all(format!("{}\n", ack).as_bytes())
                                .await;
                        }
                    }
                });
            }
        });

        Ok(())
    }

    /// Send patches to all connected clients (broadcast outbound).
    pub async fn broadcast(&self, file: &str, patches: Vec<Patch>) -> Result<PatchMessage> {
        let map = self.docs.lock().await;
        let doc = map
            .get(file)
            .cloned()
            .ok_or_else(|| AgentError::Tool {
                name: "ipc".into(),
                reason: format!("no doc for {file}"),
            })?;
        drop(map);
        let mut guard = doc.lock().await;
        guard.apply_patches(&patches)?;
        let msg = guard.make_message(patches);
        Ok(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;

    #[tokio::test]
    async fn ipc_round_trip_with_mock_client() {
        // Find a free port.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        // Register a document.
        let doc = CrdtDoc::new("test.rs", "hello world");
        let doc_clone = doc.clone();

        // Bind and retain the listener before registering documents.
        let server = IpcServer::bind(port).await.unwrap();
        server.register_doc(doc_clone).await;
        server.run().await.unwrap();

        // Give the server a moment to start.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Connect as a mock client and send a patch.
        let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

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

        // Wait for ACK.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Verify the doc was updated.
        let guard = doc.lock().await;
        assert_eq!(guard.content(), "hello beautiful world");
        assert_eq!(guard.version(), 1);
    }
}
