//! Bounded MCP client with correlated JSON-RPC routing over supervised stdio.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard};
use std::time::Duration;

use agent_types::{AgentError, Result, ToolSchema};
use sandbox::{ProcessSupervisor, SupervisedStdioChild};
use serde_json::{json, Value};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::schema::{JsonRpcError, JsonRpcNotification, JsonRpcResponse, RpcId, ValidatedMcpTool};

pub const DEFAULT_MAX_JSON_LINE_BYTES: usize = 8 * 1024 * 1024;
pub const DEFAULT_INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
pub const DEFAULT_WRITER_QUEUE_CAPACITY: usize = 64;
pub const DEFAULT_NOTIFICATION_QUEUE_CAPACITY: usize = 64;

/// Injectable MCP transport bounds. Values may be reduced for embedders and
/// deterministic tests, but an unbounded or larger-than-supported value is
/// rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpClientConfig {
    pub max_json_line_bytes: usize,
    pub initialization_timeout: Duration,
    pub request_timeout: Duration,
    pub writer_queue_capacity: usize,
    pub notification_queue_capacity: usize,
}

impl Default for McpClientConfig {
    fn default() -> Self {
        Self {
            max_json_line_bytes: DEFAULT_MAX_JSON_LINE_BYTES,
            initialization_timeout: DEFAULT_INITIALIZATION_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            writer_queue_capacity: DEFAULT_WRITER_QUEUE_CAPACITY,
            notification_queue_capacity: DEFAULT_NOTIFICATION_QUEUE_CAPACITY,
        }
    }
}

impl McpClientConfig {
    fn validate(&self) -> Result<()> {
        validate_nonzero_bound(
            "max_json_line_bytes",
            self.max_json_line_bytes,
            DEFAULT_MAX_JSON_LINE_BYTES,
        )?;
        validate_nonzero_bound(
            "writer_queue_capacity",
            self.writer_queue_capacity,
            DEFAULT_WRITER_QUEUE_CAPACITY,
        )?;
        validate_nonzero_bound(
            "notification_queue_capacity",
            self.notification_queue_capacity,
            DEFAULT_NOTIFICATION_QUEUE_CAPACITY,
        )?;
        validate_timeout(
            "initialization_timeout",
            self.initialization_timeout,
            DEFAULT_INITIALIZATION_TIMEOUT,
        )?;
        validate_timeout(
            "request_timeout",
            self.request_timeout,
            DEFAULT_REQUEST_TIMEOUT,
        )?;
        Ok(())
    }
}

fn validate_nonzero_bound(name: &str, value: usize, maximum: usize) -> Result<()> {
    if value == 0 || value > maximum {
        return Err(mcp_error(format!(
            "{name} must be in 1..={maximum}, got {value}"
        )));
    }
    Ok(())
}

fn validate_timeout(name: &str, value: Duration, maximum: Duration) -> Result<()> {
    if value.is_zero() || value > maximum {
        return Err(mcp_error(format!(
            "{name} must be non-zero and no greater than {maximum:?}, got {value:?}"
        )));
    }
    Ok(())
}

pub struct McpClient {
    process: StdMutex<Option<SupervisedStdioChild>>,
    writer: Option<mpsc::Sender<WriteRequest>>,
    writer_task: Option<JoinHandle<()>>,
    reader_task: Option<JoinHandle<()>>,
    router: Arc<RouterState>,
    notifications: Mutex<mpsc::Receiver<JsonRpcNotification>>,
    cancel: CancellationToken,
    next_id: AtomicI64,
    config: McpClientConfig,
    tools: Vec<ToolSchema>,
    discovered_tools: Vec<ValidatedMcpTool>,
}

impl McpClient {
    /// Connect using the production MCP transport bounds.
    pub async fn connect(cmd: &str, args: &[&str], server_name: &str) -> Result<Self> {
        Self::connect_with_config(cmd, args, server_name, McpClientConfig::default()).await
    }

    /// Connect with explicit bounded transport limits. This is intended for
    /// embedders and deterministic fake-server tests.
    pub async fn connect_with_config(
        cmd: &str,
        args: &[&str],
        server_name: &str,
        config: McpClientConfig,
    ) -> Result<Self> {
        let mut command = Command::new(cmd);
        command.args(args);
        Self::connect_command_with_config(command, server_name, config).await
    }

    /// Connect a preconfigured command after its caller has completed any
    /// required trust decision. This preserves the supervised process owner
    /// while allowing a trusted caller to set the workspace and explicit
    /// environment without adding another spawn path.
    pub async fn connect_command(command: Command, server_name: &str) -> Result<Self> {
        Self::connect_command_with_config(command, server_name, McpClientConfig::default()).await
    }

    /// Connect a preconfigured command with explicit bounded transport limits.
    pub async fn connect_command_with_config(
        command: Command,
        server_name: &str,
        config: McpClientConfig,
    ) -> Result<Self> {
        config.validate()?;

        let command_label = command
            .as_std()
            .get_program()
            .to_string_lossy()
            .into_owned();
        let mut process = ProcessSupervisor::default()
            .spawn_stdio(command)
            .map_err(|error| mcp_error(format!("spawn '{command_label}': {error}")))?;
        let stdin = process
            .take_stdin()
            .map_err(|error| mcp_error(format!("take stdin: {error}")))?;
        let stdout = process
            .take_stdout()
            .map_err(|error| mcp_error(format!("take stdout: {error}")))?;

        let router = Arc::new(RouterState::default());
        let cancel = CancellationToken::new();
        let (writer, writer_rx) = mpsc::channel(config.writer_queue_capacity);
        let (notification_tx, notifications) = mpsc::channel(config.notification_queue_capacity);
        let writer_task = tokio::spawn(writer_loop(
            stdin,
            writer_rx,
            router.clone(),
            cancel.clone(),
        ));
        let reader_task = tokio::spawn(reader_loop(
            BufReader::new(stdout),
            config.max_json_line_bytes,
            router.clone(),
            notification_tx,
            cancel.clone(),
        ));

        let mut client = Self {
            process: StdMutex::new(Some(process)),
            writer: Some(writer),
            writer_task: Some(writer_task),
            reader_task: Some(reader_task),
            router,
            notifications: Mutex::new(notifications),
            cancel,
            next_id: AtomicI64::new(1),
            config,
            tools: Vec::new(),
            discovered_tools: Vec::new(),
        };

        client
            .request_with_timeout(
                "initialize",
                json!({"capabilities": {}}),
                client.config.initialization_timeout,
            )
            .await?;

        let tools_result = client
            .request_with_timeout(
                "tools/list",
                json!({}),
                client.config.initialization_timeout,
            )
            .await?;
        if let Some(tools_arr) = tools_result.get("tools").and_then(Value::as_array) {
            for raw_tool in tools_arr {
                if let Ok(tool) = ValidatedMcpTool::validate(raw_tool.clone(), server_name) {
                    client.tools.push(tool.model_schema());
                    client.discovered_tools.push(tool);
                }
            }
        }

        Ok(client)
    }

    /// Get the model-facing discovered tools. These schemas expose only the
    /// collision-safe local aliases.
    pub fn tools(&self) -> &[ToolSchema] {
        &self.tools
    }

    /// Get validated discovery records for adapters that must retain the exact
    /// server-advertised name for protocol invocation.
    pub fn discovered_tools(&self) -> &[ValidatedMcpTool] {
        &self.discovered_tools
    }

    /// Receive the next server notification independently of request results.
    pub async fn next_notification(&self) -> Option<JsonRpcNotification> {
        self.notifications.lock().await.recv().await
    }

    /// Invoke a remote tool via `tools/call`.
    pub async fn call_tool(&self, tool_name: &str, input: Value) -> Result<String> {
        let result = self
            .request_with_timeout(
                "tools/call",
                json!({"name": tool_name, "arguments": input}),
                self.config.request_timeout,
            )
            .await?;

        let content = result
            .get("content")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("");
        Ok(content.to_string())
    }

    async fn request_with_timeout(
        &self,
        method: &str,
        params: Value,
        deadline: Duration,
    ) -> Result<Value> {
        let id = RpcId::Integer(self.next_id.fetch_add(1, Ordering::SeqCst));
        self.request_with_id(method, params, id, deadline).await
    }

    async fn request_with_id(
        &self,
        method: &str,
        params: Value,
        id: RpcId,
        deadline: Duration,
    ) -> Result<Value> {
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let line = serde_json::to_vec(&message)
            .map_err(|error| mcp_error(format!("serialize request: {error}")))?;
        if line.len() > self.config.max_json_line_bytes {
            return Err(mcp_error(format!(
                "request line exceeds {} byte limit",
                self.config.max_json_line_bytes
            )));
        }

        let (response_tx, response_rx) = oneshot::channel();
        let pending = PendingRegistration::register(self.router.clone(), id.clone(), response_tx)
            .map_err(mcp_error)?;
        let (written_tx, written_rx) = oneshot::channel();
        let writer = self
            .writer
            .as_ref()
            .ok_or_else(|| mcp_error("MCP writer is closed"))?;

        let exchange = async {
            writer
                .send(WriteRequest {
                    line,
                    completion: written_tx,
                })
                .await
                .map_err(|_| "MCP writer task stopped".to_string())?;
            written_rx
                .await
                .map_err(|_| "MCP writer dropped write acknowledgement".to_string())??;
            response_rx
                .await
                .map_err(|_| "MCP response router dropped request".to_string())?
        };

        let response = match tokio::time::timeout(deadline, exchange).await {
            Ok(Ok(response)) => response,
            Ok(Err(reason)) => return Err(mcp_error(reason)),
            Err(_) => {
                return Err(mcp_error(format!(
                    "request '{method}' timed out after {deadline:?}"
                )))
            }
        };
        drop(pending);

        if let Some(error) = response.error {
            return Err(mcp_error(format!(
                "rpc error {}: {}{}",
                error.code,
                error.message,
                error
                    .data
                    .map(|data| format!(" ({data})"))
                    .unwrap_or_default()
            )));
        }
        Ok(response.result.unwrap_or(Value::Null))
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.writer.take();
        self.router.fail_all("MCP client dropped".into());
        if let Some(task) = self.writer_task.take() {
            task.abort();
        }
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
        // SupervisedStdioChild owns the process tree and starts a reaper from
        // Drop, including when this client is dropped outside an async context.
        self.process
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
    }
}

struct WriteRequest {
    line: Vec<u8>,
    completion: oneshot::Sender<std::result::Result<(), String>>,
}

type PendingResult = std::result::Result<JsonRpcResponse, String>;

#[derive(Default)]
struct RouterState {
    inner: StdMutex<RouterStateInner>,
}

#[derive(Default)]
struct RouterStateInner {
    pending: HashMap<RpcId, oneshot::Sender<PendingResult>>,
    terminal: Option<String>,
}

impl RouterState {
    fn lock(&self) -> MutexGuard<'_, RouterStateInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn register(
        &self,
        id: RpcId,
        sender: oneshot::Sender<PendingResult>,
    ) -> std::result::Result<(), String> {
        let mut inner = self.lock();
        if let Some(reason) = &inner.terminal {
            return Err(reason.clone());
        }
        match inner.pending.entry(id.clone()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                Err(format!("duplicate pending MCP request ID {id}"))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(sender);
                Ok(())
            }
        }
    }

    fn remove(&self, id: &RpcId) {
        self.lock().pending.remove(id);
    }

    fn route(&self, response: JsonRpcResponse) -> bool {
        let sender = self.lock().pending.remove(&response.id);
        sender.is_some_and(|sender| sender.send(Ok(response)).is_ok())
    }

    fn fail_all(&self, reason: String) {
        let senders = {
            let mut inner = self.lock();
            if inner.terminal.is_none() {
                inner.terminal = Some(reason.clone());
            }
            inner
                .pending
                .drain()
                .map(|(_, sender)| sender)
                .collect::<Vec<_>>()
        };
        for sender in senders {
            let _ = sender.send(Err(reason.clone()));
        }
    }

    #[cfg(test)]
    fn pending_count(&self) -> usize {
        self.lock().pending.len()
    }
}

struct PendingRegistration {
    router: Arc<RouterState>,
    id: RpcId,
}

impl PendingRegistration {
    fn register(
        router: Arc<RouterState>,
        id: RpcId,
        sender: oneshot::Sender<PendingResult>,
    ) -> std::result::Result<Self, String> {
        router.register(id.clone(), sender)?;
        Ok(Self { router, id })
    }
}

impl Drop for PendingRegistration {
    fn drop(&mut self) {
        self.router.remove(&self.id);
    }
}

async fn writer_loop<W>(
    mut writer: W,
    mut requests: mpsc::Receiver<WriteRequest>,
    router: Arc<RouterState>,
    cancel: CancellationToken,
) where
    W: AsyncWrite + Unpin,
{
    loop {
        let request = tokio::select! {
            _ = cancel.cancelled() => return,
            request = requests.recv() => match request {
                Some(request) => request,
                None => return,
            },
        };
        let result = async {
            writer.write_all(&request.line).await?;
            writer.write_all(b"\n").await?;
            writer.flush().await
        }
        .await;
        match result {
            Ok(()) => {
                let _ = request.completion.send(Ok(()));
            }
            Err(error) => {
                let reason = format!("MCP writer failed: {error}");
                let _ = request.completion.send(Err(reason.clone()));
                router.fail_all(reason);
                cancel.cancel();
                return;
            }
        }
    }
}

async fn reader_loop<R>(
    mut reader: R,
    max_line_bytes: usize,
    router: Arc<RouterState>,
    notifications: mpsc::Sender<JsonRpcNotification>,
    cancel: CancellationToken,
) where
    R: AsyncBufRead + Unpin,
{
    let terminal_reason = loop {
        let line = tokio::select! {
            _ = cancel.cancelled() => return,
            line = read_bounded_json_line(&mut reader, max_line_bytes) => line,
        };
        let line = match line {
            Ok(Some(line)) => line,
            Ok(None) => break "MCP server closed stdout".to_string(),
            Err(error) => break format!("MCP reader failed: {error}"),
        };
        match parse_incoming(&line) {
            Ok(IncomingMessage::Response(response)) => {
                let id = response.id.clone();
                if !router.route(response) {
                    eprintln!("MCP ignored unknown or duplicate response ID {id}");
                }
            }
            Ok(IncomingMessage::Notification(notification)) => {
                if let Err(error) = notifications.try_send(notification) {
                    eprintln!("MCP notification was not queued: {error}");
                }
            }
            Err(reason) => break format!("malformed MCP protocol: {reason}"),
        }
    };
    router.fail_all(terminal_reason);
    cancel.cancel();
}

async fn read_bounded_json_line<R>(
    reader: &mut R,
    max_line_bytes: usize,
) -> io::Result<Option<Vec<u8>>>
where
    R: AsyncBufRead + Unpin,
{
    let mut line = Vec::with_capacity(max_line_bytes.min(8192));
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            if line.is_empty() {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unterminated MCP JSON line",
            ));
        }
        if let Some(newline) = available.iter().position(|byte| *byte == b'\n') {
            if line.len().saturating_add(newline) > max_line_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("MCP JSON line exceeds {max_line_bytes} byte limit"),
                ));
            }
            line.extend_from_slice(&available[..newline]);
            reader.consume(newline + 1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some(line));
        }
        if line.len().saturating_add(available.len()) > max_line_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("MCP JSON line exceeds {max_line_bytes} byte limit"),
            ));
        }
        let consumed = available.len();
        line.extend_from_slice(available);
        reader.consume(consumed);
    }
}

enum IncomingMessage {
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

fn parse_incoming(line: &[u8]) -> std::result::Result<IncomingMessage, String> {
    let value: Value = serde_json::from_slice(line).map_err(|error| error.to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "JSON-RPC message is not an object".to_string())?;
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err("JSON-RPC version must be exactly '2.0'".into());
    }

    if let Some(raw_id) = object.get("id") {
        if object.contains_key("method") {
            return Err("server requests are not supported by this MCP client".into());
        }
        let has_result = object.contains_key("result");
        let has_error = object.contains_key("error");
        if has_result == has_error {
            return Err("response must contain exactly one of result or error".into());
        }
        let id: RpcId = serde_json::from_value(raw_id.clone())
            .map_err(|_| "invalid response ID".to_string())?;
        let error = object
            .get("error")
            .map(|value| {
                serde_json::from_value::<JsonRpcError>(value.clone())
                    .map_err(|error| format!("invalid response error: {error}"))
            })
            .transpose()?;
        return Ok(IncomingMessage::Response(JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: object.get("result").cloned(),
            error,
        }));
    }

    if object.contains_key("result") || object.contains_key("error") {
        return Err("response is missing an ID".into());
    }
    let notification: JsonRpcNotification =
        serde_json::from_value(value).map_err(|error| format!("invalid notification: {error}"))?;
    if notification.method.is_empty() {
        return Err("notification method is empty".into());
    }
    Ok(IncomingMessage::Notification(notification))
}

fn mcp_error(reason: impl Into<String>) -> AgentError {
    AgentError::Tool {
        name: "mcp_client".into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{duplex, AsyncWriteExt};

    fn response(id: RpcId, text: &str) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(json!({"content":[{"type":"text", "text":text}]})),
            error: None,
        }
    }

    #[tokio::test]
    async fn exact_integer_and_string_ids_route_independently_in_any_order() {
        // **Validates: Requirements 2.4**
        let router = Arc::new(RouterState::default());
        let integer = RpcId::Integer(7);
        let string = RpcId::String("7".into());
        let (integer_tx, integer_rx) = oneshot::channel();
        let (string_tx, string_rx) = oneshot::channel();
        let integer_guard =
            PendingRegistration::register(router.clone(), integer.clone(), integer_tx).unwrap();
        let string_guard =
            PendingRegistration::register(router.clone(), string.clone(), string_tx).unwrap();

        assert!(router.route(response(string, "string")));
        assert!(router.route(response(integer, "integer")));
        assert_eq!(
            string_rx.await.unwrap().unwrap().result.unwrap()["content"][0]["text"],
            "string"
        );
        assert_eq!(
            integer_rx.await.unwrap().unwrap().result.unwrap()["content"][0]["text"],
            "integer"
        );
        drop((integer_guard, string_guard));
        assert_eq!(router.pending_count(), 0);
    }

    #[tokio::test]
    async fn notification_queue_is_bounded_without_blocking_response_routing() {
        // **Validates: Requirements 2.4**
        let router = Arc::new(RouterState::default());
        let (response_tx, response_rx) = oneshot::channel();
        let guard =
            PendingRegistration::register(router.clone(), RpcId::Integer(1), response_tx).unwrap();
        let (notification_tx, mut notification_rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let (mut peer, client) = duplex(2048);
        let task = tokio::spawn(reader_loop(
            BufReader::new(client),
            1024,
            router,
            notification_tx,
            cancel.clone(),
        ));
        peer.write_all(
            b"{\"jsonrpc\":\"2.0\",\"method\":\"one\"}\n{\"jsonrpc\":\"2.0\",\"method\":\"two\"}\n{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":null}\n",
        )
        .await
        .unwrap();

        assert_eq!(notification_rx.recv().await.unwrap().method, "one");
        assert!(response_rx.await.unwrap().is_ok());
        drop(guard);
        cancel.cancel();
        task.await.unwrap();
    }

    #[tokio::test]
    async fn eof_fans_out_one_terminal_error_and_clears_pending() {
        // **Validates: Requirements 2.4**
        let router = Arc::new(RouterState::default());
        let (first_tx, first_rx) = oneshot::channel();
        let (second_tx, second_rx) = oneshot::channel();
        let first =
            PendingRegistration::register(router.clone(), RpcId::Integer(1), first_tx).unwrap();
        let second =
            PendingRegistration::register(router.clone(), RpcId::String("two".into()), second_tx)
                .unwrap();
        let (notification_tx, _notification_rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let (peer, client) = duplex(64);
        drop(peer);

        reader_loop(
            BufReader::new(client),
            32,
            router.clone(),
            notification_tx,
            cancel,
        )
        .await;
        assert!(first_rx
            .await
            .unwrap()
            .unwrap_err()
            .contains("closed stdout"));
        assert!(second_rx
            .await
            .unwrap()
            .unwrap_err()
            .contains("closed stdout"));
        drop((first, second));
        assert_eq!(router.pending_count(), 0);
    }

    #[tokio::test]
    async fn malformed_and_oversized_lines_fail_before_unbounded_growth() {
        // **Validates: Requirements 2.4**
        let (mut peer, client) = duplex(64);
        let reader = tokio::spawn(async move {
            let mut reader = BufReader::new(client);
            read_bounded_json_line(&mut reader, 8).await
        });
        peer.write_all(b"123456789").await.unwrap();
        assert_eq!(
            reader.await.unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        assert!(parse_incoming(b"not-json").is_err());
        assert!(parse_incoming(b"{\"jsonrpc\":\"2.0\",\"id\":1}").is_err());
    }

    #[test]
    fn dropped_or_timed_out_registration_removes_pending_entry() {
        // **Validates: Requirements 2.4**
        let router = Arc::new(RouterState::default());
        let (sender, _receiver) = oneshot::channel();
        let guard =
            PendingRegistration::register(router.clone(), RpcId::Integer(9), sender).unwrap();
        assert_eq!(router.pending_count(), 1);
        drop(guard);
        assert_eq!(router.pending_count(), 0);
    }

    #[tokio::test]
    async fn unknown_and_duplicate_responses_are_ignored() {
        // **Validates: Requirements 2.4**
        let router = Arc::new(RouterState::default());
        assert!(!router.route(response(RpcId::Integer(404), "unknown")));
        let (sender, receiver) = oneshot::channel();
        let guard =
            PendingRegistration::register(router.clone(), RpcId::Integer(3), sender).unwrap();
        assert!(router.route(response(RpcId::Integer(3), "first")));
        assert!(!router.route(response(RpcId::Integer(3), "duplicate")));
        assert!(receiver.await.unwrap().is_ok());
        drop(guard);
        assert_eq!(router.pending_count(), 0);
    }
}
