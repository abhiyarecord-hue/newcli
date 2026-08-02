//! Identity-bound trust gate for repository-controlled MCP startup.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use agent_types::{ApprovalDecision, ApprovalKind, ApprovalProvider, ApprovalRequest, ToolEffects};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

const TRUST_SCHEMA_VERSION: u32 = 1;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_TRUST_STORE_BYTES: u64 = 1024 * 1024;

#[derive(Debug)]
pub(crate) struct McpTrustError(String);

impl McpTrustError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for McpTrustError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for McpTrustError {}

type Result<T> = std::result::Result<T, McpTrustError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceMcpConfig {
    servers: Vec<McpServerConfig>,
}

#[derive(Clone, Debug)]
struct CanonicalMcpConfig {
    workspace: PathBuf,
    workspace_hash: String,
    config_hash: String,
    servers: Vec<McpServerConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StoredDecision {
    Approved,
    Denied,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustDecisionRecord {
    schema_version: u32,
    canonical_workspace_hash: String,
    config_hash: String,
    decision: StoredDecision,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustStore {
    schema_version: u32,
    decisions: Vec<TrustDecisionRecord>,
}

impl Default for TrustStore {
    fn default() -> Self {
        Self {
            schema_version: TRUST_SCHEMA_VERSION,
            decisions: Vec::new(),
        }
    }
}

pub(crate) struct SpawnAttempt<T> {
    pub server_name: String,
    pub result: std::result::Result<T, String>,
}

#[async_trait::async_trait]
pub(crate) trait McpSpawner: Sync {
    type Client;

    async fn spawn(
        &self,
        canonical_workspace: &Path,
        server: &McpServerConfig,
    ) -> std::result::Result<Self::Client, String>;
}

pub(crate) struct ProductionMcpSpawner;

#[async_trait::async_trait]
impl McpSpawner for ProductionMcpSpawner {
    type Client = mcp::McpClient;

    async fn spawn(
        &self,
        canonical_workspace: &Path,
        server: &McpServerConfig,
    ) -> std::result::Result<Self::Client, String> {
        let mut command = Command::new(&server.command);
        command
            .args(&server.args)
            .envs(&server.env)
            .current_dir(canonical_workspace);
        mcp::McpClient::connect_command(command, &server.name)
            .await
            // The lower transport error can include the executable label. Do
            // not surface it because repository configuration may place a
            // credential-bearing URL or token there; the redacted approval
            // display is the only command rendering owned by this boundary.
            .map_err(|_| "startup or initialization failed".to_string())
    }
}

/// Authorize the exact effective workspace configuration, then and only then
/// invoke the supplied spawner for each configured server.
pub(crate) async fn start_workspace_servers<S: McpSpawner>(
    workspace: &Path,
    approval: &dyn ApprovalProvider,
    approval_is_interactive: bool,
    spawner: &S,
) -> Result<Vec<SpawnAttempt<S::Client>>> {
    let Some(config) = canonicalize_configuration(workspace)? else {
        return Ok(Vec::new());
    };
    if config.servers.is_empty() {
        return Ok(Vec::new());
    }
    let store_path = default_trust_store_path()?;
    authorize_and_spawn(
        config,
        &store_path,
        approval,
        approval_is_interactive,
        spawner,
    )
    .await
}

#[cfg(test)]
async fn start_workspace_servers_at<S: McpSpawner>(
    workspace: &Path,
    store_path: &Path,
    approval: &dyn ApprovalProvider,
    approval_is_interactive: bool,
    spawner: &S,
) -> Result<Vec<SpawnAttempt<S::Client>>> {
    let Some(config) = canonicalize_configuration(workspace)? else {
        return Ok(Vec::new());
    };
    if config.servers.is_empty() {
        return Ok(Vec::new());
    }
    authorize_and_spawn(
        config,
        store_path,
        approval,
        approval_is_interactive,
        spawner,
    )
    .await
}

async fn authorize_and_spawn<S: McpSpawner>(
    config: CanonicalMcpConfig,
    store_path: &Path,
    approval: &dyn ApprovalProvider,
    approval_is_interactive: bool,
    spawner: &S,
) -> Result<Vec<SpawnAttempt<S::Client>>> {
    let mut store = load_trust_store(store_path, &config.workspace)?;
    let reusable = store
        .decisions
        .iter()
        .find(|record| {
            record.canonical_workspace_hash == config.workspace_hash
                && record.config_hash == config.config_hash
        })
        .map(|record| record.decision);

    let decision = match reusable {
        Some(decision) => decision,
        None => {
            let request = approval_request(&config);
            let decision = approval.request_approval(request).await.map_err(|error| {
                McpTrustError::new(format!("MCP trust approval failed: {error}"))
            })?;
            let stored = match decision {
                ApprovalDecision::Approved => Some(StoredDecision::Approved),
                ApprovalDecision::Denied { .. } if approval_is_interactive => {
                    Some(StoredDecision::Denied)
                }
                ApprovalDecision::Denied { .. } => None,
            };

            if let Some(stored) = stored {
                // A new effective configuration invalidates every older decision
                // for this canonical workspace, including a later rollback to it.
                store
                    .decisions
                    .retain(|record| record.canonical_workspace_hash != config.workspace_hash);
                store.decisions.push(TrustDecisionRecord {
                    schema_version: TRUST_SCHEMA_VERSION,
                    canonical_workspace_hash: config.workspace_hash.clone(),
                    config_hash: config.config_hash.clone(),
                    decision: stored,
                });
                persist_trust_store(store_path, &config.workspace, &store).await?;
            }
            match decision {
                ApprovalDecision::Approved => StoredDecision::Approved,
                ApprovalDecision::Denied { .. } => StoredDecision::Denied,
            }
        }
    };

    if decision == StoredDecision::Denied {
        return Ok(Vec::new());
    }

    let mut attempts = Vec::with_capacity(config.servers.len());
    for server in &config.servers {
        attempts.push(SpawnAttempt {
            server_name: server.name.clone(),
            result: spawner.spawn(&config.workspace, server).await,
        });
    }
    Ok(attempts)
}

fn canonicalize_configuration(workspace: &Path) -> Result<Option<CanonicalMcpConfig>> {
    let canonical_workspace = fs::canonicalize(workspace).map_err(|error| {
        McpTrustError::new(format!(
            "cannot canonicalize MCP workspace '{}': {error}",
            workspace.display()
        ))
    })?;
    if !canonical_workspace.is_dir() {
        return Err(McpTrustError::new(format!(
            "MCP workspace '{}' is not a directory",
            canonical_workspace.display()
        )));
    }

    let configured_path = canonical_workspace.join(".agent").join("mcp.json");
    match fs::symlink_metadata(&configured_path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(McpTrustError::new(format!(
                "cannot inspect MCP configuration '{}': {error}",
                configured_path.display()
            )))
        }
    }

    let canonical_config_path = fs::canonicalize(&configured_path).map_err(|error| {
        McpTrustError::new(format!(
            "cannot canonicalize MCP configuration '{}': {error}",
            configured_path.display()
        ))
    })?;
    if !canonical_config_path.starts_with(&canonical_workspace) {
        return Err(McpTrustError::new(
            "MCP configuration resolves outside the canonical workspace",
        ));
    }

    let bytes = read_bounded_file(
        &canonical_config_path,
        MAX_CONFIG_BYTES,
        "MCP configuration",
    )?;
    let parsed: WorkspaceMcpConfig = serde_json::from_slice(&bytes)
        .map_err(|error| McpTrustError::new(format!("malformed MCP configuration: {error}")))?;
    validate_configuration(&parsed)?;

    let canonical_bytes = serde_json::to_vec(&parsed)
        .map_err(|error| McpTrustError::new(format!("canonicalize MCP configuration: {error}")))?;
    Ok(Some(CanonicalMcpConfig {
        workspace_hash: hash_bytes(&path_identity_bytes(&canonical_workspace)),
        config_hash: hash_bytes(&canonical_bytes),
        workspace: canonical_workspace,
        servers: parsed.servers,
    }))
}

fn validate_configuration(config: &WorkspaceMcpConfig) -> Result<()> {
    let mut names = HashSet::new();
    for server in &config.servers {
        if server.name.trim().is_empty()
            || server.name.chars().any(char::is_control)
            || !names.insert(server.name.as_str())
        {
            return Err(McpTrustError::new(
                "MCP server names must be non-empty, unique, and contain no control characters",
            ));
        }
        if server.command.trim().is_empty() || server.command.contains('\0') {
            return Err(McpTrustError::new(format!(
                "MCP server '{}' has an invalid command",
                server.name
            )));
        }
        if server.args.iter().any(|argument| argument.contains('\0')) {
            return Err(McpTrustError::new(format!(
                "MCP server '{}' has an argument containing NUL",
                server.name
            )));
        }
        if server.env.iter().any(|(key, value)| {
            key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0')
        }) {
            return Err(McpTrustError::new(format!(
                "MCP server '{}' has an invalid environment entry",
                server.name
            )));
        }
    }
    Ok(())
}

fn approval_request(config: &CanonicalMcpConfig) -> ApprovalRequest {
    let displayed_servers: Vec<serde_json::Value> = config
        .servers
        .iter()
        .map(|server| {
            serde_json::json!({
                "name": server.name,
                "command": redact_urlish(&server.command),
                "args": redact_arguments(&server.args),
                "environment": server.env.keys().map(|key| format!("{key}=<redacted>")).collect::<Vec<_>>(),
            })
        })
        .collect();

    let mut prompt = format!(
        "Workspace '{}' requests permission to start MCP servers. The exact approved configuration will be remembered for this canonical workspace:\n",
        config.workspace.display()
    );
    for server in &displayed_servers {
        prompt.push_str(&format!(
            "  - {}: command={} args={} environment={}\n",
            server["name"], server["command"], server["args"], server["environment"]
        ));
    }

    let has_environment = config.servers.iter().any(|server| !server.env.is_empty());
    let mut effects = ToolEffects::PROCESS_SPAWN.union(ToolEffects::NETWORK_OUTBOUND);
    if has_environment {
        effects = effects.union(ToolEffects::CREDENTIAL_BEARING);
    }

    ApprovalRequest {
        id: format!(
            "workspace-mcp:{}:{}",
            &config.workspace_hash[..16],
            &config.config_hash[..16]
        ),
        kind: ApprovalKind::WorkspaceMcpStartup,
        prompt,
        tool_name: None,
        requested_input: serde_json::json!({
            "canonical_workspace_hash": config.workspace_hash,
            "config_hash": config.config_hash,
            "servers": displayed_servers,
        }),
        effects,
    }
}

fn redact_arguments(arguments: &[String]) -> Vec<String> {
    let mut redact_next = false;
    arguments
        .iter()
        .map(|argument| {
            if redact_next {
                redact_next = false;
                return "<redacted>".to_string();
            }
            if let Some((key, _)) = argument.split_once('=') {
                if is_sensitive_key(key) {
                    return format!("{key}=<redacted>");
                }
            }
            if is_sensitive_key(argument) {
                redact_next = true;
                return argument.clone();
            }
            redact_urlish(argument)
        })
        .collect()
}

fn is_sensitive_key(value: &str) -> bool {
    let normalized = value
        .trim_start_matches('-')
        .replace(['-', '_'], "")
        .to_ascii_lowercase();
    [
        "token",
        "password",
        "passwd",
        "secret",
        "apikey",
        "authorization",
        "credential",
        "privatekey",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn redact_urlish(value: &str) -> String {
    let mut redacted = value.to_string();
    if let Some(scheme_end) = redacted.find("://") {
        let authority_start = scheme_end + 3;
        if let Some(relative_at) = redacted[authority_start..].find('@') {
            let at = authority_start + relative_at;
            redacted.replace_range(authority_start..at, "<redacted>");
        }
        if let Some(query) = redacted.find('?') {
            redacted.truncate(query);
            redacted.push_str("?<redacted>");
        }
    }
    if redacted.to_ascii_lowercase().starts_with("bearer ") {
        return "Bearer <redacted>".to_string();
    }
    redacted
}

fn load_trust_store(path: &Path, canonical_workspace: &Path) -> Result<TrustStore> {
    validate_store_location(path, canonical_workspace)?;
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(TrustStore::default())
        }
        Err(error) => {
            return Err(McpTrustError::new(format!(
                "cannot inspect MCP trust store: {error}"
            )))
        }
    }

    let canonical_store = fs::canonicalize(path).map_err(|error| {
        McpTrustError::new(format!("cannot canonicalize MCP trust store: {error}"))
    })?;
    if canonical_store.starts_with(canonical_workspace) {
        return Err(McpTrustError::new(
            "MCP trust store must remain outside the workspace",
        ));
    }
    let bytes = read_bounded_file(&canonical_store, MAX_TRUST_STORE_BYTES, "MCP trust store")?;
    let store: TrustStore = serde_json::from_slice(&bytes)
        .map_err(|error| McpTrustError::new(format!("malformed MCP trust store: {error}")))?;
    if store.schema_version != TRUST_SCHEMA_VERSION
        || store
            .decisions
            .iter()
            .any(|record| record.schema_version != TRUST_SCHEMA_VERSION)
    {
        return Err(McpTrustError::new("unsupported MCP trust store schema"));
    }
    Ok(store)
}

async fn persist_trust_store(
    path: &Path,
    canonical_workspace: &Path,
    store: &TrustStore,
) -> Result<()> {
    validate_store_location(path, canonical_workspace)?;
    let parent = path
        .parent()
        .ok_or_else(|| McpTrustError::new("MCP trust store has no parent directory"))?;
    fs::create_dir_all(parent)
        .map_err(|error| McpTrustError::new(format!("create MCP trust directory: {error}")))?;
    let canonical_parent = fs::canonicalize(parent).map_err(|error| {
        McpTrustError::new(format!("canonicalize MCP trust directory: {error}"))
    })?;
    if canonical_parent.starts_with(canonical_workspace) {
        return Err(McpTrustError::new(
            "MCP trust store must remain outside the workspace",
        ));
    }

    let bytes = serde_json::to_vec_pretty(store)
        .map_err(|error| McpTrustError::new(format!("serialize MCP trust store: {error}")))?;
    runtime_core::atomic_replace(
        path,
        &bytes,
        runtime_core::AtomicWriteOptions::default(),
        &CancellationToken::new(),
    )
    .await
    .map_err(|error| McpTrustError::new(format!("persist MCP trust decision: {error}")))
}

fn validate_store_location(path: &Path, canonical_workspace: &Path) -> Result<()> {
    if !path.is_absolute() {
        return Err(McpTrustError::new(
            "MCP trust store path must be an absolute user-local path",
        ));
    }
    if path.starts_with(canonical_workspace) {
        return Err(McpTrustError::new(
            "MCP trust store must remain outside the workspace",
        ));
    }
    if let Some(parent) = path.parent() {
        if parent.exists() {
            let canonical_parent = fs::canonicalize(parent).map_err(|error| {
                McpTrustError::new(format!("canonicalize MCP trust directory: {error}"))
            })?;
            if canonical_parent.starts_with(canonical_workspace) {
                return Err(McpTrustError::new(
                    "MCP trust store must remain outside the workspace",
                ));
            }
        }
    }
    Ok(())
}

fn default_trust_store_path() -> Result<PathBuf> {
    let base = platform_config_home().ok_or_else(|| {
        McpTrustError::new("cannot determine the OS user configuration directory for MCP trust")
    })?;
    if !base.is_absolute() {
        return Err(McpTrustError::new(
            "OS user configuration directory for MCP trust is not absolute",
        ));
    }
    Ok(base.join("srijandev").join("mcp-workspace-trust-v1.json"))
}

#[cfg(target_os = "windows")]
fn platform_config_home() -> Option<PathBuf> {
    std::env::var_os("APPDATA")
        .or_else(|| std::env::var_os("LOCALAPPDATA"))
        .map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn platform_config_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library").join("Application Support"))
}

#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
fn platform_config_home() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        })
}

fn read_bounded_file(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let file = fs::File::open(path)
        .map_err(|error| McpTrustError::new(format!("read {label}: {error}")))?;
    let metadata = file
        .metadata()
        .map_err(|error| McpTrustError::new(format!("inspect {label}: {error}")))?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err(McpTrustError::new(format!(
            "{label} must be a regular file no larger than {maximum} bytes"
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| McpTrustError::new(format!("read {label}: {error}")))?;
    if bytes.len() as u64 > maximum {
        return Err(McpTrustError::new(format!(
            "{label} exceeds the {maximum}-byte limit"
        )));
    }
    Ok(bytes)
}

fn path_identity_bytes(path: &Path) -> Vec<u8> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str()
            .encode_wide()
            .flat_map(u16::to_le_bytes)
            .collect()
    }
    #[cfg(not(any(unix, windows)))]
    {
        path.to_string_lossy().as_bytes().to_vec()
    }
}

fn hash_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use agent_types::{AgentError, Result as AgentResult};
    use proptest::prelude::*;
    use serde_json::json;

    use super::*;

    struct RecordingApproval {
        decision: ApprovalDecision,
        calls: AtomicUsize,
        requests: Mutex<Vec<ApprovalRequest>>,
        spawn_calls: Option<Arc<AtomicUsize>>,
    }

    impl RecordingApproval {
        fn new(decision: ApprovalDecision) -> Self {
            Self {
                decision,
                calls: AtomicUsize::new(0),
                requests: Mutex::new(Vec::new()),
                spawn_calls: None,
            }
        }

        fn asserting_no_spawn(decision: ApprovalDecision, spawn_calls: Arc<AtomicUsize>) -> Self {
            Self {
                decision,
                calls: AtomicUsize::new(0),
                requests: Mutex::new(Vec::new()),
                spawn_calls: Some(spawn_calls),
            }
        }
    }

    #[async_trait::async_trait]
    impl ApprovalProvider for RecordingApproval {
        async fn request_approval(
            &self,
            request: ApprovalRequest,
        ) -> AgentResult<ApprovalDecision> {
            if let Some(calls) = &self.spawn_calls {
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    0,
                    "MCP command spawned before approval completed"
                );
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.requests.lock().unwrap().push(request);
            Ok(self.decision.clone())
        }
    }

    struct FailingApproval;

    #[async_trait::async_trait]
    impl ApprovalProvider for FailingApproval {
        async fn request_approval(
            &self,
            _request: ApprovalRequest,
        ) -> AgentResult<ApprovalDecision> {
            Err(AgentError::Tool {
                name: "mcp-trust".into(),
                reason: "approval unavailable".into(),
            })
        }
    }

    #[derive(Clone)]
    struct FakeSpawner {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl McpSpawner for FakeSpawner {
        type Client = String;

        async fn spawn(
            &self,
            canonical_workspace: &Path,
            server: &McpServerConfig,
        ) -> std::result::Result<Self::Client, String> {
            assert!(canonical_workspace.is_absolute());
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(format!("{}:{}", server.name, server.command))
        }
    }

    fn fixture() -> (tempfile::TempDir, tempfile::TempDir, PathBuf) {
        let workspace = tempfile::tempdir().unwrap();
        let user_local = tempfile::tempdir().unwrap();
        fs::create_dir(workspace.path().join(".agent")).unwrap();
        let store = user_local.path().join("mcp-trust.json");
        (workspace, user_local, store)
    }

    fn write_config(workspace: &Path, argument: &str, secret: &str) {
        let config = json!({
            "servers": [{
                "name": "fixture",
                "command": "fake-mcp",
                "args": ["--mode", argument, "--api-token", secret],
                "env": {"MCP_API_TOKEN": secret}
            }]
        });
        fs::write(
            workspace.join(".agent").join("mcp.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn approval_is_durable_before_spawn_and_exact_identity_is_reused() {
        // **Validates: Requirements 2.26, 3.11**
        let (workspace, _user_local, store) = fixture();
        let secret = "never-persist-this-credential";
        write_config(workspace.path(), "first", secret);
        let spawn_calls = Arc::new(AtomicUsize::new(0));
        let spawner = FakeSpawner {
            calls: spawn_calls.clone(),
        };
        let approval =
            RecordingApproval::asserting_no_spawn(ApprovalDecision::Approved, spawn_calls.clone());

        let attempts =
            start_workspace_servers_at(workspace.path(), &store, &approval, true, &spawner)
                .await
                .unwrap();
        assert_eq!(attempts.len(), 1);
        assert!(attempts[0].result.is_ok());
        assert_eq!(approval.calls.load(Ordering::SeqCst), 1);
        assert_eq!(spawn_calls.load(Ordering::SeqCst), 1);
        assert!(store.exists(), "approval must be durable before spawn");

        let request_text = serde_json::to_string(&approval.requests.lock().unwrap()[0]).unwrap();
        let store_text = fs::read_to_string(&store).unwrap();
        assert!(!request_text.contains(secret));
        assert!(!store_text.contains(secret));
        assert!(!store_text.contains("fake-mcp"));
        assert!(request_text.contains("<redacted>"));

        let should_not_prompt = RecordingApproval::new(ApprovalDecision::Denied {
            reason: "must not be consulted for an exact reusable approval".into(),
        });
        let attempts = start_workspace_servers_at(
            workspace.path(),
            &store,
            &should_not_prompt,
            true,
            &spawner,
        )
        .await
        .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(should_not_prompt.calls.load(Ordering::SeqCst), 0);
        assert_eq!(spawn_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn changed_configuration_invalidates_prior_decision_and_prunes_rollback_reuse() {
        // **Validates: Requirements 2.26**
        let (workspace, _user_local, store) = fixture();
        write_config(workspace.path(), "first", "secret-one");
        let calls = Arc::new(AtomicUsize::new(0));
        let spawner = FakeSpawner {
            calls: calls.clone(),
        };
        let first = RecordingApproval::new(ApprovalDecision::Approved);
        start_workspace_servers_at(workspace.path(), &store, &first, true, &spawner)
            .await
            .unwrap();

        write_config(workspace.path(), "changed", "secret-two");
        let changed = RecordingApproval::new(ApprovalDecision::Approved);
        start_workspace_servers_at(workspace.path(), &store, &changed, true, &spawner)
            .await
            .unwrap();
        assert_eq!(changed.calls.load(Ordering::SeqCst), 1);

        write_config(workspace.path(), "first", "secret-one");
        let rollback = RecordingApproval::new(ApprovalDecision::Denied {
            reason: "reverted content requires a fresh decision".into(),
        });
        let attempts =
            start_workspace_servers_at(workspace.path(), &store, &rollback, true, &spawner)
                .await
                .unwrap();
        assert!(attempts.is_empty());
        assert_eq!(rollback.calls.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let persisted: TrustStore = serde_json::from_slice(&fs::read(&store).unwrap()).unwrap();
        assert_eq!(persisted.decisions.len(), 1);
        assert_eq!(persisted.decisions[0].decision, StoredDecision::Denied);
    }

    #[tokio::test]
    async fn malformed_noninteractive_denied_and_failed_approval_cases_never_spawn() {
        // **Validates: Requirements 2.26**
        let (workspace, _user_local, store) = fixture();
        let calls = Arc::new(AtomicUsize::new(0));
        let spawner = FakeSpawner {
            calls: calls.clone(),
        };
        fs::write(
            workspace.path().join(".agent").join("mcp.json"),
            br#"{"servers":[{"name":"fixture","command":7}]}"#,
        )
        .unwrap();
        assert!(start_workspace_servers_at(
            workspace.path(),
            &store,
            &RecordingApproval::new(ApprovalDecision::Approved),
            true,
            &spawner,
        )
        .await
        .is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        write_config(workspace.path(), "valid", "secret");
        let redirected = RecordingApproval::new(ApprovalDecision::Denied {
            reason: "interactive approval unavailable".into(),
        });
        let attempts =
            start_workspace_servers_at(workspace.path(), &store, &redirected, false, &spawner)
                .await
                .unwrap();
        assert!(attempts.is_empty());
        assert!(
            !store.exists(),
            "automatic non-TTY denial is not a user decision"
        );

        assert!(start_workspace_servers_at(
            workspace.path(),
            &store,
            &FailingApproval,
            true,
            &spawner,
        )
        .await
        .is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn trust_store_inside_workspace_is_rejected_before_spawn() {
        // **Validates: Requirements 2.26**
        let (workspace, _user_local, _store) = fixture();
        write_config(workspace.path(), "valid", "secret");
        let calls = Arc::new(AtomicUsize::new(0));
        let spawner = FakeSpawner {
            calls: calls.clone(),
        };
        let result = start_workspace_servers_at(
            workspace.path(),
            &workspace.path().join(".agent").join("trust.json"),
            &RecordingApproval::new(ApprovalDecision::Approved),
            true,
            &spawner,
        )
        .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn property_effective_argument_changes_configuration_identity(
            left in "[A-Za-z0-9._-]{1,32}",
            right in "[A-Za-z0-9._-]{1,32}",
        ) {
            // **Validates: Requirements 2.26**
            prop_assume!(left != right);
            let make = |argument: String| WorkspaceMcpConfig {
                servers: vec![McpServerConfig {
                    name: "fixture".into(),
                    command: "fake-mcp".into(),
                    args: vec![argument],
                    env: BTreeMap::new(),
                }],
            };
            let left_hash = hash_bytes(&serde_json::to_vec(&make(left)).unwrap());
            let right_hash = hash_bytes(&serde_json::to_vec(&make(right)).unwrap());
            prop_assert_ne!(left_hash, right_hash);
        }
    }
}
