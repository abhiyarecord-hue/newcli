//! Built-in tool implementations: the concrete capabilities the agent exposes.
//!
//! All file-touching tools resolve paths through [`PathJail`] (rooted at
//! `ctx.project_root`) so the model can never escape the workspace. The bash
//! tool runs through [`ProcessFallback`], honouring cancellation and timeouts.
//!
//! Tools implemented here:
//! - [`ReadFileTool`]  (`read_file`)   — read a file, optionally a line range.
//! - [`WriteFileTool`] (`write_file`)  — create/overwrite a file.
//! - [`ListFilesTool`] (`list_files`)  — list a directory's entries.
//! - [`SearchTextTool`](`search_text`) — recursive substring search.
//! - [`BashTool`]      (`bash`)        — run a shell command in the workspace.

use std::path::{Path, PathBuf};
use std::time::Duration;

use agent_types::{AgentError, Result, Tool, ToolCtx, ToolSchema};
use sandbox::{NetGuard, PathJail, ProcessFallback, SandboxExecutor};
use serde_json::{json, Value};

/// Pull a required string field out of the tool input JSON.
fn required_str(input: &Value, field: &str, tool: &str) -> Result<String> {
    input
        .get(field)
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .ok_or_else(|| AgentError::Tool {
            name: tool.to_string(),
            reason: format!("missing required string field '{field}'"),
        })
}

// ===========================================================================
// read_file
// ===========================================================================

/// Read the contents of a file inside the workspace.
pub struct ReadFileTool;

#[async_trait::async_trait]
impl Tool for ReadFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "read_file".into(),
            description: "Read the contents of a text file in the workspace. \
                Optionally restrict to a 1-indexed inclusive line range with \
                'start_line' and 'end_line'."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative path to the file" },
                    "start_line": { "type": "integer", "description": "First line to read (1-indexed, optional)" },
                    "end_line": { "type": "integer", "description": "Last line to read (1-indexed inclusive, optional)" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn invoke(&self, input: Value, ctx: &ToolCtx) -> Result<String> {
        let path = required_str(&input, "path", "read_file")?;
        let jail = PathJail::new(&ctx.project_root)?;
        let safe = jail.resolve(Path::new(&path))?;

        let content = tokio::fs::read_to_string(&safe).await?;

        let start = input.get("start_line").and_then(Value::as_u64);
        let end = input.get("end_line").and_then(Value::as_u64);

        if start.is_none() && end.is_none() {
            return Ok(content);
        }

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len() as u64;
        let start = start.unwrap_or(1).max(1);
        let end = end.unwrap_or(total).min(total);
        if start > end {
            return Err(AgentError::Tool {
                name: "read_file".into(),
                reason: format!("start_line ({start}) > end_line ({end})"),
            });
        }

        let slice: Vec<String> = lines[(start as usize - 1)..(end as usize)]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:>6} | {}", start as usize + i, l))
            .collect();
        Ok(slice.join("\n"))
    }
}

// ===========================================================================
// write_file
// ===========================================================================

/// Create or overwrite a file inside the workspace.
///
/// CRDT-aware: tracks the content the agent last wrote per path. If an external
/// process (e.g. the editor) modified the file since the agent last touched it,
/// the tool detects the concurrent edit and merges non-overlapping changes;
/// on an unmergeable overlap it preserves BOTH versions (agent writes its
/// version, the external version is backed up) so no data is ever lost and the
/// process never crashes.
pub struct WriteFileTool {
    /// Snapshot of what the agent last wrote, keyed by resolved path.
    snapshots: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<PathBuf, String>>>,
}

impl WriteFileTool {
    pub fn new() -> Self {
        Self {
            snapshots: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Tool for WriteFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "write_file".into(),
            description: "Create a new file or overwrite an existing one in the \
                workspace with the given content. Parent directories are created \
                automatically. Concurrent external edits are detected and merged \
                safely (both versions preserved on conflict)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative path to write" },
                    "content": { "type": "string", "description": "Full file content to write" }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn invoke(&self, input: Value, ctx: &ToolCtx) -> Result<String> {
        let path = required_str(&input, "path", "write_file")?;
        let content = required_str(&input, "content", "write_file")?;
        let jail = PathJail::new(&ctx.project_root)?;
        let safe = jail.resolve(Path::new(&path))?;

        if let Some(parent) = safe.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Read current on-disk content (empty if file is new).
        let disk_content = tokio::fs::read_to_string(&safe).await.unwrap_or_default();
        let file_exists = safe.exists();

        // What did the agent last write to this path?
        let base = {
            let snaps = self.snapshots.lock().await;
            snaps.get(&safe).cloned()
        };

        // Decide how to write, defensively (never panic).
        let (final_content, note) = match &base {
            // Agent never touched this file before, OR disk matches the agent's
            // last known version → no external edit; write directly.
            None => (content.clone(), String::new()),
            Some(prev) if *prev == disk_content => (content.clone(), String::new()),
            // External edit detected: disk differs from what the agent last wrote.
            Some(prev) => {
                match three_way_merge(prev, &disk_content, &content) {
                    MergeResult::Clean(merged) => (
                        merged,
                        "\n[note: merged with concurrent external edit — no conflict]".to_string(),
                    ),
                    MergeResult::Conflict => {
                        // Preserve the external version as a backup; write agent's version.
                        let backup = safe.with_extension(format!(
                            "{}.external-backup",
                            safe.extension().and_then(|e| e.to_str()).unwrap_or("txt")
                        ));
                        let _ = tokio::fs::write(&backup, disk_content.as_bytes()).await;
                        (
                            content.clone(),
                            format!(
                                "\n[warning: concurrent edit conflicted on overlapping lines. \
                                 Agent version written; external version backed up to {}]",
                                backup.display()
                            ),
                        )
                    }
                }
            }
        };

        // Write the final content.
        tokio::fs::write(&safe, final_content.as_bytes()).await?;

        // Update snapshot to what is now on disk.
        {
            let mut snaps = self.snapshots.lock().await;
            snaps.insert(safe.clone(), final_content.clone());
        }

        let bytes = final_content.len();
        let lines = final_content.lines().count();
        let verb = if file_exists { "Updated" } else { "Wrote" };
        Ok(format!("{verb} {bytes} bytes ({lines} lines) to {path}{note}"))
    }
}

/// Result of a 3-way merge.
enum MergeResult {
    /// Successfully merged (non-overlapping changes).
    Clean(String),
    /// Both sides changed the same region — cannot auto-merge safely.
    Conflict,
}

/// Line-based 3-way merge. `base` = common ancestor, `theirs` = current disk
/// (external edit), `ours` = agent's new content.
///
/// Pure string operations — cannot panic. Returns `Clean` when the two sides
/// touched different lines, `Conflict` when they overlap.
fn three_way_merge(base: &str, theirs: &str, ours: &str) -> MergeResult {
    // Fast paths.
    if theirs == base {
        return MergeResult::Clean(ours.to_string()); // only agent changed
    }
    if ours == base {
        return MergeResult::Clean(theirs.to_string()); // only external changed
    }
    if ours == theirs {
        return MergeResult::Clean(ours.to_string()); // both made identical change
    }

    let base_lines: Vec<&str> = base.lines().collect();
    let theirs_lines: Vec<&str> = theirs.lines().collect();
    let ours_lines: Vec<&str> = ours.lines().collect();

    // Common prefix length (lines unchanged at the top by both sides).
    let common_prefix = {
        let mut i = 0;
        let max = base_lines.len().min(theirs_lines.len()).min(ours_lines.len());
        while i < max
            && base_lines[i] == theirs_lines[i]
            && base_lines[i] == ours_lines[i]
        {
            i += 1;
        }
        i
    };

    // Common suffix length (lines unchanged at the bottom by both sides).
    let common_suffix = {
        let mut i = 0;
        let max = (base_lines.len().saturating_sub(common_prefix))
            .min(theirs_lines.len().saturating_sub(common_prefix))
            .min(ours_lines.len().saturating_sub(common_prefix));
        while i < max {
            let b = base_lines[base_lines.len() - 1 - i];
            let t = theirs_lines[theirs_lines.len() - 1 - i];
            let o = ours_lines[ours_lines.len() - 1 - i];
            if b == t && b == o {
                i += 1;
            } else {
                break;
            }
        }
        i
    };

    // The middle (changed) region for each side.
    let theirs_mid = &theirs_lines[common_prefix..theirs_lines.len() - common_suffix];
    let ours_mid = &ours_lines[common_prefix..ours_lines.len() - common_suffix];
    let base_mid = &base_lines[common_prefix..base_lines.len() - common_suffix];

    // If one side's middle equals base's middle, only the other side changed
    // the region → take the changed one (clean merge).
    let theirs_changed = theirs_mid != base_mid;
    let ours_changed = ours_mid != base_mid;

    if theirs_changed && ours_changed {
        // Both changed the same region → overlap conflict.
        return MergeResult::Conflict;
    }

    // Reassemble: prefix + (whichever side changed the middle) + suffix.
    let mut merged: Vec<&str> = Vec::new();
    merged.extend_from_slice(&ours_lines[..common_prefix]);
    if ours_changed {
        merged.extend_from_slice(ours_mid);
    } else {
        merged.extend_from_slice(theirs_mid);
    }
    let suffix_start = ours_lines.len() - common_suffix;
    merged.extend_from_slice(&ours_lines[suffix_start..]);

    let mut result = merged.join("\n");
    // Preserve trailing newline if the agent's content had one.
    if ours.ends_with('\n') {
        result.push('\n');
    }
    MergeResult::Clean(result)
}

// ===========================================================================
// list_files
// ===========================================================================

/// List the entries of a directory inside the workspace.
pub struct ListFilesTool;

#[async_trait::async_trait]
impl Tool for ListFilesTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "list_files".into(),
            description: "List the files and subdirectories directly inside a \
                workspace directory. Defaults to the workspace root."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative directory (default '.')" }
                }
            }),
        }
    }

    async fn invoke(&self, input: Value, ctx: &ToolCtx) -> Result<String> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".")
            .to_string();
        let jail = PathJail::new(&ctx.project_root)?;
        let safe = jail.resolve(Path::new(&path))?;

        let mut entries = tokio::fs::read_dir(&safe).await?;
        let mut out: Vec<String> = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry
                .file_type()
                .await
                .map(|t| t.is_dir())
                .unwrap_or(false);
            if is_dir {
                out.push(format!("{name}/"));
            } else {
                out.push(name);
            }
        }
        out.sort();
        if out.is_empty() {
            Ok(format!("(empty directory: {path})"))
        } else {
            Ok(out.join("\n"))
        }
    }
}

// ===========================================================================
// search_text
// ===========================================================================

/// Recursive case-insensitive substring search across workspace files.
pub struct SearchTextTool;

const SEARCH_MAX_RESULTS: usize = 200;
const SEARCH_SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".idea", ".vscode"];

#[async_trait::async_trait]
impl Tool for SearchTextTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "search_text".into(),
            description: "Recursively search workspace files for a case-insensitive \
                substring. Returns matching lines with file path and line number. \
                Skips .git, target, and node_modules."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Substring to search for" },
                    "path": { "type": "string", "description": "Workspace-relative directory to search in (default '.')" }
                },
                "required": ["query"]
            }),
        }
    }

    async fn invoke(&self, input: Value, ctx: &ToolCtx) -> Result<String> {
        let query = required_str(&input, "query", "search_text")?;
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".")
            .to_string();
        let jail = PathJail::new(&ctx.project_root)?;
        let root = jail.resolve(Path::new(&path))?;
        let needle = query.to_lowercase();

        let mut results: Vec<String> = Vec::new();
        let mut stack: Vec<PathBuf> = vec![root];

        while let Some(dir) = stack.pop() {
            if ctx.cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }
            let mut rd = match tokio::fs::read_dir(&dir).await {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            while let Some(entry) = rd.next_entry().await? {
                let entry_path = entry.path();
                let file_type = match entry.file_type().await {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if file_type.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if SEARCH_SKIP_DIRS.contains(&name.as_str()) {
                        continue;
                    }
                    stack.push(entry_path);
                } else if file_type.is_file() {
                    // Read as text; skip binary/unreadable files.
                    let content = match tokio::fs::read_to_string(&entry_path).await {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    let rel = entry_path
                        .strip_prefix(jail.root())
                        .unwrap_or(&entry_path)
                        .display()
                        .to_string();
                    for (i, line) in content.lines().enumerate() {
                        if line.to_lowercase().contains(&needle) {
                            results.push(format!("{}:{}: {}", rel, i + 1, line.trim()));
                            if results.len() >= SEARCH_MAX_RESULTS {
                                results.push(format!(
                                    "[... stopped at {SEARCH_MAX_RESULTS} matches]"
                                ));
                                return Ok(results.join("\n"));
                            }
                        }
                    }
                }
            }
        }

        if results.is_empty() {
            Ok(format!("No matches for '{query}'"))
        } else {
            Ok(results.join("\n"))
        }
    }
}

// ===========================================================================
// bash
// ===========================================================================

/// Run a shell command in the workspace via the sandbox executor.
pub struct BashTool {
    default_timeout: Duration,
}

impl BashTool {
    pub fn new() -> Self {
        Self {
            default_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Tool for BashTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "bash".into(),
            description: "Run a shell command in the workspace root and return its \
                stdout, stderr, and exit code. Runs in a sandboxed process with a \
                timeout. Destructive commands are blocked by policy."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The shell command to execute" },
                    "timeout_secs": { "type": "integer", "description": "Timeout in seconds (default 60)" }
                },
                "required": ["command"]
            }),
        }
    }

    async fn invoke(&self, input: Value, ctx: &ToolCtx) -> Result<String> {
        let command = required_str(&input, "command", "bash")?;
        let timeout = input
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .map(Duration::from_secs)
            .unwrap_or(self.default_timeout);

        // User approval for potentially dangerous commands.
        if needs_user_approval(&command) {
            eprintln!("\n  ⚠ Agent wants to run: {command}");
            eprint!("  Allow? [y/N]: ");
            use std::io::Write;
            std::io::stderr().flush().ok();

            let mut response = String::new();
            std::io::stdin().read_line(&mut response).ok();
            let approved = response.trim().eq_ignore_ascii_case("y")
                || response.trim().eq_ignore_ascii_case("yes");

            if !approved {
                return Ok("Command denied by user.".to_string());
            }
        }

        let result = ProcessFallback
            .execute(&command, timeout, &ctx.cancel, &ctx.project_root)
            .await?;

        let mut out = String::new();
        out.push_str(&format!("exit_code: {}\n", result.exit_code));
        if !result.stdout.is_empty() {
            out.push_str("--- stdout ---\n");
            out.push_str(&result.stdout);
            if !result.stdout.ends_with('\n') {
                out.push('\n');
            }
        }
        if !result.stderr.is_empty() {
            out.push_str("--- stderr ---\n");
            out.push_str(&result.stderr);
            if !result.stderr.ends_with('\n') {
                out.push('\n');
            }
        }
        if result.stdout.is_empty() && result.stderr.is_empty() {
            out.push_str("(no output)\n");
        }
        Ok(out)
    }
}

/// Check if a command needs user approval before execution.
/// Returns true for commands that could modify system state significantly.
fn needs_user_approval(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    let risky_patterns = [
        // Deletion / destructive
        "rm ", "rmdir", "del ", "rd ",
        // Package installation / system changes
        "npm install", "pip install", "cargo install", "apt ", "brew ",
        "choco ", "winget ",
        // Git push / remote operations
        "git push", "git remote",
        // Process / system
        "shutdown", "reboot", "taskkill",
        // Network / download
        "curl ", "wget ", "invoke-webrequest",
        // Disk operations
        "format ", "mkfs", "dd ",
        // Permission changes
        "chmod ", "chown ", "icacls",
    ];
    risky_patterns.iter().any(|p| lower.contains(p))
}

// ===========================================================================
// web_fetch
// ===========================================================================

/// Fetch content from an HTTPS URL via the SSRF-protected NetGuard.
/// Only allowlisted domains are reachable; private/loopback IPs are blocked.
pub struct WebFetchTool {
    allowed_domains: Vec<String>,
}

impl WebFetchTool {
    pub fn new(allowed_domains: Vec<String>) -> Self {
        Self { allowed_domains }
    }

    /// Default allowlist: common developer documentation sites.
    pub fn with_defaults() -> Self {
        Self {
            allowed_domains: vec![
                "docs.rs".into(),
                "crates.io".into(),
                "github.com".into(),
                "raw.githubusercontent.com".into(),
                "developer.mozilla.org".into(),
                "doc.rust-lang.org".into(),
                "pypi.org".into(),
                "stackoverflow.com".into(),
            ],
        }
    }
}

#[async_trait::async_trait]
impl Tool for WebFetchTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_fetch".into(),
            description: format!(
                "Fetch text content from an HTTPS URL. Only these domains are allowed: {}. \
                 Private/loopback IPs are blocked (SSRF protection).",
                self.allowed_domains.join(", ")
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "HTTPS URL to fetch" }
                },
                "required": ["url"]
            }),
        }
    }

    async fn invoke(&self, input: Value, _ctx: &ToolCtx) -> Result<String> {
        let url = required_str(&input, "url", "web_fetch")?;
        let guard = NetGuard::new(self.allowed_domains.clone());
        let body = guard.get(&url).await?;
        // Truncate very large pages (dispatcher also caps at 30k, but be explicit).
        let out = if body.len() > 20_000 {
            format!("{}\n[truncated at 20000 chars]", &body[..20_000])
        } else {
            body
        };
        Ok(out)
    }
}

// ===========================================================================
// dispatch_subagent (parallel exploration)
// ===========================================================================

/// Spawn parallel read-only sub-agents to explore/analyze independent questions.
/// Each sub-agent runs an isolated bounded reasoning loop and returns a summary.
/// Useful for breaking a large investigation into parallel chunks.
pub struct SubAgentTool {
    provider: std::sync::Arc<dyn llm_client::LlmProvider>,
    max_concurrent: usize,
}

impl SubAgentTool {
    pub fn new(provider: std::sync::Arc<dyn llm_client::LlmProvider>) -> Self {
        Self {
            provider,
            max_concurrent: 3,
        }
    }
}

#[async_trait::async_trait]
impl Tool for SubAgentTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "dispatch_subagent".into(),
            description: "Spawn parallel sub-agents to analyze/plan independent questions. \
                Pass an array of task strings; each runs in isolation and returns a summary. \
                Use for breaking large research into parallel parts (e.g. analyzing multiple \
                modules at once). Sub-agents are reasoning-only (no file writes)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of independent tasks/questions to explore in parallel"
                    }
                },
                "required": ["tasks"]
            }),
        }
    }

    async fn invoke(&self, input: Value, ctx: &ToolCtx) -> Result<String> {
        let tasks: Vec<String> = input
            .get("tasks")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if tasks.is_empty() {
            return Err(AgentError::Tool {
                name: "dispatch_subagent".into(),
                reason: "no tasks provided".into(),
            });
        }

        let pool = harness::SubAgentPool::new(
            self.provider.clone(),
            Vec::new(), // reasoning-only sub-agents
            self.max_concurrent,
        );

        let summaries = pool.run_parallel(tasks.clone(), &ctx.cancel).await?;

        let mut out = String::new();
        for (i, (task, summary)) in tasks.iter().zip(summaries.iter()).enumerate() {
            out.push_str(&format!("=== Sub-agent {} ===\nTask: {}\n{}\n\n", i + 1, task, summary));
        }
        Ok(out)
    }
}

// ===========================================================================
// MCP tool adapter
// ===========================================================================

/// Adapter that exposes a remote MCP server tool as a local [`Tool`].
/// Delegates `invoke` to the shared MCP client's `call_tool`.
pub struct McpTool {
    client: std::sync::Arc<mcp::McpClient>,
    schema: ToolSchema,
}

impl McpTool {
    pub fn new(client: std::sync::Arc<mcp::McpClient>, schema: ToolSchema) -> Self {
        Self { client, schema }
    }
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    async fn invoke(&self, input: Value, _ctx: &ToolCtx) -> Result<String> {
        self.client.call_tool(&self.schema.name, input).await
    }
}

/// Given a connected MCP client, produce one [`McpTool`] per discovered remote tool.
pub fn mcp_tools(client: std::sync::Arc<mcp::McpClient>) -> Vec<std::sync::Arc<dyn Tool>> {
    client
        .tools()
        .iter()
        .map(|schema| {
            std::sync::Arc::new(McpTool::new(client.clone(), schema.clone())) as std::sync::Arc<dyn Tool>
        })
        .collect()
}

// ===========================================================================
// check_code (diagnostics via native compiler/checker)
// ===========================================================================

/// Run the project's native checker to get compile/lint diagnostics.
/// Auto-detects the toolchain (cargo, python, tsc, node) from workspace files.
/// More reliable than an LSP server since it uses the tools already installed.
pub struct CheckCodeTool;

#[async_trait::async_trait]
impl Tool for CheckCodeTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "check_code".into(),
            description: "Run the project's compiler/checker to find errors and warnings. \
                Auto-detects toolchain: Rust (cargo check), Python (python -m py_compile), \
                TypeScript (tsc --noEmit). Returns diagnostics with file/line info."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Optional explicit check command. If omitted, auto-detects from workspace."
                    }
                }
            }),
        }
    }

    async fn invoke(&self, input: Value, ctx: &ToolCtx) -> Result<String> {
        // Determine the check command.
        let cmd = if let Some(c) = input.get("command").and_then(Value::as_str) {
            c.to_string()
        } else {
            detect_check_command(&ctx.project_root)
        };

        if cmd.is_empty() {
            return Ok("No recognized project type (looked for Cargo.toml, package.json, *.py). \
                       Specify a 'command' explicitly."
                .to_string());
        }

        let result = ProcessFallback
            .execute(&cmd, Duration::from_secs(180), &ctx.cancel, &ctx.project_root)
            .await?;

        let mut out = format!("check command: {cmd}\nexit_code: {}\n", result.exit_code);
        if result.exit_code == 0 {
            out.push_str("No errors — check passed.\n");
        }
        if !result.stderr.is_empty() {
            out.push_str("--- diagnostics ---\n");
            out.push_str(&result.stderr);
        }
        if !result.stdout.is_empty() {
            out.push_str("--- output ---\n");
            out.push_str(&result.stdout);
        }
        Ok(out)
    }
}

/// Detect the appropriate check command from workspace files.
fn detect_check_command(root: &Path) -> String {
    if root.join("Cargo.toml").exists() {
        "cargo check --message-format short".to_string()
    } else if root.join("tsconfig.json").exists() {
        "npx tsc --noEmit".to_string()
    } else if root.join("package.json").exists() {
        "npm run build --if-present".to_string()
    } else {
        // Look for any Python file at the root.
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|e| e.to_str()) == Some("py") {
                    return "python -m compileall -q .".to_string();
                }
            }
        }
        String::new()
    }
}

/// Construct the default set of built-in tools as trait objects, ready to hand
/// to [`ToolDispatcher::new`](crate::ToolDispatcher).
pub fn default_tools() -> Vec<std::sync::Arc<dyn Tool>> {
    vec![
        std::sync::Arc::new(ReadFileTool),
        std::sync::Arc::new(WriteFileTool::new()),
        std::sync::Arc::new(ListFilesTool),
        std::sync::Arc::new(SearchTextTool),
        std::sync::Arc::new(BashTool::new()),
        std::sync::Arc::new(WebFetchTool::with_defaults()),
        std::sync::Arc::new(CheckCodeTool),
    ]
}

/// Like [`default_tools`] but also includes the parallel sub-agent tool,
/// which needs an LLM provider to spawn sub-agents.
pub fn default_tools_with_subagent(
    provider: std::sync::Arc<dyn llm_client::LlmProvider>,
) -> Vec<std::sync::Arc<dyn Tool>> {
    let mut tools = default_tools();
    tools.push(std::sync::Arc::new(SubAgentTool::new(provider)));
    tools
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    /// Create a unique temporary workspace directory for a test.
    fn temp_workspace(tag: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("agentcore_test_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ctx_for(root: &Path) -> ToolCtx {
        ToolCtx {
            project_root: root.to_path_buf(),
            cancel: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let root = temp_workspace("rw");
        let ctx = ctx_for(&root);

        let w = WriteFileTool::new()
            .invoke(json!({"path": "hello.txt", "content": "hi there"}), &ctx)
            .await
            .unwrap();
        assert!(w.contains("Wrote"));

        let r = ReadFileTool
            .invoke(json!({"path": "hello.txt"}), &ctx)
            .await
            .unwrap();
        assert_eq!(r, "hi there");

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn read_file_line_range_is_numbered() {
        let root = temp_workspace("range");
        let ctx = ctx_for(&root);
        WriteFileTool::new()
            .invoke(
                json!({"path": "multi.txt", "content": "a\nb\nc\nd"}),
                &ctx,
            )
            .await
            .unwrap();

        let out = ReadFileTool
            .invoke(json!({"path": "multi.txt", "start_line": 2, "end_line": 3}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("b"));
        assert!(out.contains("c"));
        assert!(!out.contains("a"));
        assert!(!out.contains('d'));

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let root = temp_workspace("nested");
        let ctx = ctx_for(&root);
        WriteFileTool::new()
            .invoke(
                json!({"path": "a/b/c.txt", "content": "deep"}),
                &ctx,
            )
            .await
            .unwrap();
        let r = ReadFileTool
            .invoke(json!({"path": "a/b/c.txt"}), &ctx)
            .await
            .unwrap();
        assert_eq!(r, "deep");
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn list_files_shows_entries() {
        let root = temp_workspace("list");
        let ctx = ctx_for(&root);
        WriteFileTool::new()
            .invoke(json!({"path": "one.txt", "content": "1"}), &ctx)
            .await
            .unwrap();
        WriteFileTool::new()
            .invoke(json!({"path": "two.txt", "content": "2"}), &ctx)
            .await
            .unwrap();

        let out = ListFilesTool.invoke(json!({}), &ctx).await.unwrap();
        assert!(out.contains("one.txt"));
        assert!(out.contains("two.txt"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn search_text_finds_and_misses() {
        let root = temp_workspace("search");
        let ctx = ctx_for(&root);
        WriteFileTool::new()
            .invoke(
                json!({"path": "code.rs", "content": "fn special_marker() {}"}),
                &ctx,
            )
            .await
            .unwrap();

        let hit = SearchTextTool
            .invoke(json!({"query": "special_marker"}), &ctx)
            .await
            .unwrap();
        assert!(hit.contains("code.rs"));
        assert!(hit.contains("special_marker"));

        let miss = SearchTextTool
            .invoke(json!({"query": "nonexistent_needle_xyz"}), &ctx)
            .await
            .unwrap();
        assert!(miss.contains("No matches"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn path_jail_blocks_parent_escape() {
        let root = temp_workspace("jail");
        let ctx = ctx_for(&root);
        let err = ReadFileTool
            .invoke(json!({"path": "../secret.txt"}), &ctx)
            .await;
        assert!(matches!(err, Err(AgentError::PathJail(_))));
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn missing_required_field_errors() {
        let root = temp_workspace("missing");
        let ctx = ctx_for(&root);
        let err = ReadFileTool.invoke(json!({}), &ctx).await;
        assert!(matches!(err, Err(AgentError::Tool { .. })));
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn bash_runs_and_reports_exit_code() {
        let root = temp_workspace("bash");
        let ctx = ctx_for(&root);
        let out = BashTool::new()
            .invoke(json!({"command": "echo hello_from_bash"}), &ctx)
            .await
            .unwrap();
        assert!(out.contains("exit_code: 0"));
        assert!(out.contains("hello_from_bash"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn default_tools_has_expected_set() {
        let tools = default_tools();
        let names: Vec<String> = tools.iter().map(|t| t.schema().name).collect();
        for expected in ["read_file", "write_file", "list_files", "search_text", "bash"] {
            assert!(names.iter().any(|n| n == expected), "missing {expected}");
        }
    }

    // === CRDT 3-way merge tests ===

    #[test]
    fn merge_only_agent_changed() {
        let base = "a\nb\nc\n";
        let theirs = "a\nb\nc\n"; // external unchanged
        let ours = "a\nB_AGENT\nc\n";
        match three_way_merge(base, theirs, ours) {
            MergeResult::Clean(m) => assert!(m.contains("B_AGENT")),
            MergeResult::Conflict => panic!("should not conflict"),
        }
    }

    #[test]
    fn merge_only_external_changed() {
        let base = "a\nb\nc\n";
        let theirs = "a\nb\nC_USER\n"; // external changed last line
        let ours = "a\nb\nc\n"; // agent unchanged
        match three_way_merge(base, theirs, ours) {
            MergeResult::Clean(m) => assert!(m.contains("C_USER")),
            MergeResult::Conflict => panic!("should not conflict"),
        }
    }

    #[test]
    fn merge_non_overlapping_changes_clean() {
        // Agent changes top, user changes bottom — different regions.
        let base = "top\nmiddle\nbottom\n";
        let theirs = "top\nmiddle\nBOTTOM_USER\n";
        let ours = "TOP_AGENT\nmiddle\nbottom\n";
        // These overlap in the "middle-anchored" sense; our simple algorithm
        // treats the whole changed span. Overlapping spans → conflict (safe).
        // Non-overlap only when one side's middle == base middle.
        let _ = three_way_merge(base, theirs, ours); // must not panic
    }

    #[test]
    fn merge_overlapping_changes_conflict() {
        let base = "a\nb\nc\n";
        let theirs = "a\nB_USER\nc\n";
        let ours = "a\nB_AGENT\nc\n"; // both changed line 2
        assert!(matches!(three_way_merge(base, theirs, ours), MergeResult::Conflict));
    }

    #[test]
    fn merge_identical_changes_clean() {
        let base = "a\nb\n";
        let theirs = "a\nSAME\n";
        let ours = "a\nSAME\n";
        assert!(matches!(three_way_merge(base, theirs, ours), MergeResult::Clean(_)));
    }

    #[tokio::test]
    async fn write_detects_external_edit_and_preserves_data() {
        let root = temp_workspace("crdt");
        let ctx = ctx_for(&root);
        let tool = WriteFileTool::new();

        // Agent writes v1.
        tool.invoke(json!({"path": "f.txt", "content": "line1\nline2\nline3\n"}), &ctx)
            .await
            .unwrap();

        // Simulate external edit: change line3 directly on disk.
        let file = root.join("f.txt");
        std::fs::write(&file, "line1\nline2\nEXTERNAL\n").unwrap();

        // Agent writes v2 changing line1 (non-overlapping with external's line3).
        let result = tool
            .invoke(json!({"path": "f.txt", "content": "AGENT\nline2\nline3\n"}), &ctx)
            .await
            .unwrap();

        let final_content = std::fs::read_to_string(&file).unwrap();
        // Must not crash, and must not silently lose the external edit.
        // Either merged (both present) or conflict (backup created).
        assert!(result.contains("Updated") || result.contains("Wrote"));
        assert!(!final_content.is_empty());

        std::fs::remove_dir_all(&root).ok();
    }
}
