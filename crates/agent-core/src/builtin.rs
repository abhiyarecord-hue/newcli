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
use sandbox::{PathJail, ProcessFallback, SandboxExecutor};
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
pub struct WriteFileTool;

#[async_trait::async_trait]
impl Tool for WriteFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "write_file".into(),
            description: "Create a new file or overwrite an existing one in the \
                workspace with the given content. Parent directories are created \
                automatically."
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
        tokio::fs::write(&safe, content.as_bytes()).await?;

        let bytes = content.len();
        let lines = content.lines().count();
        Ok(format!("Wrote {bytes} bytes ({lines} lines) to {path}"))
    }
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

/// Construct the default set of built-in tools as trait objects, ready to hand
/// to [`ToolDispatcher::new`](crate::ToolDispatcher).
pub fn default_tools() -> Vec<std::sync::Arc<dyn Tool>> {
    vec![
        std::sync::Arc::new(ReadFileTool),
        std::sync::Arc::new(WriteFileTool),
        std::sync::Arc::new(ListFilesTool),
        std::sync::Arc::new(SearchTextTool),
        std::sync::Arc::new(BashTool::new()),
    ]
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

        let w = WriteFileTool
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
        WriteFileTool
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
        WriteFileTool
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
        WriteFileTool
            .invoke(json!({"path": "one.txt", "content": "1"}), &ctx)
            .await
            .unwrap();
        WriteFileTool
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
        WriteFileTool
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
}
