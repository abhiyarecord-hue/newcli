# NewGen CLI — Autonomous AI Coding Agent (Rust)

A fully autonomous, terminal-based AI coding agent built entirely in Rust. It reads your codebase, understands it semantically, plans changes through a structured spec pipeline, and executes edits — all from your terminal.

**What makes it different:**
- 🇮🇳 **Hinglish-first** — The only AI coding agent with native Hinglish support (Hindi in Latin script) for prose, while keeping all code/paths strictly English. Built for millions of Indian developers.
- 🔌 **Offline-capable** — First-class Ollama/local-model support. Run a fully offline AI coding agent — your proprietary code never leaves your machine.
- 🦀 **Pure Rust, no lock-in** — Single binary, 6 LLM providers, no vendor lock-in, no Python/Docker dependency.
- 📦 **Library + app** — Use the crates (`agent-core`, `sandbox`, `llm-client`) to build your own agents.

## What It Does

You give it a task in natural language, and it:

1. Indexes your codebase into a semantic search store (tree-sitter + FTS5 + vector embeddings)
2. Streams responses from your chosen LLM with real-time tool use
3. Reads, writes, and searches files autonomously
4. Merges concurrent edits safely (CRDT 3-way merge — no data loss)
5. Enforces security boundaries (path jail, secret detection, user approval for risky commands)

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| **Google Gemini** | 3.5 Flash, 3.5 Pro, 3.1 Pro, 3.1 Flash-Lite | ✅ Fully tested |
| **OpenAI** | GPT-5.6 Sol, GPT-5.5, GPT-5.4, GPT-5 | 🔧 Implemented (community-tested) |
| **Anthropic** | Claude Fable 5, Opus 4.8, Sonnet 5, Haiku 4.5 | 🔧 Implemented (community-tested) |
| **Mistral** | Medium 3.5, Small 4, Large 3 | 🔧 Implemented (community-tested) |
| **DeepSeek** | V4-Pro, V4-Flash, V3.1 | 🔧 Implemented (community-tested) |
| **Ollama** | Llama 3.3, Qwen 3, any local model | 🔧 Implemented (FREE, offline) |

> Gemini is fully tested by us. Other providers use the same `LlmProvider` trait — contributions and test reports welcome!

## Quick Start

```bash
# Build
cargo build --release -p cli

# Set your provider and key
export LLM_PROVIDER=gemini
export LLM_API_KEY=your-api-key

# Start chatting
./target/release/cli chat
```

## CLI Commands

| Command | What It Does |
|---------|-------------|
| `cli chat` | Interactive AI agent — reads/writes files, runs commands |
| `cli index` | Index codebase (Merkle diff → tree-sitter → chunk → FTS5 + embeddings) |
| `cli search "query"` | Search indexed code (keyword BM25 + vector hybrid) |
| `cli spec specify` | Run RustySpec pipeline stage (7-stage structured workflow) |
| `cli eval run` | Run evaluation suite (SWE-bench-lite format) |
| `cli eval diff run-a run-b` | Compare two eval runs, detect regressions |
| `cli serve --port 9527` | Start IPC server for editor integration |

## Key Features

### Agent Tools
- **read_file** — Read files with optional line ranges
- **write_file** — Create/overwrite files with CRDT merge (safe concurrent editing)
- **edit_file** — Targeted str_replace-style partial edit (token-efficient, preferred for modifications)
- **list_files** — Directory listing
- **search_text** — Recursive text search across workspace
- **bash** — Shell commands with user approval for risky operations
- **web_fetch** — Fetch docs from allowlisted domains (SSRF-protected via NetGuard)
- **check_code** — Run project compiler/checker for diagnostics (cargo/tsc/python) — powers the self-healing edit loop
- **dispatch_subagent** — Spawn parallel sub-agents for independent research tasks
- **MCP tools** — Auto-loaded from `.agent/mcp.json` external MCP servers

### Interactive Commands (in chat)
- `/remember <text>` — Save a note to long-term memory (persists across sessions)
- `/undo` — Revert the file changes from the last agent turn (git checkpoint)
- `/quit` — Exit

### Security
- **PathJail** — Blocks directory traversal, symlink escapes, access outside workspace
- **SecretLeakHook** — Blocks AWS keys, API tokens, private keys from being written/executed
- **DestructiveCommandHook** — Blocks `rm -rf /`, force push, mkfs
- **User Approval** — Agent asks permission before running risky commands (installs, deletions, git push, downloads)

### Code Intelligence
- **Tree-sitter parsing** — Rust, Python, TypeScript entity extraction (functions, classes, methods)
- **AST chunking** — Token-budget-aware code chunking with entity boundaries
- **Hybrid search** — Vector KNN (sqlite-vec) + BM25 (FTS5) + Graph traversal, fused with RRF
- **Incremental sync** — Merkle tree diffing, only re-indexes changed files

### Agentic Architecture
- **Multi-turn tool loop** — LLM decides which tools to call, processes results, continues autonomously (up to 200 iterations)
- **CRDT concurrent editing** — 3-way merge detects external edits, merges non-overlapping changes, backs up conflicts
- **Thinking display** — Live thought summaries from Gemini 3.x thinking models
- **Token tracking** — Real-time token usage per turn and session total
- **Repetition detection** — Breaks out of loops when model gets stuck

### Structured Spec Pipeline (RustySpec)
7-stage workflow: Specify → Clarify → Plan → Tasks → Tests → Implement → Analyze. Each stage produces versioned markdown artifacts with prerequisite validation.

### Evaluation Harness
- SWE-bench-lite format (TOML cases + check commands)
- JSONL results with trajectory recording
- Regression detection (pass→fail = hard regression, CI gate)

## Provider Configuration

```powershell
# Google Gemini (default, tested)
$env:LLM_PROVIDER = "gemini"
$env:LLM_API_KEY = "your-gemini-key"
$env:LLM_MODEL = "gemini-3.5-flash"

# OpenAI
$env:LLM_PROVIDER = "openai"
$env:LLM_API_KEY = "sk-..."
$env:LLM_MODEL = "gpt-5.6-sol"

# Anthropic (Claude Fable 5)
$env:LLM_PROVIDER = "anthropic"
$env:LLM_API_KEY = "sk-ant-..."
$env:LLM_MODEL = "claude-fable-5"

# Mistral
$env:LLM_PROVIDER = "mistral"
$env:LLM_API_KEY = "..."
$env:LLM_MODEL = "mistral-medium-3.5"

# DeepSeek
$env:LLM_PROVIDER = "deepseek"
$env:LLM_API_KEY = "..."
$env:LLM_MODEL = "deepseek-v4-pro"

# Ollama (local, FREE)
$env:LLM_PROVIDER = "ollama"
$env:LLM_MODEL = "llama3.3"
```

## Architecture

```
bin/cli                    CLI entrypoint (clap subcommands)
 │
 ▼
crates/agent-core          Turn orchestrator + tool dispatcher + builtin tools
 │
 ├── crates/llm-client     Multi-provider LLM (Gemini, OpenAI, Anthropic, Mistral, DeepSeek, Ollama)
 ├── crates/compaction     Context window compaction engine
 ├── crates/harness        Hooks, skills, sub-agents, language guard
 ├── crates/spec-pipeline  7-stage RustySpec workflow
 ├── crates/state-store    Persistent + ephemeral state
 ├── crates/mcp            Model Context Protocol client/server
 ├── crates/indexer        Tree-sitter parsing, AST chunking, Merkle tree
 ├── crates/vecstore       SQLite + sqlite-vec + FTS5 hybrid retrieval
 ├── crates/lsp-client     Language Server Protocol client
 ├── crates/apply-engine   CRDT docs, semantic diff, fast-apply
 ├── crates/sandbox        Path jail, process executor, SSRF network guard
 └── crates/evals          SWE-bench runner, trajectory tracking, regression diff
 │
 ▼
crates/runtime-core        Structured concurrency (TaskScope, EventBus, Scheduler)
 │
 ▼
crates/agent-types         Shared types (Message, Tool, Error, LanguageMode)
```

## Hinglish Mode

Set `LanguageMode::Hinglish` for prose output in Hindi-Latin script. All code, tool calls, paths, and JSON remain strictly English/ASCII. The `SchemaLangGuard` enforces this deterministically.

## Development

```bash
# Run all tests (149+)
cargo test --workspace

# Run specific crate tests
cargo test -p agent-core
cargo test -p sandbox
cargo test -p harness

# Check compilation
cargo check --workspace
```

## Security Model

Every file operation goes through `PathJail`:
- Rejects `..` path components
- Canonicalizes paths, detects symlink escapes
- Works for non-existing paths (safe for new file creation)

Network access restricted by `NetGuard`:
- HTTPS only, domain allowlist
- DNS resolution checked against private/loopback/link-local IPs
- Manual redirect following (max 3 hops)

Policy hooks run before every tool invocation:
- `SecretLeakHook` — blocks leaked credentials
- `DestructiveCommandHook` — blocks dangerous shell commands
- `SchemaLangGuard` — ensures machine surfaces stay ASCII

## Editor Integration (IPC Server)

Run `cli serve --port 9527` to start the IPC server. A VS Code extension (or any
editor plugin) can connect via newline-delimited JSON PatchMessages over TCP loopback.
The server applies incoming edits through the CRDT engine, enabling real-time
concurrent editing between the agent and your editor. The VS Code extension itself
is planned as a separate project.

## MCP (Model Context Protocol)

Add external MCP servers in `.agent/mcp.json`:

```json
{
  "servers": [
    { "name": "github", "command": "mcp-github", "args": [] },
    { "name": "postgres", "command": "mcp-postgres", "args": ["--dsn", "..."] }
  ]
}
```

Their tools are auto-discovered and added to the agent's toolset on startup.

## Work In Progress

These features have code/structure but are not yet fully production-ready:

- **LSP client** — Full LSP client exists (goto-def, find-refs); the `check_code` tool
  currently provides diagnostics via native compilers. LSP-based navigation tools planned.
- **MicroVM sandbox** — Currently uses process-based isolation with user approval; true VM isolation planned for Linux
- **SWE-bench actual runs** — Runner infrastructure + 300 cases ready; full benchmark execution pending
- **VS Code extension** — IPC backend ready (`cli serve`); extension is a planned separate project

## Supported Languages (Indexing)

| Language | Extensions | Parser |
|----------|-----------|--------|
| Rust | `.rs` | tree-sitter-rust |
| Python | `.py`, `.pyi` | tree-sitter-python |
| TypeScript | `.ts`, `.tsx` | tree-sitter-typescript |

## Contributing

Contributions welcome! Especially:
- Testing with OpenAI/Anthropic/Mistral/DeepSeek keys and reporting results
- Adding new tool implementations
- SWE-bench case creation and evaluation
- VS Code extension development

Implement `LlmProvider` trait to add new AI backends — see `crates/llm-client/src/openai_compat.rs` for the pattern.

## License

MIT
