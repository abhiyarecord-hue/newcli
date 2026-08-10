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
| **Google Gemini** | 3.5 Flash, 3.5 Pro, 3.1 Pro, 3.1 Flash-Lite | ✅ Default; covered by mock-server tests |
| **OpenAI** | GPT-5.6 Sol, GPT-5.5, GPT-5.4, GPT-5 | 🔧 Implemented (community-tested) |
| **Anthropic** | Claude Fable 5, Opus 4.8, Sonnet 5, Haiku 4.5 | 🔧 Implemented (community-tested) |
| **Mistral** | Medium 3.5, Small 4, Large 3 | 🔧 Implemented (community-tested) |
| **DeepSeek** | V4-Pro, V4-Flash, V3.1 | 🔧 Implemented (community-tested) |
| **Ollama** | Llama 3.3, Qwen 3, any local model | 🔧 Implemented (FREE, offline) |

> **What "tested" means here:** provider request/response handling, streaming, retries, and error
> mapping are exercised against local mock HTTP servers in the automated suite. **Live network
> interoperability against the real provider APIs is not part of that suite and has not been
> executed**, so no provider is claimed as end-to-end verified. Other providers use the same
> `LlmProvider` trait — contributions and test reports welcome!

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
| `cli spec specify --from-file req.md` | Same, reading the description from a file — preferred for anything long |
| `cli eval run` | Run evaluation suite (SWE-bench-lite format) |
| `cli eval diff run-a run-b` | Compare two eval runs, detect regressions |
| `cli serve --port 9527` | Start IPC server for editor integration (loopback only, unauthenticated) |

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
- **dispatch_subagent** — Spawn parallel sub-agents to reason about independent questions.
  **Reasoning-only by design: sub-agents are given no tools**, so they cannot read, search, or edit
  the repository themselves — they only work from the task text they are handed
- **MCP tools** — Auto-loaded from `.agent/mcp.json` external MCP servers

### Interactive Commands (in chat)
- `/remember <text>` — Save a note to long-term memory (persists across sessions)
- `/undo` — Revert the file changes from the last agent turn (git checkpoint)
- `/quit` — Exit

### Security

> These are **best-effort policy layers, not an isolation boundary.** There is no MicroVM and no
> hard network isolation. Command scanning is normalized across Unix shells, `cmd.exe`, and
> PowerShell, but it is not a proof of shell parsing. Run untrusted work in a real VM or container.

- **PathJail** — Confines file access to the workspace: blocks directory traversal, symlink escapes
- **SecretLeakHook** — Scans for AWS keys, API tokens, private keys before writes/execution
- **DestructiveCommandHook** — Recognizes and blocks known destructive forms (`rm -rf /`, force push, mkfs)
- **User Approval** — Agent asks permission before running risky commands (installs, deletions, git push, downloads)

### Code Intelligence
- **Tree-sitter parsing** — Rust, Python, TypeScript entity extraction (functions, classes, methods)
- **AST chunking** — Token-budget-aware code chunking with entity boundaries
- **Hybrid search** — Vector KNN (sqlite-vec) + BM25 (FTS5), fused with RRF. A graph-traversal mode
  exists in the schema and search path, but **indexing does not populate the graph tables yet**, so
  graph mode currently returns no results
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
- **Execution mode: `check_only`.** Each case runs its `check_cmd`; the agent is **not** driven
  against `case.prompt`. Turn, tool-call, and token counts are omitted rather than reported as zero,
  because they are not measured in this mode. Full-agent SWE-bench execution is a separate
  unimplemented feature.
- SWE-bench-lite format (TOML cases + check commands)
- JSONL results with trajectory recording
- Regression detection (pass→fail = hard regression, CI gate)

## Provider Configuration

```powershell
# Google Gemini (default)
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

### MCP trust gate

Repository-controlled MCP servers do **not** start on trust alone. A server defined by the workspace
is spawned only after an explicit approval that is recorded **outside the repository**, in the
user-local OS config directory, so a cloned repository can never ship its own approval. The stored
record is versioned (`TRUST_SCHEMA_VERSION = 1`) and keyed by two hashes — `workspace_hash` and
`config_hash` — computed over the canonical workspace path and the effective command, arguments, and
environment.

Consequences you should expect:

- **Editing an MCP server's command, arguments, or environment invalidates the prior approval.**
  The `config_hash` changes, so you are asked again. Reverting the edit restores the earlier hash and
  the earlier decision applies again.
- **Approval prompts and stored records are redacted.** Environment values are never persisted, so a
  secret placed in an MCP server's `env` does not land in the trust store.
- **Without a TTY the request is denied and the denial is not persisted.** A non-interactive run
  cannot silently grant trust, and it also cannot poison a later interactive decision.
- Configuration parsing is strict and bounded: unknown fields are rejected, and the config and trust
  store are each capped at `MAX_CONFIG_BYTES` / `MAX_TRUST_STORE_BYTES` (1 MiB).

### Custom tool effect declarations

Tools declare what they may do via `ToolEffects` (`workspace_read`, `workspace_write`, and the other
boolean fields). Declarations **fail closed**: the struct is `#[serde(default, deny_unknown_fields)]`
and a missing declaration deserializes to `ToolEffects::UNKNOWN`, in which every capability is
`false`. An older or newer schema therefore loses capability rather than silently gaining it. If you
add a custom tool and omit its effects, expect it to be treated as capability-less, not trusted.

## Migration and Compatibility

Upgrading an existing workspace touches four on-disk contracts. Each migration is designed to be
non-destructive: on failure the original data is left in place.

### Conversation history → v2 snapshot

The authoritative format is a versioned snapshot at `.agent/HISTORY.v2.json`
(`CONVERSATION_SCHEMA_VERSION = 2`), atomically replaced after a completed turn or compaction.

- The legacy append-only `.agent/HISTORY.jsonl` **remains readable** and is imported once. Malformed
  lines are reported by line number and incomplete tool groups are trimmed.
- The legacy file is **retained as a migration backup** and is not deleted. A failed migration is
  non-destructive: `.agent/HISTORY.jsonl` is left in place.
- `generation` increases on every successful replacement, and `next_message_id` is monotonic and
  never reused even after compaction drops earlier messages.
- A missing `messages` field is treated as a **corrupt snapshot, not an empty conversation**, so a
  damaged file can never load as a silent reset.
- No downgrade: an older build must not be pointed at a newer snapshot.
- On load, history is pruned to the most recent `MAX_HISTORY_MESSAGES` (200) messages.

### Long-term memory is unchanged

`/remember` and persona state live in separate files under `.agent/` — `MEMORY.md` (append-only, with
ISO-8601 timestamps), `SOUL.md`, and `HEARTBEAT.md`. **These are a separate contract from
conversation history and are not migrated, rewritten, or cleared by this work.** `/clear` removes the
v2 snapshot and the legacy JSONL store and never touches the three memory files.

### VecStore: when you must re-index

Stored embeddings carry their identity as columns on `chunks`: `embedding_provider`,
`embedding_model`, `embedding_dimension`, and `embedding_valid`. Provider, model, and dimension must
**all** match exactly (`EmbeddingProfile`) for a stored vector to be considered comparable with a
query vector. Defaults are `DEFAULT_EMBEDDING_PROVIDER = "gemini"`,
`DEFAULT_EMBEDDING_MODEL = "text-embedding-004"`, `EMBEDDING_DIMENSION = 768`.

Those constants describe the profile recorded when **no** embedder is configured, which is the
keyword-only case where vectors are placeholders and `embedding_valid = 0`. They are not a model
choice. When embeddings are enabled, the profile comes from the embedder itself: the Gemini path now
uses `gemini-embedding-001` by default, since Google deprecated `text-embedding-004` in January 2026,
and any model may be named with `EMBEDDING_MODEL`. The width is **measured** from a real response
rather than assumed, so a 3072- or 1536-wide model is recorded at its true width.

On mismatch, search **degrades to BM25-only instead of returning wrong neighbours**, and tells you
which case you hit:

```
BM25-only: no valid embeddings match provider '<p>' model '<m>' dimension <d>; run `srijandev index` with the same embedding configuration to rebuild semantic data.
```

```
BM25-only: query embedding dimension <n> does not match configured dimension <d>; regenerate the query embedding or re-index with provider '<p>' model '<m>'.
```

So: **changing the embedding provider, model, or dimension requires a full re-index.** Keyword
search keeps working in the meantime. Index replacement is transactional, so an interrupted re-index
does not leave a half-swapped index.

A width change is handled for you rather than failing per insert. The `chunks_vec` virtual table is
created for one fixed width, so `cli index` rebuilds it when the configured embedder produces a
different width, marks every existing chunk `embedding_valid = 0`, and regenerates vectors. **Chunk
text, file records, and the keyword index survive that rebuild**; only the vectors are dropped,
because a 768-wide vector cannot be reinterpreted as 1536-wide and keeping it would make comparisons
silently wrong.

### Embeddings: any provider, not just one

Semantic search is not tied to a vendor. Two request shapes cover the practical field:

- **Gemini** `embedContent`.
- **OpenAI-compatible** `POST {base}/embeddings`, which is what OpenAI, Azure AI Foundry, Mistral,
  DeepSeek, Together, OpenRouter, and local runtimes such as Ollama, LM Studio, and vLLM all speak.

| Variable | Purpose |
|----------|---------|
| `EMBEDDING_PROVIDER` | `gemini`, `openai`, `azure`, `mistral`, `deepseek`, `together`, `openrouter`, `ollama`, `lmstudio`, `vllm`, or any label you like when you also set `EMBEDDING_BASE_URL`. Unset falls back to `GEMINI_API_KEY` if present, otherwise keyword-only |
| `EMBEDDING_MODEL` | Model name, or the **deployment name** on Azure. Required for non-Gemini providers |
| `EMBEDDING_BASE_URL` | API root. Optional when the provider has a built-in default; required for anything else |
| `EMBEDDING_API_KEY` | Embedding credential. Falls back to `LLM_API_KEY`, then `OPENAI_API_KEY`, then `GEMINI_API_KEY` |
| `EMBEDDING_DIMENSIONS` | Request a specific width. Sent as `dimensions` and enforced on the response |

Two behaviours worth knowing:

- **No credential is required for local runtimes.** Ollama, LM Studio, and vLLM need none, so an
  empty key is not treated as an error for them.
- **Width is measured, not assumed.** If the width is not known from the model, one small embedding
  request is made at startup to observe it. Building an index at the wrong width does not fail
  loudly, it just returns wrong neighbours, so the extra call is deliberate. If you set
  `EMBEDDING_DIMENSIONS` and the endpoint ignores it, the mismatch is reported as an error rather
  than written into the index.

```powershell
# Local, no key needed
$env:EMBEDDING_PROVIDER = "ollama"; $env:EMBEDDING_MODEL = "nomic-embed-text"

# OpenAI
$env:EMBEDDING_PROVIDER = "openai"; $env:EMBEDDING_MODEL = "text-embedding-3-small"

# Azure AI Foundry — EMBEDDING_MODEL is the deployment name
$env:EMBEDDING_PROVIDER = "azure"
$env:EMBEDDING_BASE_URL = "https://<resource>.services.ai.azure.com/openai/v1"
$env:EMBEDDING_MODEL = "<your-embedding-deployment>"
```

> **Not yet verified against a live endpoint.** The OpenAI-compatible embedder is covered by tests
> against local mock HTTP servers that exercise the real request and response shapes, including
> batching, index ordering, width enforcement, and error redaction. No live embedding endpoint has
> been contacted, so this remains `not_run` in the evidence manifest rather than being claimed as
> end-to-end tested.

### Spec pipeline: Tests artifact became a directory

The Tests stage artifact root is now the directory `tests/` with primary file `tests/test-plan.md`.
A workspace where `tests` is a legacy **regular file** is migrated by renaming that file to
`tests.backup`; the new directory is created only after the rename succeeds.

Migration is serialized by an exclusive `tests.migration.lock` and **refuses rather than deletes**:
it stops without removing the legacy file if the backup name `tests.backup` already exists, if the
source is no longer an unambiguous regular file (including a symlink standing in for it), or if the
rename fails. Resolve the reported condition and rerun.

### IPC v1 is backward compatible

The framed protocol uses a tagged envelope (`"type"` with PascalCase names): inbound `OpenDocument`,
`CloseDocument`, `ApplyPatches`, `Subscribe`; outbound `Ack`, `Error`, `DocumentUpdate`. Every
response carries the originating `request_id`.

Legacy untagged `PatchMessage` frames are still accepted **for documents that are already
registered**; an untagged frame for an unregistered document is rejected rather than implicitly
creating one. Note that a `Subscribe` request is answered with its `Ack` first — a
`DocumentUpdate` arrives on subsequent frames, not as the reply to the subscription.

## Bounds and Durability

### Default bounds (all bounded, most configurable)

| Bound | Default | Where |
|-------|---------|-------|
| IPC frame size | 1 MiB | `IpcConfig::max_frame_bytes` |
| IPC read idle timeout | 30 s | `IpcConfig::read_idle_timeout` |
| IPC single-write deadline | 5 s | `IpcConfig::write_deadline` |
| IPC subscriber queue | 64 messages | `IpcConfig::subscriber_queue_capacity` |
| LSP message body cap | 16 MiB | `MAX_CONTENT_LENGTH` |
| LSP header block cap | 8 KiB | `MAX_HEADER_BYTES` |
| Captured stdout/stderr per process | 1 MiB | `DEFAULT_OUTPUT_LIMIT` |
| Process termination grace | 250 ms | `DEFAULT_TERMINATION_GRACE` |
| Retained history messages | 200 | `MAX_HISTORY_MESSAGES` |
| Agent tool-loop iterations | 200 | `MAX_ITERATIONS` |
| Auto-continue after `MaxTokens` | 5 | `MAX_CONTINUATIONS` |
| Atomic temp-name attempts | 64 (ceiling 1024) | `DEFAULT_TEMP_CREATE_ATTEMPTS` / `MAX_TEMP_CREATE_ATTEMPTS` |
| Atomic replace attempts | 16 (ceiling 128) | `DEFAULT_REPLACE_ATTEMPTS` / `MAX_REPLACE_ATTEMPTS` |
| MCP config / trust store size | 1 MiB each | `MAX_CONFIG_BYTES` / `MAX_TRUST_STORE_BYTES` |

`IpcConfig` is injectable via `IpcServer::bind_with_config`. The LSP caps are configurable downward
by embedders. Bounds are enforced *before* allocation, so an oversized frame or `Content-Length` is
rejected without first reserving the claimed size. A subscriber that stops reading is dropped rather
than allowed to stall the server.

### Durability is scoped to local filesystems

Writes are committed by preparing an exclusively-created temporary file in the destination directory
(named `.<file>.atomic-<seed>-<attempt>.tmp`), writing it, `sync_all`-ing it, then replacing the
destination in the namespace. A successful return means the complete byte slice is visible at the
destination, and cancellation is checked immediately before the replacement so a committed write is
never reported as an ambiguous error.

**These guarantees assume a local filesystem with normal atomic rename/replace behavior.** Network
shares and removable or FAT-like filesystems may provide weaker replacement, flush, or
crash-durability semantics — **crash durability is not promised there.** On Unix, parent-directory
flush is best effort, because not every filesystem permits opening or flushing a directory.

On Unix the temporary is created `0600` so no reader observes partial content, and the destination
mode is applied after the content is written (new files land at `NEW_DESTINATION_MODE = 0o644`)
rather than letting the rename silently strip bits from an existing file.

### Windows sharing violations

On Windows a replacement can transiently fail because another process — an editor, indexer, or
antivirus scanner — holds the destination open. Replacement uses `ReplaceFileW`, falling back to
`MoveFileExW` with `MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH` when the destination does not
yet exist, and retries only these three transient codes:

- `32` `ERROR_SHARING_VIOLATION`
- `1175` `ERROR_UNABLE_TO_REMOVE_REPLACED`
- `1176` `ERROR_UNABLE_TO_MOVE_REPLACEMENT`

Each leaves the temporary owned by the cleanup guard, so a bounded retry stays unambiguous. Retries
are `DEFAULT_REPLACE_ATTEMPTS` (16), hard-capped at `MAX_REPLACE_ATTEMPTS` (128), spaced 1 ms apart;
exclusive temporary-name creation retries `DEFAULT_TEMP_CREATE_ATTEMPTS` (64) times, capped at
`MAX_TEMP_CREATE_ATTEMPTS` (1024). **Any other error fails immediately, and if the holder never
releases the file the operation fails with the underlying OS error rather than silently skipping the
write.** If you see this persistently, close editors holding the file or exclude the workspace from
real-time scanning.

## Work In Progress

These features have code/structure but are not yet fully production-ready:

- **LSP client** — Full LSP client exists (goto-def, find-refs); the `check_code` tool
  currently provides diagnostics via native compilers. LSP-based navigation tools planned.
- **MicroVM sandbox** — Currently a best-effort process policy with user approval. It is **not** a
  security boundary: there is no MicroVM and no hard network isolation, so a determined command can
  still reach the host; true VM isolation planned for Linux
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
