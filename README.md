# Rust Autonomous AI Coding Agent

A fully autonomous, terminal-based AI coding agent built entirely in Rust. It reads your codebase, understands it semantically, plans changes through a structured spec pipeline, and executes edits — all from your terminal. No IDE plugin required.

## What Is This?

This is a standalone CLI application that acts as an AI pair-programmer. You give it a task in natural language (English or Hinglish), and it:

1. Indexes your codebase into a semantic vector store
2. Streams responses from an LLM (Anthropic Claude) with real-time tool use
3. Reads, writes, and searches files autonomously
4. Validates its own edits via LSP diagnostics and semantic diff
5. Enforces security boundaries (path jail, secret detection, SSRF prevention)

Think of it as a Rust-native alternative to AI coding assistants — running locally, with full control over what it can access.

## Key Features

- **Streaming LLM integration** — SSE-based streaming with the Anthropic Messages API; provider-agnostic trait for swapping backends
- **Semantic code indexing** — Tree-sitter parsing (Rust, Python, TypeScript) extracts functions, classes, methods with qualified names and scope trees
- **Hybrid search** — SQLite-backed vector KNN (sqlite-vec), BM25 full-text (FTS5), and graph-based entity traversal, fused with Reciprocal Rank Fusion
- **Incremental sync** — Merkle tree diffing detects only changed files across runs, skipping re-indexing of unchanged code
- **Structured spec pipeline** — 7-stage "RustySpec" workflow (Specify → Clarify → Plan → Tasks → Tests → Implement → Analyze) producing versioned markdown artifacts
- **Security sandbox** — PathJail prevents directory traversal, SandboxExecutor runs commands with hard timeouts and output caps, NetGuard blocks SSRF against private IPs
- **Policy hooks** — Deterministic hook engine scans for leaked secrets (AWS keys, API tokens, private keys) and blocks destructive commands (rm -rf /, force push)
- **Hinglish mode** — Prose output in Hindi-Latin script while all code, paths, and tool schemas remain strictly English/ASCII, with a deterministic language guard
- **Evaluation harness** — SWE-bench-lite compatible runner with trajectory recording and regression detection (pass/fail flip = hard regression, CI gate)
- **CRDT concurrent editing** — Agent and editor can edit the same file simultaneously; convergent state via operational patches over IPC
- **Session compaction** — Automatic context-window management that preserves tool-use pair integrity and summarizes compacted history

## Architecture

```
bin/cli                    CLI entrypoint (clap subcommands)
 |
 v
crates/agent-core          Turn orchestrator + tool dispatcher
 |
 +-- crates/llm-client     LLM streaming (SSE parser, Anthropic provider)
 +-- crates/compaction     Context window compaction engine
 +-- crates/harness        Hooks, skills, sub-agents, language guard
 +-- crates/spec-pipeline  7-stage spec workflow
 +-- crates/state-store    Persistent (SOUL/HEARTBEAT/MEMORY.md) + ephemeral state
 +-- crates/mcp            Model Context Protocol client/server (JSON-RPC)
 +-- crates/indexer        Tree-sitter parsing, AST chunking, Merkle tree
 +-- crates/vecstore       SQLite + sqlite-vec + FTS5 hybrid retrieval
 +-- crates/lsp-client     Language Server Protocol client (stdio, diagnostics)
 +-- crates/apply-engine   Lazy edits, fast-apply, semantic diff, CRDT, IPC
 +-- crates/sandbox        Path jail, process executor, SSRF network guard
 +-- crates/evals          SWE-bench runner, trajectory tracking, regression diff
 |
 v
crates/runtime-core        Structured concurrency (TaskScope, EventBus, Scheduler)
 |
 v
crates/agent-types         Shared types (Message, Tool, Error, LanguageMode)
```

## CLI Usage

```bash
# Build
cargo build --release

# Interactive chat with the agent
cargo run -p cli -- chat

# Index your codebase (Merkle diff → parse → chunk → embed → store)
cargo run -p cli -- index

# Search indexed code
cargo run -p cli -- search "authentication middleware" -k 5

# Run a spec pipeline stage
cargo run -p cli -- spec specify

# Run evaluation suite
cargo run -p cli -- eval run --suite swebench-lite --max-concurrent 4

# Set eval baseline and compare runs
cargo run -p cli -- eval baseline 2025-07-10T12:00:00Z
cargo run -p cli -- eval diff run-a run-b
```

## Configuration

Create `.agent/config.toml` in your project root:

```toml
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"   # name of the env var, NOT the key itself
language = "en"                      # or "hinglish"
allowed_domains = ["crates.io", "docs.rs", "github.com"]
```

The agent stores persistent state in `.agent/`:
- `SOUL.md` — persona, policies, language preference
- `HEARTBEAT.md` — current task list with checkboxes
- `MEMORY.md` — append-only long-term memory
- `specs/<session>/` — spec pipeline artifacts
- `index.merkle` — cached Merkle tree for incremental sync

## Supported Languages (Indexing)

| Language | Extensions | Parser |
|----------|-----------|--------|
| Rust | `.rs` | tree-sitter-rust |
| Python | `.py`, `.pyi` | tree-sitter-python |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` | tree-sitter-typescript |

## Security Model

Every file operation goes through `PathJail` — a canonicalized root boundary that:
- Rejects `..` path components
- Canonicalizes deepest existing ancestors for new files
- Detects symlink escapes

Network access is restricted by `NetGuard`:
- HTTPS only
- Domain allowlist with suffix matching
- DNS resolution checked against RFC1918, loopback, link-local, cloud metadata IPs
- Manual redirect following (max 3 hops, re-checked per hop)

Policy hooks run before every tool invocation:
- `SecretLeakHook` — blocks AWS keys, API tokens, private key material
- `DestructiveCommandHook` — blocks `rm -rf /`, `mkfs`, forced pushes
- `SchemaLangGuard` — ensures JSON keys and machine fields stay ASCII

## Development

```bash
# Run all 128 tests
cargo test --workspace

# Check compilation
cargo check --workspace

# Run a specific crate's tests
cargo test -p indexer
cargo test -p sandbox
cargo test -p harness
```

### Requirements

- Rust 1.75+ (uses edition 2021)
- C compiler (for tree-sitter grammar compilation)
- Windows / Linux / macOS

## Crate Dependency Layers

| Layer | Crates | Purpose |
|-------|--------|---------|
| L0 | agent-types | Zero-dep shared types |
| L1 | runtime-core, llm-client, compaction | Async runtime, LLM, context management |
| L2 | indexer, vecstore, lsp-client, apply-engine, sandbox | Code understanding + security |
| L3 | harness, spec-pipeline, state-store, mcp | Policy, workflow, state, interop |
| L4 | agent-core | Orchestrator wiring everything together |
| L5 | evals, cli | Binary + evaluation harness |

## License

MIT
