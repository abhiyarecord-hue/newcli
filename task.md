# SPECIFICATION-DRIVEN TASK LIST

> Execution contract for the implementing model: read ONE task + the referenced `plan.md` sections, then write complete, compiling Rust. Do not read ahead. Do not refactor previous tasks. All signatures come from `plan.md` section 3 and are **non-negotiable**.

---

## PHASE 0 — Foundation

### TASK-0.1: Workspace Scaffold & Shared Types
- **Target Context:** `Cargo.toml` (workspace root), `crates/agent-types/src/*`
- **Dependencies:** none
- **Implementation Specifications:** Create the full workspace exactly as in `plan.md` section 1. Crates: `thiserror = "1"`, `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`, `async-trait = "0.1"`, `tokio-util = "0.7"`. Implement `AgentError`, `Result`, `Message`, `ContentBlock`, `Role`, `ToolSchema`, `Tool`, `ToolCtx`, `LanguageMode` verbatim from `plan.md` section 3.
- **Expected Input & Output:** `cargo check --workspace` passes; every type derives `Debug` + `Clone` where shown; `serde` round-trip test for `Message`.
- **Context Guard for Smaller LLMs:** Do NOT add `tokio` as a dependency of `agent-types` (only `tokio-util` for `CancellationToken`). Keep this crate dependency-light — every other crate transitively pays for what you add here.

### TASK-0.2: Structured-Concurrency Runtime Core
- **Target Context:** `crates/runtime-core/src/{scheduler.rs, task_scope.rs, event_bus.rs}`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Crates: `tokio = { features = ["rt-multi-thread", "sync", "time", "macros"] }`, `tokio-util`. Implement `TaskScope { token: CancellationToken, handles: Vec<JoinHandle<Result<()>>> }` with `spawn(fut)` (wraps future to abort on child token) and `async fn join_all(self) -> Result<()>` (cancels remaining children on first error). `EventBus` = thin wrapper over `tokio::sync::broadcast::Sender<AgentEvent>` where `AgentEvent { TurnStarted, ToolInvoked{name}, ToolCompleted{name}, TurnEnded }` lives in `agent-types/src/event.rs`.
- **Expected Input & Output:** `TaskScope::spawn<F: Future<Output=Result<()>> + Send + 'static>(&mut self, f: F)`; test proving parent-cancel propagates to children within 100ms (no orphaned futures).
- **Context Guard for Smaller LLMs:** Use `token.child_token()` per spawn — never clone the parent token into children (breaks hierarchical cancel). `JoinHandle::abort()` does not run async `Drop` cleanup; select on the token *inside* the wrapped future instead of relying on abort.

---

## PHASE 1 — LLM Interface & Context

### TASK-1.1: SSE Streaming Parser
- **Target Context:** `crates/llm-client/src/sse.rs`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Pure incremental parser: `struct SseParser { buf: Vec<u8> }` with `fn feed(&mut self, bytes: &[u8]) -> Vec<RawSseFrame>` where `RawSseFrame { event: Option<String>, data: String }`. Split on `\n\n`, handle `event:`/`data:` lines, multi-line `data:` joined with `\n`. No network code here.
- **Expected Input & Output:** feeding `b"event: delta\ndata: {\"x\":1}\n\n"` yields exactly one frame. Must handle a frame split across two `feed` calls.
- **Context Guard for Smaller LLMs:** Frames arrive fragmented across TCP reads — NEVER assume `feed` receives whole frames. Retain the unterminated tail in `buf`. Normalize `\r\n` to `\n` before splitting. Cap `buf` at 1 MiB and return `AgentError::Llm` above it (malicious-server DoS guard).

### TASK-1.2: LlmProvider for Anthropic-Style APIs
- **Target Context:** `crates/llm-client/src/{provider.rs, anthropic.rs}`
- **Dependencies:** TASK-1.1
- **Implementation Specifications:** Crates: `reqwest = { features = ["stream", "rustls-tls"], default-features = false }`, `futures-util`. Implement `LlmProvider::stream` from `plan.md` section 3: POST with `stream: true`, pipe `bytes_stream()` into `SseParser`, map frames (`content_block_delta` → `SseEvent::Delta`, `content_block_start` with `tool_use` → accumulate JSON, `message_stop` → `Stop`). Send events through `mpsc::channel(64)`. API key passed to the constructor — never read env vars inside the lib.
- **Expected Input & Output:** signature exactly as `plan.md` section 3; the spawned reader task exits when `cancel` fires or the receiver drops.
- **Context Guard for Smaller LLMs:** Tool-use input JSON arrives as *partial fragments* (`input_json_delta`) — accumulate the raw string and `serde_json::from_str` only at `content_block_stop`; parsing early produces wrong code. If `send().await` fails (receiver dropped), break the loop — do not `unwrap`.

### TASK-1.3: Session Compaction Engine
- **Target Context:** `crates/compaction/src/{config.rs, compactor.rs, token_count.rs}`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Implement `CompactionConfig` + `Compactor` trait from `plan.md` section 3. `token_count.rs`: heuristic `fn estimate_tokens(s: &str) -> u32` (`(chars/4).max(words)` — deterministic, no tokenizer dep). `ThresholdCompactor`: when total tokens > `max_context_tokens`, keep last `keep_recent_messages` (default 4) verbatim; fold the older prefix into a structured summary string (`"[COMPACTED] decisions: ...; files touched: ...; open tasks: ..."`) appended to the system prompt.
- **Expected Input & Output:** `compact(&cfg, &history) -> (String, Vec<Message>)`, pure function. Property test: retained token estimate <= `max_context_tokens`.
- **Context Guard for Smaller LLMs:** Never split a `ToolUse`/`ToolResult` pair across the compaction boundary — the API rejects orphaned tool results. If message N-4 is a `ToolResult`, extend retention to include its matching `ToolUse`.

---

## PHASE 2 — Semantic Indexing

### TASK-2.1: Tree-sitter Parser & Entity Extraction
- **Target Context:** `crates/indexer/src/{parser.rs, entity.rs, scope_tree.rs}`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Crates: `tree-sitter = "0.22"`, `tree-sitter-rust`, `tree-sitter-python`, `tree-sitter-typescript`. `Language` enum + `fn parse(path: &Path, source: &str) -> Result<Vec<CodeEntity>>` producing `CodeEntity` per `plan.md` section 3: signature = source slice up to body-open brace/colon, docstring = preceding comment nodes, `parent_id` from the enclosing named scope (scope tree). Use tree-sitter *queries* (`.scm` strings) per language, not manual node-kind matching where a query suffices.
- **Expected Input & Output:** parsing a Rust file with `impl Foo { fn bar() {} }` yields `bar` with `kind: Method`, `qualified_name: "Foo::bar"`, `parent_id = Some(id_of_Foo)`.
- **Context Guard for Smaller LLMs:** Deep AST recursion overflows the stack on minified/generated files — use an **iterative traversal with `TreeCursor`** (`goto_first_child`/`goto_next_sibling`/`goto_parent`), never recursive functions. Tree-sitter byte ranges are byte offsets, not char offsets — slicing `&source[a..b]` panics on non-UTF-8 boundaries; use `source.get(a..b).ok_or(...)`. Keep the `Tree` alive as long as any `Node` borrowed from it (lifetime tie).

### TASK-2.2: Greedy Recursive AST Chunker
- **Target Context:** `crates/indexer/src/chunker.rs`
- **Dependencies:** TASK-2.1, TASK-1.3 (reuse `estimate_tokens`)
- **Implementation Specifications:** `fn chunk(entities: &[CodeEntity], source: &str, budget: u32, overlap_nodes: usize) -> Vec<Chunk>`. Greedy algorithm: walk top-level entities in byte order; if entity <= budget, pack into current window; if entity > budget, recurse into its children (iteratively, via explicit `Vec` stack); leaf larger than budget → split at line boundaries as last resort. Prepend the last `overlap_nodes` entities of the previous chunk to the next chunk's text.
- **Expected Input & Output:** every `Chunk.token_count <= budget` (except unsplittable single lines); chunks never cut a function mid-body when the function fits the budget.
- **Context Guard for Smaller LLMs:** Use an explicit stack (`Vec<u64>` of entity ids) for the recursion — same stack-overflow hazard as TASK-2.1. Overlap must not count against the budget of *new* content or chunks shrink to overlap-only and the loop never advances (infinite loop). Guard: assert forward progress of the byte cursor each iteration.

### TASK-2.3: Merkle Tree Incremental Sync
- **Target Context:** `crates/indexer/src/merkle.rs`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Crates: `sha2`, `walkdir`, `ignore` (respect `.gitignore`). `struct MerkleTree { root_hash: [u8; 32], nodes: HashMap<PathBuf, [u8; 32]> }`. File node = SHA-256(content); dir node = SHA-256(sorted child (name, hash) pairs). `fn diff(&self, other: &MerkleTree) -> Vec<PathBuf>` returns changed *files* only, pruning descent into subtrees with equal dir hashes.
- **Expected Input & Output:** `build(root: &Path) -> Result<MerkleTree>`; editing 1 file in a 50k-file tree makes `diff` return exactly that path; serialize with `serde` to `.agent/index.merkle`.
- **Context Guard for Smaller LLMs:** Sort children **byte-wise by file name** before hashing dirs — HashMap iteration order is nondeterministic and will produce a different root hash every run. Skip symlinks entirely (`follow_links(false)`) — cycles hang the scan. Hash file *content*, using mtime only as a cheap pre-filter to skip rehashing.

---

## PHASE 3 — Local Vector Store & Retrieval

### TASK-3.1: SQLite Schema with sqlite-vec + FTS5
- **Target Context:** `crates/vecstore/src/{schema.rs, store.rs}`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Crates: `rusqlite = { features = ["bundled"] }`, `sqlite-vec` (loadable extension via `sqlite_vec::sqlite3_vec_init`), `zerocopy` for vector byte-casting. Tables exactly per research doc: `files(path PK, mtime, content_hash)`, `chunks(id PK, file_path, start_line, end_line, text, token_count)`, `chunks_vec` = `vec0(embedding FLOAT[768])` virtual table keyed by chunk id, `chunks_fts` = `fts5(text, content='chunks', content_rowid='id')`. `VecStore::open(path) -> Result<Self>` runs migrations idempotently (`user_version` pragma). API: `upsert_file`, `insert_chunks(Vec<(Chunk, Vec<f32>)>)`, `delete_file(path)` (cascade to all three chunk tables).
- **Expected Input & Output:** insert 3 chunks with embeddings; `SELECT count(*)` agrees across `chunks`, `chunks_vec`, `chunks_fts`.
- **Context Guard for Smaller LLMs:** `rusqlite::Connection` is `!Sync` — run DB work in `tokio::task::spawn_blocking` behind a dedicated blocking thread or `std::sync::Mutex<Connection>`; do NOT hold the connection across `.await` points. Embeddings go into sqlite-vec as **little-endian f32 byte slices** — use `zerocopy`/`bytemuck` casting, never `serde_json`. FTS5 external-content tables need explicit delete triggers or `delete_file` leaves ghost rows that still match searches.

### TASK-3.2: Hybrid 3-Mode Search (Vector + BM25 + Graph)
- **Target Context:** `crates/vecstore/src/{hybrid.rs, graph.rs}`
- **Dependencies:** TASK-3.1
- **Implementation Specifications:** Implement `Retriever` from `plan.md` section 3. Vector: `... WHERE embedding MATCH ? AND k = ?` KNN on `chunks_vec` (cosine). Keyword: FTS5 `bm25()` ranking. Graph: `entity_edges(src_id, dst_id, kind)` table + recursive CTE walking <= 2 hops from seed entities matched by name. `SearchMode::Hybrid`: run all three, merge with Reciprocal Rank Fusion `score = sum(1/(60 + rank_i))`, dedupe by `chunk_id`, return top-k.
- **Expected Input & Output:** `search(query, mode, k) -> Result<Vec<SearchHit>>` sorted by score desc; hybrid returns a hit found by only one mode (union, not intersection).
- **Context Guard for Smaller LLMs:** FTS5 chokes on raw user queries containing `"` `*` `(` — sanitize by quoting each term: `"foo" "bar"`. BM25 in SQLite returns **negative-is-better** values; normalize before fusing or hybrid ranking silently inverts. Bound the recursive CTE (`LIMIT` + depth column) — unbounded recursion on a cyclic call graph never terminates.

---

## PHASE 4 — LSP Integration

### TASK-4.1: LSP Client Lifecycle
- **Target Context:** `crates/lsp-client/src/{client.rs, lifecycle.rs}`
- **Dependencies:** TASK-0.2
- **Implementation Specifications:** Crates: `tower-lsp = "0.20"`, `lsp-types`, `tokio::process`. Spawn the server (e.g., `rust-analyzer`) as a child process over stdio. Implement framing (`Content-Length: N\r\n\r\n{json}`), `initialize` → `initialized` handshake with `rootUri = project_root`, and a pending-request map `HashMap<i64, oneshot::Sender<Value>>` keyed by JSON-RPC id. `LspClient::start(server_cmd, root) -> Result<Self>` and `async fn shutdown(self)` (`shutdown` request then `exit` notification).
- **Expected Input & Output:** `start` resolves after the server's `InitializeResult`; requests time out with `AgentError::Lsp` after 10s.
- **Context Guard for Smaller LLMs:** `Content-Length` counts **bytes, not chars**. Read stdout with one dedicated task; never interleave reads. Server-initiated requests (e.g., `workspace/configuration`) MUST be answered (even with `null` result) or rust-analyzer stalls the pipeline. Kill the child process on `Drop` — orphaned rust-analyzer processes eat GBs of RAM.

### TASK-4.2: Diagnostics, Definitions & References
- **Target Context:** `crates/lsp-client/src/{diagnostics.rs, symbols.rs}`
- **Dependencies:** TASK-4.1
- **Implementation Specifications:** Send `textDocument/didOpen` / `didChange` (full-sync mode) whenever the agent reads/edits a file. Buffer `textDocument/publishDiagnostics` notifications into `DashMap<Url, Vec<Diagnostic>>`. API: `async fn goto_definition(&self, file, line, col) -> Result<Vec<Location>>`, `async fn find_references(...) -> Result<Vec<Location>>`, `async fn diagnostics_for(&self, file: &Path, settle: Duration) -> Result<Vec<Diagnostic>>` (waits until no new diagnostics arrive for the `settle` window).
- **Expected Input & Output:** editing a file to introduce a type error surfaces >= 1 `Error`-severity diagnostic via `diagnostics_for` within 5s.
- **Context Guard for Smaller LLMs:** LSP positions are **0-based line + UTF-16 code-unit column** — convert from byte offsets or all positions are wrong on non-ASCII lines. Diagnostics are asynchronous and versioned: track `didChange` version numbers and discard diagnostics for stale versions, or the agent will "fix" already-fixed errors.

---

## PHASE 5 — Edit Application

### TASK-5.1: Lazy Edit Snippet & Fast Apply Bridge
- **Target Context:** `crates/apply-engine/src/{lazy_edit.rs, fast_apply.rs}`
- **Dependencies:** TASK-1.2
- **Implementation Specifications:** `lazy_edit.rs`: validate snippets — must contain >= 1 marker matching regex `(?m)^\s*(//|#|--|/\*)\s*\.\.\.\s*existing code\s*\.\.\.` ; reject empty snippets. `fast_apply.rs`: `FastApplyStrategy` implementing `ApplyStrategy` (`plan.md` section 3) that POSTs `{original, snippet}` to a configurable SLM endpoint (Morph-style Fast Apply API) via `reqwest` and returns merged content. Also implement `FallbackStrategy`: deterministic marker-based splice (anchor lines around markers matched against original with exact-then-trimmed matching) for offline mode. `ApplyEngine` tries fast-apply first, falls back on error.
- **Expected Input & Output:** `apply(original, &LazyEdit) -> Result<String>`; merged output preserves every original line not covered by the snippet; result must parse under TASK-2.1's parser (syntax gate).
- **Context Guard for Smaller LLMs:** Never trust the SLM blindly — if merged output loses > 40% of original lines while the snippet didn't indicate deletion, return `AgentError::Tool` instead of writing. Anchor matching must compare with `trim_end()` (trailing-whitespace brittleness is the #1 diff-format failure named in the research doc). Write to disk atomically: temp file in the same directory + `rename` (cross-device rename fails — same dir matters).

### TASK-5.2: Semantic Diff
- **Target Context:** `crates/apply-engine/src/semantic_diff.rs`
- **Dependencies:** TASK-2.1, TASK-5.1
- **Implementation Specifications:** `fn semantic_diff(old_src: &str, new_src: &str, lang: Language) -> Result<Vec<EntityChange>>` where `EntityChange { kind: Added | Removed | Modified, entity: CodeEntity }`. Parse both versions with `indexer::parser`, key entities by `qualified_name + kind`, compare body byte-slices by hash. Output feeds back to the planner LLM ("function `Foo::bar` modified") instead of line noise.
- **Expected Input & Output:** renaming one function among 50 yields exactly `Removed(old)` + `Added(new)`, nothing else.
- **Context Guard for Smaller LLMs:** Compare entity *bodies* after normalizing whitespace-only changes, or every reformat drowns the diff. Byte ranges differ between versions by construction — never compare `byte_range` for equality; compare content hashes.

### TASK-5.3: CRDT Concurrent Document & IPC Patches
- **Target Context:** `crates/apply-engine/src/{crdt_doc.rs, ipc.rs}`
- **Dependencies:** TASK-5.1
- **Implementation Specifications:** Crates: `loro` (or `cola`) for text CRDT, `tokio::net::UnixListener`, `serde_json`. `CrdtDoc` wraps one file: agent edits and user edits (received over the socket) both apply as CRDT operations; convergent state resolves the "Externally Modified" conflict. `ipc.rs`: newline-delimited JSON over Unix domain socket at `.agent/editor.sock`; outbound message `{"file":..., "patches":[{"range":[a,b],"insert":"..."}], "version": n}` so editor plugins preserve cursor and undo history.
- **Expected Input & Output:** interleaving one agent edit and one user edit on different lines converges to a state containing both changes; socket protocol round-trip test with a mock client.
- **Context Guard for Smaller LLMs:** CRDT positions are internal IDs, not byte offsets — always translate through the library's cursor API when emitting JSON patches; hand-computed offsets desync after the first concurrent edit. Unlink a stale socket file on startup (`bind` fails with `AddrInUse` otherwise). One `CrdtDoc` per file behind `Arc<Mutex<...>>` — async state isolation: never share a doc across files.

---

## PHASE 6 — Security

### TASK-6.1: Path Jail
- **Target Context:** `crates/sandbox/src/path_jail.rs`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Implement `PathJail` per `plan.md` section 3. `new`: `std::fs::canonicalize` the root. `resolve`: join `user_path` to root, then canonicalize the **deepest existing ancestor** and verify `starts_with(root)`; re-append non-existing tail components after checking each for `..`. Reject absolute inputs that don't already start with root.
- **Expected Input & Output:** `resolve("src/main.rs")` → ok; `resolve("../../etc/passwd")` → `AgentError::PathJail`; a symlink inside the jail pointing to `/etc` → `PathJail` error.
- **Context Guard for Smaller LLMs:** `canonicalize` fails on paths that don't exist yet (new files!) — that is why you canonicalize the deepest existing ancestor, not the full path. Checking for `..` as a substring is wrong (`..foo` is a legal filename); check `Component::ParentDir` components. This function is the security boundary for EVERY file tool — no fast paths, no caching.

### TASK-6.2: MicroVM Sandbox Executor
- **Target Context:** `crates/sandbox/src/{microvm.rs, snapshot.rs}`
- **Dependencies:** TASK-0.2, TASK-6.1
- **Implementation Specifications:** Implement `SandboxExecutor` per `plan.md` section 3 with two backends behind a trait object: (a) `HyperlightVm` (feature `hyperlight`, Linux/KVM + macOS/HVF) embedding the `hyperlight-host` crate — guest runs command, host reads sandboxed stdout/stderr; `reset()` restores the memory snapshot for millisecond-fresh VMs; (b) `ProcessFallback` (default feature) using `tokio::process::Command` with cleared env, cwd = jail root, hard timeout — clearly documented as dev-only. Enforce: network disabled in guest, wall-clock timeout via `tokio::time::timeout`, output capped at 1 MiB.
- **Expected Input & Output:** `execute("echo hi", 5s, &token)` → `ExecResult { stdout: "hi\n", exit_code: 0, .. }`; a command sleeping past timeout returns `AgentError::Sandbox("timeout")` and the VM/process is dead afterwards.
- **Context Guard for Smaller LLMs:** On timeout you MUST kill the child (`start_kill()` + reap) — `tokio::time::timeout` dropping the future does NOT kill the OS process. Read stdout and stderr **concurrently** (two tasks or `try_join!`) — sequential reads deadlock when the child fills the un-drained pipe's 64 KiB buffer. Cancellation token firing takes the same kill path as timeout.

### TASK-6.3: SSRF-Filtered Network Guard
- **Target Context:** `crates/sandbox/src/net_guard.rs`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Crates: `reqwest`, `url`, `ipnet`. `NetGuard::new(allowed_domains: Vec<String>)` exposing `async fn get(&self, url: &str) -> Result<String>`. Pipeline: parse URL → scheme must be `https` → host must suffix-match allowlist → resolve DNS → **reject if any resolved IP** is loopback, private (RFC1918), link-local `169.254.0.0/16` (cloud metadata), or unique-local IPv6 → connect to the vetted IP via `reqwest` `resolve()` override (pins IP, defeats DNS rebinding). Disable redirects (`Policy::none()`); follow manually, re-running the full check per hop, max 3 hops.
- **Expected Input & Output:** `get("https://crates.io/...")` with `crates.io` allowlisted → ok; `get("http://169.254.169.254/latest/meta-data")` → `AgentError::Sandbox`.
- **Context Guard for Smaller LLMs:** Check IPs **after DNS resolution** and pin the connection to the checked IP — hostname-only checks are defeated by DNS rebinding. Automatic redirect following bypasses your checks; that is why redirects are manual. Also reject IPv4-mapped IPv6 (`::ffff:169.254.x.x`).

---

## PHASE 7 — Harness & Policy

### TASK-7.1: Hook Engine
- **Target Context:** `crates/harness/src/{hooks.rs, policy.rs}`
- **Dependencies:** TASK-0.1
- **Implementation Specifications:** Implement `Hook`, `HookVerdict`, `HookPoint` per `plan.md` section 3 plus `HookEngine { hooks: Vec<Arc<dyn Hook>> }` with `async fn run(&self, point, tool, payload) -> Result<HookVerdict>` (first `Deny` wins; `Rewrite`s compose left-to-right). Ship two built-ins: `SecretLeakHook` (PreTool on `bash`/`write_file`: regex scan for AWS keys `AKIA[0-9A-Z]{16}`, generic `(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}`, private key headers) and `DestructiveCommandHook` (deny `rm -rf /`, `mkfs`, `dd of=/dev/`, forced pushes to protected branches).
- **Expected Input & Output:** payload `{"command": "curl -H 'x-api-key: AKIA...'"}` at `PreTool` → `Deny` with a reason naming the pattern class (never echoing the secret itself).
- **Context Guard for Smaller LLMs:** Hooks are the system's **determinism layer** — no LLM calls, no network, no filesystem writes inside a hook. A hook that errors must be treated as `Deny` (fail-closed), never `Allow`.

### TASK-7.2: Skills Registry
- **Target Context:** `crates/harness/src/skills.rs`
- **Dependencies:** TASK-7.1
- **Implementation Specifications:** Implement `Skill` per `plan.md` section 3 + `SkillRegistry::activate(user_msg: &str, lang: LanguageMode) -> Vec<&dyn Skill>`. Skills load from `.agent/skills/*.toml` (`serde` + `toml` crate, `#[serde(deny_unknown_fields)]`): fields `name`, `triggers: Vec<String>` (case-insensitive substring or `regex:`-prefixed), `prompt: String`. Ship built-in `code-review` skill injecting an evaluation rubric when the message mentions review. Also ship built-in `hinglish-mode` skill: it is **config-driven, not trigger-driven** — it activates on every turn iff `lang == LanguageMode::Hinglish`, ignoring message content. Its `system_prompt_fragment()` (1) instructs the model to reason, plan, and explain concepts **exclusively in Hinglish** (Hindi written in the English/Latin alphabet — never Devanagari script), and (2) embeds the LANG-GUARD RULE from `plan.md` section 5 verbatim: all code blocks, variable/function/type names, tool calls, JSON keys and schemas, file paths, and shell commands remain strictly English/ASCII.
- **Expected Input & Output:** message "please do a code review of src/" activates `code-review`; its `system_prompt_fragment()` is appended to the system prompt for that turn only. With `lang = Hinglish`, every turn's system prompt contains both the Hinglish directive and the LANG-GUARD RULE; with `lang = En`, `hinglish-mode` never activates.
- **Context Guard for Smaller LLMs:** Compile regexes once at load and surface bad patterns as config errors at startup — compiling per-message is the classic hidden O(n) trap. Skill prompts are additive and turn-scoped: never persist them into compacted history. The hinglish fragment is prompt-level *guidance only* — deterministic enforcement lives in TASK-11.1; never translate skill trigger keywords, tool names, or schema fields into Hinglish.

### TASK-7.3: Sub-agent Spawner
- **Target Context:** `crates/harness/src/subagent.rs`
- **Dependencies:** TASK-0.2, TASK-1.2
- **Implementation Specifications:** Implement `SubAgent` per `plan.md` section 3 + `ExplorerSubAgent` running a bounded mini-loop (own fresh `Vec<Message>`, max 10 turns, read-only tool set) against an injected `Arc<dyn LlmProvider>`. `SubAgentPool::run_parallel(tasks: Vec<String>, max_concurrent: usize) -> Result<Vec<String>>` using `TaskScope` from TASK-0.2 + `tokio::sync::Semaphore`. Only the final summary string (<= 2000 chars, truncate with marker) returns to the parent context.
- **Expected Input & Output:** three exploration tasks with `max_concurrent = 2` never exceed 2 in-flight LLM streams (assert via test counter); parent cancel kills all sub-agents.
- **Context Guard for Smaller LLMs:** **Async state isolation:** a sub-agent must never receive the parent's message history or mutate shared state — its whole purpose is keeping the main context window clean. Fresh history, fresh child cancel token, summary-only return. Do not let a sub-agent spawn sub-agents (depth limit 1, enforced in code).

---

## PHASE 8 — Spec Pipeline, State, MCP

### TASK-8.1: RustySpec 7-Stage Pipeline
- **Target Context:** `crates/spec-pipeline/src/{stages.rs, artifacts.rs}`
- **Dependencies:** TASK-1.2, TASK-6.1
- **Implementation Specifications:** `enum Stage { Specify, Clarify, Plan, Tasks, Tests, Implement, Analyze }` with `fn artifact(&self) -> &'static str` mapping to `spec.md`, `clarifications.md`, `plan.md`, `tasks.md`, `tests/`, code, `analysis.md`. `Pipeline::run_stage(stage, ctx) -> Result<PathBuf>`: builds a stage-specific prompt (embedding prior artifacts read via `PathJail`), calls the LLM, writes the artifact atomically under `.agent/specs/<session>/`. Stages enforce ordering: `run_stage(Plan)` errors if `spec.md` is missing.
- **Expected Input & Output:** `run_stage(Specify, ctx)` produces `spec.md` containing `## User Stories` and `## Functional Requirements` headers (validate before writing; retry once with a corrective prompt).
- **Context Guard for Smaller LLMs:** The pipeline is a *state machine* — encode stage order as data (`Stage::prerequisites() -> &[Stage]`) and check artifact existence on disk; do not trust in-memory flags across process restarts. All file writes go through `PathJail::resolve`.

### TASK-8.2: Persistent & Ephemeral State Store
- **Target Context:** `crates/state-store/src/{persistent.rs, ephemeral.rs}`
- **Dependencies:** TASK-6.1
- **Implementation Specifications:** `PersistentState` maps to on-disk markdown: `SOUL.md` (persona + policies), `HEARTBEAT.md` (task list, checkbox syntax `- [ ]`), `MEMORY.md` (append-only long-term memory with ISO-8601 timestamps). API: `load(root) -> Result<Self>`, `append_memory(&self, entry: &str)`, `heartbeat_tasks() -> Vec<(bool, String)>`, `set_task_done(idx)`. Language persona policy: `SOUL.md` carries a `## Policies` section with a `language: hinglish` (or `language: en`) line; API: `language_mode() -> LanguageMode` (defaults to `En` when the line is absent) and `set_language_mode(LanguageMode)` (surgical single-line edit, same discipline as heartbeat checkboxes) — this is what makes the Hinglish preference survive across sessions. `EphemeralState`: in-memory `VecDeque<PendingTask>` + ring buffer of last 500 tool outputs; `wipe()` clears everything after task completion. No SQLite here — human-readable files are the contract.
- **Expected Input & Output:** round-trip: `set_task_done(0)` flips `- [ ]` to `- [x]` for the first task and preserves every other byte of `HEARTBEAT.md`. `set_language_mode(Hinglish)` followed by `load` returns `language_mode() == Hinglish` while preserving every other byte of `SOUL.md`.
- **Context Guard for Smaller LLMs:** Edit `HEARTBEAT.md` and `SOUL.md` surgically (line-targeted), never regenerate the whole file from a parsed model — users hand-edit these files and full rewrites destroy their formatting/comments. `append_memory` opens with `OpenOptions::append(true)`; read-modify-write races with concurrent sub-agents. Precedence contract (see `plan.md` section 5): explicit `config.toml` value wins over `SOUL.md` and is written through; parse unknown `language:` values as a startup config error, never silently as `En`.

### TASK-8.3: MCP Client & Server
- **Target Context:** `crates/mcp/src/{client.rs, server.rs, schema.rs}`
- **Dependencies:** TASK-0.1, TASK-6.3
- **Implementation Specifications:** JSON-RPC 2.0 over stdio (newline-delimited). `schema.rs`: strongly-typed `serde` structs for `initialize`, `tools/list`, `tools/call` (`#[serde(deny_unknown_fields)]` on configs, tagged enums for messages). Client: `McpClient::connect(cmd: &str) -> Result<Self>` spawning an external MCP server (GitHub/Slack/JIRA bridges) and exposing its tools as `Vec<ToolSchema>` + a `Tool` adapter struct so they plug into the normal dispatcher. Server: expose this agent's own tools (`search`, `read_file`) to external MCP hosts. HTTP transports go through `NetGuard` (TASK-6.3).
- **Expected Input & Output:** connecting to a mock MCP server yields its tool list; invoking a remote tool round-trips a `tools/call` and returns text content.
- **Context Guard for Smaller LLMs:** Remote tool schemas are **untrusted input** — validate `input_schema` is a JSON object, cap description length (4 KiB), and namespace remote tool names (`mcp__github__create_issue`) so a malicious server cannot shadow built-in tools like `bash`. JSON-RPC ids must be matched, not assumed sequential.

---

## PHASE 9 — Orchestration & CLI

### TASK-9.1: Tool Dispatcher & Built-in Tools
- **Target Context:** `crates/agent-core/src/tools/`
- **Dependencies:** TASK-3.2, TASK-4.2, TASK-5.1, TASK-6.1, TASK-6.2, TASK-7.1
- **Implementation Specifications:** `ToolDispatcher { tools: HashMap<String, Arc<dyn Tool>>, hooks: HookEngine }`. Built-ins (each in its own file, each implementing `Tool` from `plan.md` section 3): `read_file`, `write_file` (via `ApplyEngine`), `bash` (via `SandboxExecutor`), `search_code` (via `Retriever`), `goto_definition`, `find_references`, `diagnostics` (via `LspClient`). Every path argument resolves through `PathJail`; every invocation wraps in `PreTool`/`PostTool` hooks; tool output truncated to 30k chars with `[truncated]` marker.
- **Expected Input & Output:** `dispatch(ToolCall { name, input }) -> ContentBlock::ToolResult`; unknown tool → result with `is_error: true` (not a process error — the LLM must see it and recover).
- **Context Guard for Smaller LLMs:** Tool failures are **data, not crashes**: convert every `AgentError` into `ToolResult { is_error: true }` so the loop continues. Never let raw absolute host paths leak into tool output shown to the model (jail-relative paths only) — it enables path-jail probing.

### TASK-9.2: Turn Orchestrator (the Agentic Loop)
- **Target Context:** `crates/agent-core/src/{orchestrator.rs, turn.rs}`
- **Dependencies:** TASK-9.1, TASK-1.3, TASK-7.2, TASK-7.3, TASK-8.2
- **Implementation Specifications:** Implement `plan.md` section 2.1 literally: `Orchestrator::run_turn(&mut self, user_msg: String) -> Result<String>`: (1) activate skills, (2) compact if needed, (3) stream LLM, (4) on `SseEvent::ToolUse` run hooks → dispatch → append `ToolResult` → re-invoke LLM, (5) loop until `StopReason::EndTurn`, hard cap 50 iterations (`AgentError::Tool` on breach). Emit `AgentEvent`s on the `EventBus` at each step. All state lives on `self` — no globals, no `static`.
- **Expected Input & Output:** with a mock provider scripting [ToolUse(read_file), EndTurn], `run_turn` returns the final text and history contains exactly: user, assistant(tool_use), tool(result), assistant(text).
- **Context Guard for Smaller LLMs:** Message alternation is strict for LLM APIs: a `ToolUse` block MUST be immediately followed by a message containing its `ToolResult` with a matching `tool_use_id` — mismatched ids are the most common loop-corruption bug. The iteration cap is a safety net, not a feature: log loudly when hit.

### TASK-9.3: CLI Entrypoint
- **Target Context:** `bin/cli/src/main.rs`
- **Dependencies:** TASK-9.2, TASK-8.1, TASK-2.3, TASK-3.1
- **Implementation Specifications:** Crates: `clap = { features = ["derive"] }`, `tokio` (main), `tracing` + `tracing-subscriber`. Subcommands: `chat` (interactive loop, streams deltas to stdout as they arrive), `index` (Merkle diff → parse → chunk → embed → store; prints changed-file count and elapsed time), `spec <stage>` (runs a RustySpec stage), `search <query>` (hybrid search, prints hits). Config from `.agent/config.toml` (provider, model, api-key env-var *name*, allowed domains, `language = "hinglish" | "en"` — default `"en"`, deserializing to `LanguageMode`, `#[serde(deny_unknown_fields)]` rejects typos like `"hindi"`). Startup wiring: resolve `LanguageMode` per the precedence contract (config wins, written through to `SOUL.md` via TASK-8.2; `SOUL.md` decides when config omits the field) and pass it into `SkillRegistry::activate` every turn. Ctrl-C cancels the root `CancellationToken` → graceful drain via `TaskScope`.
- **Expected Input & Output:** `cargo run -p cli -- index` on this repo completes and prints stats; a second run prints `0 files changed` (Merkle short-circuit proof); `--help` documents all subcommands.
- **Context Guard for Smaller LLMs:** Print streaming deltas with `write!` + explicit `flush()` — stdout line-buffering makes streaming look frozen. Read the API key from the env var *named in config*, never store the key itself in config. First Ctrl-C cancels gracefully; second Ctrl-C hard-exits.

---

## PHASE 10 — Evaluation Harness (Evals)

> Scope: strictly automated agent benchmarks. Objective: measure raw task-completion quality and accuracy with hard data so comparisons against Kiro / Claude Code CLI are benchmark deltas, not opinions. Design per `plan.md` section 4.

### TASK-10.1: SWE-bench-lite Scripted Runner
- **Target Context:** `crates/evals/src/{swebench.rs, report.rs}`, `bin/cli/src/main.rs` (new `eval run` subcommand)
- **Dependencies:** TASK-9.2, TASK-9.3
- **Implementation Specifications:** Implement `EvalCase` and `EvalOutcome` per `plan.md` section 3. Loader reads `.agent/evals/cases/*.toml` (`#[serde(deny_unknown_fields)]`; bad case files are startup errors, never skipped silently). Runner: for each case — bounded by `--max-concurrent` via `TaskScope` + `Semaphore` (TASK-7.3 pattern) — (1) copy `repo_fixture` into a fresh temp dir, (2) construct a **fresh** `PathJail`, `Orchestrator`, and sandbox (zero state reuse across cases), (3) run the case `prompt` through `run_turn`, (4) execute `check_cmd` in the sandbox: exit code 0 = pass, (5) append one `EvalOutcome` line (JSONL) to `.agent/evals/results/<run-id>.jsonl` where `run-id` = UTC ISO-8601 timestamp. `cargo run -p cli -- eval run --suite swebench-lite --max-concurrent N` prints a pass-rate table (per-case + aggregate, turns, tool calls, tokens, wall time).
- **Expected Input & Output:** with a scripted mock provider and a trivial fixture, `eval run` deterministically reports 1/1 pass; a case whose `check_cmd` exits non-zero records `passed: false` and the suite still completes and reports.
- **Context Guard for Smaller LLMs:** Case failures are **data, not crashes** — never abort the suite on one failing case; record the outcome and continue. Enforce `timeout_secs` per case with the same kill-and-reap path as TASK-6.2 (`tokio::time::timeout` dropping a future does NOT kill the child process). Results files are append-only JSONL — one `serde_json` line per outcome, flushed per case, so a crashed run still leaves partial data.

### TASK-10.2: Regression Trajectory Tracking
- **Target Context:** `crates/evals/src/{trajectory.rs, report.rs}`, `bin/cli/src/main.rs` (`eval baseline`, `eval diff` subcommands)
- **Dependencies:** TASK-10.1
- **Implementation Specifications:** During each case, subscribe to the `EventBus` (TASK-0.2) and record the full trajectory — every `AgentEvent`, message, tool call, and tool result — as append-only JSONL at `.agent/evals/trajectories/<case-id>/<run-id>.jsonl`. `eval baseline set <run-id>` pins a baseline (path stored in `.agent/evals/baseline`). `eval diff <run-a> <run-b>` (or a single run vs. the pinned baseline) compares per-case: pass/fail, turns, tool calls, tokens in/out, wall time. **Hard regression** = any case flipping pass→fail; **soft regression** = turns +20% or tokens +30% (thresholds in `.agent/config.toml`). `eval diff` prints a per-case table and exits non-zero on any hard regression — this is the CI quality gate.
- **Expected Input & Output:** two scripted runs where one case flips pass→fail make `eval diff` list that case id and exit non-zero; identical runs diff clean and exit 0.
- **Context Guard for Smaller LLMs:** Redact trajectories through the `SecretLeakHook` regexes (TASK-7.1) **before** writing — trajectory files get committed and shared, and tool results can echo env output. Compare per-case, never aggregate-only — a 5% aggregate improvement can hide a pass→fail flip on a critical case. Trajectory files are append-only; never rewrite a recorded run.

---

## PHASE 11 — Localization Layer (Hinglish Mode)

> Scope: Hinglish (Hindi in the Latin/English alphabet) for the *prose channel only* — reasoning, plans, explanations. Pure Devanagari script is intentionally unsupported (token cost, terminal rendering). Machine surfaces stay English — always. Design per `plan.md` section 5. Prompt-side wiring lives in TASK-7.2 (skill), TASK-8.2 (SOUL.md persistence), TASK-9.3 (config); this phase adds the deterministic enforcement.

### TASK-11.1: Deterministic Schema & Code Language Guard
- **Target Context:** `crates/harness/src/lang_guard.rs`
- **Dependencies:** TASK-7.1
- **Implementation Specifications:** Implement `SchemaLangGuard` per `plan.md` section 3 as a `Hook` registered at `PreTool` (and reusable as a plain validator by TASK-11.2). `assert_english_machine_surface(v: &serde_json::Value) -> Result<()>` walks the value recursively: (1) every JSON **key** must match `^[A-Za-z_][A-Za-z0-9_-]*$`; (2) reject any codepoint in the Devanagari block (U+0900–U+097F) in keys, tool names, and in the *values* of machine-surface fields (`file_path`, `path`, `command`, `cmd`, `url`, and any field whose value is fed to a parser or the shell); (3) prose-valued fields (`explanation`, `description`, `reasoning`, `summary`) are exempt via an explicit allowlist — Hinglish is welcome there. Violations return `HookVerdict::Deny` with an English corrective reason instructing the model to re-emit the tool call with English/ASCII keys and paths.
- **Expected Input & Output:** tool input `{"file_path": "स्रोत.rs"}` → `Deny` naming the offending field; `{"file_path": "src/main.rs", "explanation": "yeh function file padhta hai"}` → `Allow`. A Devanagari JSON *key* anywhere → `Deny`.
- **Context Guard for Smaller LLMs:** This guard is the **determinism layer for localization** — no LLM calls, no I/O, and it fails closed like every hook (TASK-7.1 rule: a hook that errors is a `Deny`). The `Deny` reason becomes a `ToolResult { is_error: true }` the model can recover from — never a process crash; that is the whole point: parsers upstream never see mixed-language keys. Keep the prose-field allowlist explicit and small — defaulting unknown fields to "prose" reopens the hole.

### TASK-11.2: Spec Pipeline & Tool Schema Language Shield
- **Target Context:** `crates/spec-pipeline/src/stages.rs`, `crates/mcp/src/schema.rs`
- **Dependencies:** TASK-8.1, TASK-8.3, TASK-11.1
- **Implementation Specifications:** Shield every machine-parsed artifact from language mixing. Spec pipeline: (1) every stage prompt appends the artifact-structure rule — markdown headers, code fences, and any embedded JSON/TOML inside artifacts are strictly English even in hinglish mode; Hinglish is permitted only inside prose paragraphs; (2) post-generation, before the atomic write, validate the artifact: required English headers present (reuse the TASK-8.1 header check — single shared validator function, not a copy), every fenced code block and embedded JSON block passes `SchemaLangGuard::assert_english_machine_surface` (no Devanagari identifiers or keys); on violation, retry once with a corrective prompt (reuse TASK-8.1's retry machinery), then error. MCP: outgoing `tools/call` inputs and this agent's own exposed server tool schemas pass `assert_english_machine_surface` before serialization — a Hinglish key in a JSON-RPC payload must be impossible by construction.
- **Expected Input & Output:** in hinglish mode, `run_stage(Specify)` still produces `spec.md` with English `## User Stories` and `## Functional Requirements` headers (Hinglish body prose allowed); a stage output containing a Devanagari JSON key fails validation, triggers exactly one corrective retry, and errors if the retry also fails.
- **Context Guard for Smaller LLMs:** Validate **structure, not prose** — rejecting Hinglish sentences inside paragraphs breaks the entire feature; only headers, fences, keys, and identifiers are constrained. Extract code fences with a real fence parser (track opening/closing ``` pairs), not line-startswith heuristics — indented and nested fences are where naive scanners silently skip blocks.

---

## Execution Order Summary

```text
0.1 → 0.2 → {1.1 → 1.2, 1.3} → {2.1 → 2.2, 2.3} → 3.1 → 3.2 → 4.1 → 4.2
 → 5.1 → {5.2, 5.3} → 6.1 → {6.2, 6.3} → 7.1 → {7.2, 7.3} → {8.1, 8.2, 8.3}
 → 9.1 → 9.2 → 9.3 → {10.1 → 10.2, 11.1 → 11.2}
```
Tasks inside `{}` are parallelizable. Every task must end with `cargo check --workspace` green and its listed test passing before the next task begins.
