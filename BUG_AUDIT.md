# Repository Bug Audit

**Date:** 2026-07-14  
**Scope:** Entire Rust workspace: CLI and all 15 library crates (`agent-core`, `agent-types`, `apply-engine`, `compaction`, `evals`, `harness`, `indexer`, `llm-client`, `lsp-client`, `mcp`, `runtime-core`, `sandbox`, `spec-pipeline`, `state-store`, `vecstore`).  
**Method:** Full source inspection followed by a requirement-aware classification. Fixes are tracked separately below and are marked complete only after workspace build/test validation.

## Validation summary

- Initial audit baseline: `cargo check --workspace --all-targets` passed and `cargo test --workspace --lib` passed (154 tests).
- First fix batch (2026-07-14): `cargo check --workspace --all-targets` **passed**, with pre-existing warnings; `cargo test --workspace --lib` **passed: 154 passed, 0 failed**.
- Changed-file diagnostics and `git diff --check`: **passed**.
- `cargo fmt --all`: **not available** because `cargo-fmt.exe`/the `rustfmt` component is not installed for the active Windows toolchain.
- `cargo clippy --workspace --all-targets -- -D warnings`: **not run** because `cargo-clippy.exe` is not installed for the active Windows toolchain.
- Passing tests do not invalidate unresolved findings below; most are uncovered integration, malformed-input, concurrency, Windows, or failure-path cases.
- Live provider/MCP/LSP network interoperability was not tested because it requires external services and credentials.

## Fix progress

### Validated fixed in the first safety batch (8)

- **BUG-002:** Checkpoint creation now validates every Git command, records the exact successful checkpoint for the current turn, refuses stale/failed undo, avoids checkpoint commits on the user's branch, and creates a unique recoverable backup ref before reset.
- **BUG-003:** CRDT ranges now reject reversed, out-of-bounds, overlapping, and non-UTF-8-boundary patches; IPC rejects stale versions and reports structured errors.
- **BUG-005:** Existing-file read failures are fatal, while conflict backups are unique, exclusively created, and required to succeed before replacement.
- **BUG-006:** Compacted summaries are persisted in orchestrator state and included in later turns.
- **BUG-007:** OpenAI-compatible EOF/`[DONE]` preserves `ToolUse`, and actual emitted tool calls take precedence over an inconsistent provider stop reason.
- **BUG-019:** Long-term memory tails are selected on Unicode character boundaries.
- **BUG-042:** `IpcServer::bind` retains its original listener and `run` consumes that listener instead of rebinding.
- **BUG-045:** MCP connection output now reports each server's own tool count.

### Partially fixed; keep open (2)

- **BUG-001:** Modified/deleted/unreadable/unsupported files no longer leave duplicate or stale rows, and first indexing filters to files. A single transaction spanning delete + metadata + chunk/vector replacement is still pending.
- **BUG-031:** Global/document locks are released before socket writes and before unrelated awaits. Bounded IPC framing and socket-write timeouts are still pending.

All other Category A findings remain pending. Category B-D classifications and the BUG-044 retraction remain unchanged.

## Final requirement-aware confirmation

> **This section is authoritative and supersedes the raw severity headings later in the document.** Revalidation distinguishes implementation defects from behavior intentionally required by the architecture. Do not remove a required feature merely because its current implementation has an unsafe edge case.

### A. Confirmed implementation bugs — fix these (33)

`BUG-001`, `BUG-002`, `BUG-003`, `BUG-005`, `BUG-006`, `BUG-007`, `BUG-008`, `BUG-009`, `BUG-010`, `BUG-011`, `BUG-012`, `BUG-014`, `BUG-016`, `BUG-017`, `BUG-018`, `BUG-019`, `BUG-020`, `BUG-021`, `BUG-022`, `BUG-023`, `BUG-024`, `BUG-028`, `BUG-029`, `BUG-030`, `BUG-031`, `BUG-032`, `BUG-033`, `BUG-035`, `BUG-036`, `BUG-041`, `BUG-042`, `BUG-043`, `BUG-045`.

These have concrete incorrect behavior, panic/data-loss potential, hangs/leaks, broken protocol wiring, or misleading output. Important requirement clarifications:

- **BUG-002:** Git checkpoint/undo is a required feature; `reset --hard` is not automatically a bug. The defect is ignoring checkpoint failure, selecting a stale checkpoint, and deleting post-checkpoint user edits without protection.
- **BUG-003:** Local IPC is required; range/version validation must be added rather than removing IPC/CRDT support.
- **BUG-011:** The editor extension is WIP, but README says the IPC backend is ready. An empty document registry and a `broadcast` method that sends to nobody are implementation defects in that backend.
- **BUG-012:** Ignored timeout, suite, and concurrency arguments are confirmed bugs. Running a full agent against SWE-bench prompts is separately documented as WIP; do not treat that unfinished benchmark capability as a regression.
- **BUG-014:** Process fallback/no-MicroVM is intentional. Unbounded pre-truncation output and incomplete process-tree termination are the defects.
- **BUG-018:** Zero vectors are acceptable as keyword-only placeholders only if vector retrieval excludes them; the fallback itself need not be removed.
- **BUG-032:** No built-in post hook currently depends on this, but the exposed `PostTool` API cannot enforce its own verdicts and is therefore defective for consumers.

### B. Conditional security/design risks — policy decision required (4)

`BUG-004`, `BUG-015`, `BUG-038`, `BUG-040`.

These are real attack surfaces, but whether they violate requirements depends on the trust model:

- **BUG-004:** The current CLI uses session `default`, so traversal is primarily a public-library/API risk. Validation is still recommended.
- **BUG-015:** Shell execution is intentionally process-based and best-effort, not a true sandbox. The mismatch is claiming destructive/secret policy coverage that simple patterns do not provide. Keep shell execution, but document and strengthen the boundary.
- **BUG-038:** Google accepts API keys in query strings, so requests currently function. Moving keys to `x-goog-api-key` is security hardening against URL/log exposure, not a functional repair.
- **BUG-040:** Auto-starting configured MCP servers is needed for MCP. If workspaces are always explicitly trusted, this may be accepted behavior; if opening untrusted repositories is supported, a trust/approval gate is required.

### C. Known incomplete or intentional limitations — do not call regressions (4)

`BUG-013`, `BUG-025`, `BUG-027`, `BUG-034`.

- **BUG-013:** LSP is explicitly listed as WIP. Dropped requests/diagnostics identify work needed to finish it, not a regression in the native `check_code` path.
- **BUG-025:** Sub-agents are explicitly wired as “reasoning-only” with no tools. This limits repository exploration but is intentional current behavior. Rename/re-document it or add read-only tools if repository research is a product requirement.
- **BUG-027:** Baseline is a visible CLI stub. It is unfinished functionality, not hidden data corruption.
- **BUG-034:** Graph retrieval schema/search exists but indexing does not populate it. Treat it as an incomplete advertised feature.

### D. Quality, packaging, or CI recommendations — not runtime bugs (4)

`BUG-026`, `BUG-037`, `BUG-039`, `BUG-046`.

- **BUG-026 correction:** The original impact was overstated. The function still stays below roughly 2000 **characters**; mixing byte-length detection with character truncation can unnecessarily append `[truncated]` to short multibyte text, but it does not violate the documented character cap.
- **BUG-037/046:** CI strictness and test-target coverage are project quality policy.
- **BUG-039:** Tracking `Cargo.lock` is strongly recommended for reproducible CLI builds, but ignoring it does not make the running system incorrect.

### E. Retracted as an unproven current bug (1)

- **BUG-044 (Gemini same-name tool IDs):** Duplicate internal IDs are undesirable for provider-neutral semantics, but current Gemini request conversion maps responses by function name/order and no concrete failure path was demonstrated. Keep it as a compatibility hardening item, not a confirmed bug, unless a live multi-call test reproduces incorrect result association.

### Practical conclusion

- **Fix category A.**
- **Decide category B from the workspace trust/security model.**
- **Schedule category C only if those WIP features are part of the release scope.**
- **Category D improves maintainability/reproducibility but is not required for immediate runtime correctness.**
- **Do not implement BUG-044 as an urgent fix without a reproducer.**

## Severity guide

- **Critical:** likely data loss, workspace escape, or a core advertised feature is fundamentally unsafe.
- **High:** major feature failure, security-policy bypass, crash, indefinite hang, or persistent corruption.
- **Medium:** incorrect behavior, resource leak, misleading result, or important edge-case failure.
- **Low:** limited-impact correctness/quality issue.

## Critical findings

### BUG-001 — Incremental indexing permanently duplicates modified files and never removes deleted files
**Evidence:** `bin/cli/src/main.rs:475-567`; deletion/unreadable paths are skipped at `512-515`, while changed files are `upsert_file`d and immediately `insert_chunks` at `554-567`. `VecStore::delete_file` exists at `crates/vecstore/src/store.rs:117` but is never called by the index command.

**Trigger/impact:** Re-run `index` after modifying a supported source file: its old chunks remain and new chunks are appended. Delete a previously indexed file: its chunks remain searchable forever. Search quality degrades on every edit and deleted/confidential code can continue appearing in results.

**Required direction:** Distinguish added/modified/deleted paths, delete old rows before reinserting modified files, delete removed files, and commit file metadata/chunks/vectors atomically.

### BUG-002 — `/undo` can hard-reset to a stale checkpoint and destroy user work
**Evidence:** checkpoint failures are ignored at `bin/cli/src/main.rs:439-441`; `git_checkpoint` does not inspect either `git add` or `git commit` exit status at `941-947`; undo executes `git reset --hard` at `974-978`.

**Trigger/impact:** If checkpoint commit fails (missing Git identity, hook/config failure, lock, permissions), chat continues. `/undo` then finds an older checkpoint and hard-resets to it. Any user edits or commits made after that checkpoint can be lost. Even with a valid checkpoint, user edits made concurrently during the agent turn are discarded.

**Required direction:** Never ignore checkpoint status; record the exact successful checkpoint per turn; refuse undo when checkpoint creation failed; restore only agent-owned paths/patches instead of hard-resetting the repository.

### BUG-003 — Untrusted IPC patches can panic and stale patches silently corrupt text
**Evidence:** `crates/apply-engine/src/crdt_doc.rs:51-60` clamps offsets but does not ensure `start <= end` or UTF-8 character boundaries before subtraction and string slicing. `PatchMessage.version` is parsed in `crates/apply-engine/src/ipc.rs:73-81` but never checked before applying patches.

**Trigger/impact:** A patch with reversed offsets underflows/panics; an offset inside a multibyte UTF-8 character panics; an old editor patch is applied to current byte positions and edits unrelated text. The implementation is described as a CRDT, but version conflicts are ignored and convergence is not guaranteed.

**Required direction:** Validate range order and UTF-8 boundaries, reject stale versions or transform operations against intervening edits, and return structured errors instead of panicking connection tasks.

### BUG-004 — Spec session IDs permit path traversal outside the workspace
**Evidence:** `crates/spec-pipeline/src/stages.rs:69-74` directly appends untrusted `session_id`; all reads/writes use `self.session_dir.join(...)` at `79`, `105`, `129`, and `144`. A `PathJail` is stored but never used for these paths.

**Trigger/impact:** A library caller can use a session ID such as `..\..\outside` and make artifacts read from or written outside `.agent/specs` and potentially outside the project root.

**Required direction:** Restrict session IDs to a safe identifier format and resolve every artifact path through `PathJail` before I/O.

## High findings

### BUG-005 — Write conflict handling can overwrite the only external copy after backup failure
**Evidence:** `crates/agent-core/src/builtin.rs:171` converts any read error into empty content; conflict backup errors are discarded at `195-200`; the agent version is then written to the original path.

**Trigger/impact:** Invalid UTF-8, permission/transient read errors, a full disk, or an unwritable backup path can lead to overwrite despite the promise that both versions are preserved. Existing backup names can also be overwritten.

**Required direction:** Treat existing-file read errors as fatal, require a successful unique atomic backup before overwrite, and propagate backup failures.

### BUG-006 — Compacted conversation context is forgotten on the following turn
**Evidence:** `crates/agent-core/src/orchestrator.rs:99-105` appends the summary only to the current request’s temporary `system_text`, then replaces history with retained messages. The summary is not persisted in history or orchestrator state.

**Trigger/impact:** The first over-budget turn sees the summary; on a later under-budget turn no summary is regenerated, so all previously compacted decisions/files/tasks disappear from model context.

**Required direction:** Persist and cumulatively update the compacted summary, and include it on every subsequent provider request.

### BUG-007 — OpenAI-compatible `[DONE]` handling can emit tool calls and then mark the turn `EndTurn`
**Evidence:** `crates/llm-client/src/openai_compat.rs:282-289` drains pending tool calls, then checks the now-empty map and emits `StopReason::EndTurn`. `crates/agent-core/src/orchestrator.rs:204-207` refuses to execute tool calls whenever stop reason is `EndTurn`.

**Trigger/impact:** Providers that terminate tool-call streams with `[DONE]` but no usable `finish_reason` produce visible tool calls that are never dispatched; the agent silently stops.

**Required direction:** Remember whether tool calls existed before draining and emit `ToolUse`; in the orchestrator, actual tool calls should take precedence over an inconsistent stop reason.

### BUG-008 — Every namespaced MCP tool is invoked using the wrong remote name
**Evidence:** schemas are renamed to `mcp__<server>__<tool>` in `crates/mcp/src/schema.rs:107`; `McpTool::invoke` sends that namespaced schema name at `crates/agent-core/src/builtin.rs:861`, while the remote server expects its original advertised name.

**Trigger/impact:** Normal MCP servers return “tool not found” for all calls, making MCP discovery appear successful while invocation fails.

**Required direction:** Store both local/namespaced and original remote names; expose the former to the LLM and send the latter in `tools/call`.

### BUG-009 — MCP requests can hang forever or consume another request’s response
**Evidence:** `crates/mcp/src/client.rs:105-159` separately locks stdin and stdout, reads exactly one line, does not validate response `id`, and has no timeout/background response router.

**Trigger/impact:** Concurrent tool calls can swap responses depending on lock acquisition order. Notifications or log lines can be mistaken for responses. A silent server blocks startup or a tool call indefinitely.

**Required direction:** Use one reader task that routes JSON-RPC responses by ID, handles notifications, serializes writes, and applies initialization/request timeouts.

### BUG-010 — A multibyte MCP description can crash schema validation
**Evidence:** `crates/mcp/src/schema.rs:101-104` uses `description[..4096]`, where 4096 is a byte offset that may not be a UTF-8 boundary.

**Trigger/impact:** An untrusted MCP server can advertise a long Unicode description whose 4096th byte splits a code point, panicking the client during connection.

**Required direction:** Truncate by `chars()` or find a valid floor character boundary.

### BUG-011 — CLI IPC server has no registered documents, and `broadcast` does not broadcast
**Evidence:** `bin/cli/src/main.rs:909-912` binds/runs an empty `IpcServer` and never calls `register_doc`. Incoming patches are applied only when a document already exists (`crates/apply-engine/src/ipc.rs:78-81`). `broadcast` merely mutates a document and returns a message at `97-108`; no client writers are retained or sent to.

**Trigger/impact:** `cli serve` accepts connections but normal editor patches receive no acknowledgement and change nothing. Outbound “broadcast” reaches no editor client.

**Required direction:** Define document open/register protocol, retain connection senders, acknowledge unknown files with errors, and actually fan out outbound messages.

### BUG-012 — Eval timeout/concurrency/suite/prompt settings are ignored; the agent is never evaluated
**Evidence:** `_max_concurrent` is unused at `bin/cli/src/main.rs:755`; all case files are loaded regardless of `suite`; each case only runs `check_cmd` at `781-795` and records zero turns/tool calls/tokens; `_timeout_secs` is unused in blocking `Command::output` at `864-881`.

**Trigger/impact:** A hanging check freezes the CLI indefinitely; `--max-concurrent` and suite selection do nothing; fixtures are checked without giving `case.prompt` to an agent, so results do not measure coding-agent performance.

**Required direction:** Run isolated agent sessions from each prompt, enforce process-tree timeouts, filter suites, and use bounded concurrency.

### BUG-013 — LSP server requests and diagnostics are silently discarded
**Evidence:** `crates/lsp-client/src/client.rs:225-233` explicitly consumes server-initiated requests without replying and drops all notifications. `DiagnosticsStore::on_publish_diagnostics` is never wired to the reader.

**Trigger/impact:** Language servers can stall waiting for responses (for example capability/configuration requests), and published diagnostics never reach `DiagnosticsStore`; advertised LSP diagnostics functionality is nonfunctional.

**Required direction:** Give the reader access to stdin for JSON-RPC responses and route notifications to registered handlers/stores.

### BUG-014 — Command timeout only kills the shell, while output “cap” allocates without limit first
**Evidence:** `crates/sandbox/src/executor.rs:74-89` calls `read_to_end` before truncating to 1 MiB; cancellation/timeout calls `start_kill` only on the direct child at `97-99` and `122-125`.

**Trigger/impact:** A command can exhaust memory by streaming large output. A shell-spawned child/background process can survive shell termination, continue modifying files/network, and keep pipes open.

**Required direction:** Stream into bounded buffers and launch/terminate a process group/job object recursively on timeout and cancellation.

### BUG-015 — Secret/destructive-command policies have straightforward bypasses
**Evidence:** `SecretLeakHook` only scans `bash` and `write_file` (`crates/harness/src/hooks.rs:115-118`), excluding `edit_file` and MCP tools. `DestructiveCommandHook` only checks lowercase-sensitive Unix patterns on `bash` (`163-187`). Windows approval patterns at `crates/agent-core/src/builtin.rs:669-687` omit `powershell Remove-Item`, `erase`, `git reset --hard`, and similar commands.

**Trigger/impact:** Secrets can be inserted with `edit_file` or sent to remote MCP tools without detection. On Windows, `powershell -Command Remove-Item -Recurse ...` can run with no approval and no policy denial.

**Required direction:** Classify tool effects instead of matching tool names, scan every outbound/write payload, normalize command case/syntax, and use structured allow/deny policy for Windows and Unix shells.

### BUG-016 — “Atomic replace” fails on Windows and temp names collide across concurrent writes
**Evidence:** apply engine uses one PID-only temp name and `tokio::fs::rename` at `crates/apply-engine/src/apply_engine.rs:84-89`; spec artifacts use a fixed extension temp path at `crates/spec-pipeline/src/stages.rs:135-138`.

**Trigger/impact:** Concurrent writes in one directory race on the same temp file. On Windows, rename does not reliably replace an existing destination, so updating an existing target/artifact can fail instead of atomically replacing it.

**Required direction:** Create unique same-directory temp files and use a platform-correct atomic replace operation with cleanup on failure.

### BUG-017 — `TaskScope` does not observe the first completed error and can leak tasks when dropped
**Evidence:** `crates/runtime-core/src/task_scope.rs:73-91` awaits handles in spawn order. There is no `Drop` implementation to cancel/abort unjoined handles.

**Trigger/impact:** If the first spawned task hangs while a later task immediately fails, `join_all` waits for the hanging task and never promptly cancels siblings. Dropping a scope without `join_all` detaches all tasks, violating structured-concurrency guarantees.

**Required direction:** Use `JoinSet`/`FuturesUnordered` completion order and cancel/abort/reap remaining tasks on error and drop.

### BUG-018 — Index search can mix real query vectors with zero-placeholder document vectors
**Evidence:** indexing stores `vec![0.0; 768]` when no key or embedding failure (`bin/cli/src/main.rs:537-542`). Search switches to hybrid solely because a key exists at search time; it does not know whether stored vectors are valid.

**Trigger/impact:** Index without a key, then search with a key: vector KNN ranks zero placeholders and pollutes BM25 results with meaningless vector scores.

**Required direction:** Store embedding status/model/dimension per chunk/index and disable vector retrieval until compatible real embeddings exist.

## Medium findings

### BUG-019 — Loading long-term memory can panic on Unicode
**Evidence:** `bin/cli/src/main.rs:311-314` slices at byte offset `mem.len() - 2000` without checking a UTF-8 boundary.

**Trigger/impact:** A `MEMORY.md` longer than 2000 bytes with a multibyte character crossing that offset crashes `chat` startup.

**Required direction:** Select the tail by characters or move to the next valid boundary.

### BUG-020 — Provider usage events are normally lost
**Evidence:** orchestrator breaks immediately on `SseEvent::Stop` (`crates/agent-core/src/orchestrator.rs:166-168`). Gemini queues `Stop` before `Usage` (`crates/llm-client/src/gemini.rs:434-458`), so usage remains unread. OpenAI checks `finish_reason` and returns at `337-353` before its usage block at `355-363`; usage-only chunks are skipped at `298-303`.

**Trigger/impact:** Token/session totals shown by the CLI are missing or undercounted for Gemini/OpenAI-compatible streams.

**Required direction:** Process usage before stop, or drain final metadata after stop; request provider-specific streaming usage metadata where required.

### BUG-021 — LSP timeout/body/diagnostic-settle paths leak or misbehave
**Evidence:** pending entries are inserted before send at `crates/lsp-client/src/client.rs:107-115` and never removed on send failure/timeout; incoming `Content-Length` allocates `vec![0u8; len]` without a cap at `202`; diagnostics stability only compares item count and resets the same timer used as its “hard cap” (`crates/lsp-client/src/diagnostics.rs:182-201`).

**Trigger/impact:** Repeated timeouts grow the pending map; a malicious/broken server can request huge allocation; continuously changing diagnostic counts can prevent the claimed 10-second cap, while changed diagnostics with the same count are treated as stable.

**Required direction:** Remove pending IDs on every failure, cap message size, use a separate absolute deadline, and compare diagnostic versions/content.

### BUG-022 — LSP URIs and CRLF byte-to-column conversion are incorrect
**Evidence:** root URI is manually formed without percent encoding at `crates/lsp-client/src/client.rs:35`; `byte_offset_to_utf16_col` assumes every line separator occupies one byte at `crates/lsp-client/src/diagnostics.rs:207-215`.

**Trigger/impact:** Workspace paths containing spaces, `#`, `%`, or non-ASCII can be rejected/misinterpreted. For CRLF files, positions after the first line are shifted, producing incorrect LSP columns.

**Required direction:** Build file URIs with a URL/path library and compute line starts from actual byte terminators.

### BUG-023 — Persistent state setters can replace unreadable files with empty/partial content
**Evidence:** `crates/state-store/src/persistent.rs:69` and `130` turn every read failure into an empty string, then write at `103` and `147`.

**Trigger/impact:** Invalid UTF-8, transient I/O, or permission problems can cause `SOUL.md`/`HEARTBEAT.md` to be overwritten rather than returning the read error. `set_task_done` also returns success for an out-of-range index and still rewrites the file.

**Required direction:** Only default on `NotFound`, propagate all other errors, verify the requested task exists, and write atomically.

### BUG-024 — Workspace skills are never loaded by the CLI
**Evidence:** the loader supports a directory, but CLI always calls `SkillRegistry::load(None)` at `bin/cli/src/main.rs:257`.

**Trigger/impact:** `.agent/skills/*.toml` is ignored in normal chat despite the feature documentation.

**Required direction:** Pass `project_root.join(".agent/skills")` when it exists and report malformed skill files cleanly instead of `unwrap`.

### BUG-025 — Sub-agent repository access is an intentional current limitation
**Revalidated status:** Not a confirmed defect unless repository-aware sub-agents are a release requirement.  
**Evidence:** `crates/agent-core/src/builtin.rs:822-825` explicitly passes an empty tool list with the comment `reasoning-only sub-agents`. `crates/harness/src/subagent.rs:42-84` returns on a normal stop or attempted tool use.

**Current behavior:** Sub-agents can reason about the task text supplied to them but cannot independently read/search the repository. This matches the explicit reasoning-only wiring, although the “exploration/research” wording can create a stronger expectation.

**Required direction if repository research is desired:** Supply vetted read-only tools plus workspace context, execute tool calls, and continue a bounded tool loop. Otherwise rename/re-document the feature and simplify the unused multi-turn structure.

### BUG-026 — Byte/character mismatch can mark short Unicode summaries as truncated
**Revalidated status:** Quality issue; original severity/impact was overstated.  
**Evidence:** `crates/harness/src/subagent.rs:156-161` tests UTF-8 byte length but truncates by Unicode scalar count.

**Trigger/impact:** A summary containing fewer than 2000 characters but more than 2000 UTF-8 bytes (for example emoji-heavy text) unnecessarily enters the truncation branch and may receive a `[truncated]` marker. The resulting output still remains below roughly 2000 characters, so this is not a context-limit or memory-safety failure.

**Required direction:** Compare character count when enforcing a character limit, or rename and consistently implement the limit as bytes.

### BUG-027 — Eval baseline/diff commands are incomplete and malformed rows are silently ignored
**Evidence:** baseline only prints manual instructions at `bin/cli/src/main.rs:124-126`; omitted `run_b` defaults to empty at `78` and then attempts to load an empty run ID at `832`; JSON parse failures are dropped by `filter_map` at `825-828`.

**Trigger/impact:** `eval baseline` does not set a baseline, default diff behavior fails, and corrupted result lines can hide failures/regressions.

**Required direction:** Implement baseline persistence, resolve omitted `run_b` to it, and fail with line-numbered parse errors.

### BUG-028 — Eval run IDs collide within one second and append unrelated runs
**Evidence:** `chrono_stub_now` uses epoch seconds at `bin/cli/src/main.rs:884-890`; outcome files are append-only (`crates/evals/src/swebench.rs:86-90`).

**Trigger/impact:** Two runs started in the same second share a file, mixing duplicate/unrelated outcomes and invalidating reports/diffs.

**Required direction:** Use UUID or nanosecond timestamp plus exclusive file creation.

### BUG-029 — Regression percentages are wrong for zero baselines and removed cases are invisible
**Evidence:** `crates/evals/src/trajectory.rs:57-72` iterates only current cases; threshold calculations use raw zero but displayed percentages divide by `max(1)`.

**Trigger/impact:** A change from 0 to 1 turn/token can be flagged as a regression while displayed as `+0%`; a baseline case missing entirely from the current run is not reported.

**Required direction:** Define zero-baseline semantics explicitly and compare the union of case IDs.

### BUG-030 — `Tests` and `Implement` artifacts are declared as directories but written as files
**Evidence:** `crates/spec-pipeline/src/stages.rs:28-29` returns `tests/` and `code/`; generic `write_artifact` writes/renames one file to those paths at `129-138`.

**Trigger/impact:** Directory semantics are lost; future code expecting multiple test/code artifacts or traversable directories fails, and repeated stages interact badly with existing directories.

**Required direction:** Use concrete markdown artifact filenames or implement real directory artifact handling.

### BUG-031 — IPC accepts unbounded lines and holds global locks while writing to a client
**Evidence:** `crates/apply-engine/src/ipc.rs:70-87` uses `BufRead::lines()` without a frame-size cap, holds the global document map lock and document lock, then awaits socket `write_all`.

**Trigger/impact:** A local client can consume arbitrary memory with a never-terminated line. A slow/non-reading client can block document registration, broadcast, and patches for every other connection.

**Required direction:** Use bounded framing, clone the document handle before locking it, release state locks before network I/O, and time out writes.

### BUG-032 — Post-tool hooks cannot inspect results and their verdicts are ignored
**Evidence:** `crates/agent-core/src/tools.rs:69` invokes post hooks with `Value::Null`, discards the returned verdict/error, and does not provide input/output/status.

**Trigger/impact:** Any post-execution audit, redaction, or fail-closed policy built on the public hook API silently has no enforcement effect.

**Required direction:** Pass structured input/result/error data and define/enforce post-hook deny/rewrite semantics.

### BUG-033 — `TurnEnded` is not emitted on provider errors
**Evidence:** `TurnStarted` is emitted at `crates/agent-core/src/orchestrator.rs:77`; stream creation uses `?` at `141-144` and stream errors return directly at `176`; `TurnEnded` is only emitted on the success path at `249`.

**Trigger/impact:** UI/trajectory consumers can remain stuck in “thinking” state and record incomplete lifecycle data after network/provider errors or cancellation.

**Required direction:** Use a scope guard/finally-style path that emits terminal events for success, failure, and cancellation.

### BUG-034 — Graph search is exposed but the graph tables are never populated
**Evidence:** `crates/vecstore/src/hybrid.rs:159-190` lazily creates `entities`, `entity_edges`, and `entity_chunks`; no workspace code inserts entities/edges/mappings into those tables.

**Trigger/impact:** `SearchMode::Graph` always returns empty in normal indexing, and the advertised graph component of hybrid retrieval contributes nothing.

**Required direction:** Persist parser entities and relationships during indexing, map them to chunks, and remove/update graph rows incrementally.

## Low findings

### BUG-035 — First index counts directory-hash nodes as files
**Evidence:** `MerkleTree.nodes` contains both files and computed directories; first-index paths use all `current_tree.nodes.keys()` at `bin/cli/src/main.rs:480-483`, and the CLI prints `nodes.len()` as “Files found.”

**Impact:** Progress/file counts are inflated and the loop attempts to read directory keys before skipping them.

### BUG-036 — TypeScript declarations can miss later function variables
**Evidence:** `crates/indexer/src/parser.rs` returns `None` as soon as the first `variable_declarator` in a declaration is not function-like, instead of checking later declarators.

**Impact:** Code such as `const value = 1, handler = () => {};` does not index `handler`.

### BUG-037 — Existing CI deliberately ignores format and Clippy failures
**Evidence:** `.github/workflows/ci.yml` sets `continue-on-error: true` for both formatting and Clippy.

**Impact:** Warning/lint regressions (currently present) do not fail CI even though Clippy is invoked with `-D warnings`.

## Recommended repair order

1. **Prevent data loss/escape:** BUG-002, BUG-003, BUG-004, BUG-005, BUG-016.
2. **Repair core behavior:** BUG-001, BUG-006, BUG-007, BUG-008, BUG-011, BUG-012.
3. **Close security/runtime gaps:** BUG-009, BUG-010, BUG-014, BUG-015, BUG-017, BUG-031.
4. **Fix integrations/results:** BUG-013, BUG-018 through BUG-030, BUG-032 through BUG-036.
5. **Tighten CI:** install Clippy in the local toolchain and remove CI `continue-on-error` once warnings are resolved.


---

## Supplemental verification of additional reported candidates

The following candidates were checked separately after the main audit.

### Newly confirmed findings

### BUG-038 — Gemini API keys are placed in query strings for generation and embeddings
**Severity:** High (credential-exposure risk)  
**Evidence:** `crates/llm-client/src/gemini.rs:161-167` builds `...?alt=sse&key=<api_key>`; `crates/llm-client/src/embeddings.rs:29-33` builds `...?embedContent?key=<api_key>`.

**Trigger/impact:** Query strings are more likely than headers to appear in proxy/access logs, tracing systems, browser/debug tooling, or transport error text. The custom non-success HTTP error only prints status/body (`gemini.rs:253-255`), but reqwest transport errors are forwarded with `e.to_string()` and can include the request URL. Both generation and embedding keys are affected.

**Required direction:** Send the key in the supported `x-goog-api-key` header and keep only non-secret parameters such as `alt=sse` in the URL. Ensure errors/logging redact credentials.

### BUG-039 — `Cargo.lock` is ignored and is not tracked for the CLI workspace
**Severity:** Medium (build reproducibility)  
**Evidence:** `.gitignore:3` ignores `Cargo.lock`; `git check-ignore -v Cargo.lock` confirms that rule; `git ls-files --error-unmatch Cargo.lock` confirms the existing lockfile is not tracked.

**Trigger/impact:** Fresh clones resolve dependency versions independently. Because this workspace produces the `cli` binary, dependency drift can change or break builds without source changes.

**Required direction:** Remove `Cargo.lock` from `.gitignore` and commit the root lockfile. Library consumers still choose their own resolution when depending on individual crates.

### BUG-040 — Opening an untrusted workspace can execute repository-controlled MCP commands without approval
**Severity:** High (local arbitrary command execution / supply-chain risk)  
**Evidence:** `bin/cli/src/main.rs:222-240` reads `.agent/mcp.json`, takes each `command`/`args`, and immediately calls `McpClient::connect`; `crates/mcp/src/client.rs:25-36` spawns it. No trust check or confirmation is present.

**Trigger/impact:** Starting `chat` in a malicious or compromised project can execute any command declared in its MCP config before the user asks the agent to run a tool. The bash approval layer is bypassed because MCP processes are spawned directly.

**Required direction:** Treat workspace MCP configuration as untrusted; require explicit per-workspace approval, display command/arguments, store a trust decision keyed to config hash, and offer disabled-by-default mode.

### BUG-041 — Bash approval and chat readers can compete for the same stdin
**Severity:** Medium  
**Evidence:** chat owns `tokio::io::BufReader<Stdin>` at `bin/cli/src/main.rs:380-382`; approval separately calls blocking `std::io::stdin().read_line` at `crates/agent-core/src/builtin.rs:620-629`.

**Trigger/impact:** The async buffered reader can read ahead (especially pasted or piped multiline input), leaving the blocking approval reader waiting while the intended answer is already buffered elsewhere. Conversely, approval can consume text intended as the next chat prompt.

**Required direction:** Centralize stdin ownership in the CLI and expose approval through a channel/callback instead of letting tools read stdin directly.

### BUG-042 — IPC `bind()` drops its listener and `run()` rebinds the port
**Severity:** Medium  
**Evidence:** `crates/apply-engine/src/ipc.rs:30-40` binds only to test availability and drops `_listener`; `run()` binds again at `51-58`.

**Trigger/impact:** Another process can claim the port between the two calls, so a successful `IpcServer::bind` can still fail immediately in `run` (TOCTOU race).

**Required direction:** Store the original `TcpListener` in `IpcServer` and reuse it in `run`.

### BUG-043 — `search_text` loads every candidate file fully with no size cap
**Severity:** Medium (resource exhaustion)  
**Evidence:** `crates/agent-core/src/builtin.rs:531-535` calls `tokio::fs::read_to_string` for each regular file without checking metadata size or streaming.

**Trigger/impact:** A workspace containing very large text files can cause high memory use, long pauses, and cancellation latency. The 200-match result cap does not bound bytes read before matches are found.

**Required direction:** Skip/cap large files or stream them line-by-line with per-file and aggregate byte limits.

### BUG-044 — Gemini same-name tool IDs are a compatibility concern, not a confirmed current failure
**Revalidated status:** Retracted as a confirmed bug pending a reproducer.  
**Evidence:** `crates/llm-client/src/gemini.rs:419-423` creates `id: format!("gemini_{}", name)`; name recovery strips that prefix at `175-178`.

**Assessment:** Two same-name calls receive identical internal IDs, which is undesirable in a provider-neutral message model. However, current Gemini serialization emits `functionResponse` by function name and preserves block order; no code path was found that demonstrably swaps or drops these results solely because of the duplicate ID.

**Optional hardening:** Add a per-stream call index/UUID and store the original function name separately, but do not prioritize this as a bug fix without a live parallel same-name call reproducer.

### BUG-045 — MCP connection log prints cumulative rather than per-server tool count
**Severity:** Low  
**Evidence:** `bin/cli/src/main.rs:224-246` increments global `mcp_count` and prints that cumulative value for each server.

**Impact:** The second and later server log lines report the total tools loaded so far, not tools from that server.

**Required direction:** Capture `server_tool_count = tools.len()` for the line, then separately update the total.

### BUG-046 — CI excludes integration/doc/binary-target tests
**Severity:** Low (coverage gap)  
**Evidence:** `.github/workflows/ci.yml` runs `cargo test --workspace --lib`.

**Impact:** Only library unit-test targets run; integration tests, binary target tests, examples where applicable, and normal doctest coverage are omitted. This compounds BUG-037, where formatting and Clippy failures are also non-blocking.

**Required direction:** Use `cargo test --workspace --all-targets` plus explicit doctests as appropriate, while keeping network-dependent examples gated.

### Already covered by earlier findings

- UTF-8 memory-tail panic: **BUG-019**.
- `/undo` hard reset/checkpoint failure and user-work loss: **BUG-002**.
- Eval timeout/concurrency and nonfunctional runner: **BUG-012**.
- Baseline command is a print-only stub: **BUG-027**.
- Command-policy pattern bypasses, including alternate flags/syntax/Windows commands: **BUG-015**.
- Zero-vector search pollution: **BUG-018**.

### Verified as hygiene/performance observations, not standalone bugs

- `.gitignore` contains likely leftovers `snake-game/` and `%SystemDrive%/`: confirmed, but this is repository hygiene unless those paths are intentionally needed. `%SystemDrive%/` corresponds to junk currently present in the workspace and should be cleaned deliberately.
- `NetGuard::get()` builds a pinned `reqwest::Client` per request/redirect while the struct’s `client` field is unused: confirmed (`crates/sandbox/src/net_guard.rs`). This is a minor performance/design issue; per-request construction currently supports host-to-validated-IP pinning, so replacing it requires preserving DNS-rebinding protection.
- Default model names are hardcoded but all main chat providers support environment overrides. Moving defaults to configuration improves maintainability but is not a correctness bug by itself.
- `crates/llm-client/examples/test_gemini.rs` contains **no API key** (`API_KEY` is an empty string) and does not print credentials. It prints model output, thought summaries, and errors, which is expected for a manual example. The stale comment `Test API key (will be deleted after testing)` should still be removed to prevent someone later pasting a real key into source.

### Important qualification about shell security

The bypass concern is valid. `ProcessFallback` is explicitly documented as “no true isolation,” and the bash tool says it is not a VM sandbox. However, `crates/sandbox/src/executor.rs:9-10` incorrectly claims network is disabled merely because proxy environment variables are cleared; clearing environment variables does **not** prevent direct network access. The command hook/approval layer must therefore be described as best-effort policy, not a hard sandbox boundary.