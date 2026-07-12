# TESTPLAN.md — Kadak Parikshan Yojana (Rigorous Verification Plan)

> Yeh plan un sab claims ko verify karega jo README.md aur plan.md mein kiye gaye hain.
> Har section mein: CLAIM → CURRENT REALITY → TEST → PASS CRITERIA
> Agar koi test fail hota hai toh woh claim README se hatana padega ya implement karna padega.

---

## Category 1: CORE AGENTIC LOOP (Sabse Critical)

### TEST-1.1: Multi-Turn Tool Use Loop

**Claim:** "Reads, writes, and searches files autonomously" + "Loop until StopReason::EndTurn, hard cap 50 iterations"

**Test:**
1. Naya temp folder banao
2. CLI ko bolo: "Create a Python file with a fibonacci function, then read it back and tell me what line the def is on"
3. Verify: write_file + read_file dono call hue, correct output mila

**Pass Criteria:**
- [ ] Agent ne write_file call kiya
- [ ] Agent ne read_file call kiya
- [ ] Agent ne sahi line number bataya
- [ ] Koi crash/panic nahi hua

### TEST-1.2: Tool Error Recovery

**Claim:** "Tool failures are data, not crashes"

**Test:**
1. Agent ko bolo: "read_file karo 'nonexistent_xyz.txt'"
2. Verify: error message mila but agent crashed nahi, conversation continue hui

**Pass Criteria:**
- [ ] Agent ne error gracefully handle kiya
- [ ] Agle turn mein normal response diya

### TEST-1.3: 50-Iteration Hard Cap

**Claim:** "hard cap 50 iterations"

**Test:** Mock provider banao jo hamesha ToolUse return kare. Verify ki 50 pe ruk jaata hai.

**Pass Criteria:**
- [ ] Exactly 50 iterations pe `AgentError::Tool` mila
- [ ] No infinite loop

---

## Category 2: SECURITY SANDBOX (Public Mein Embarrassment Risk)

### TEST-2.1: Path Jail — Directory Traversal Blocked

**Claim:** "PathJail prevents directory traversal"

**Tests:**
1. `read_file("../../etc/passwd")` → MUST fail with PathJail error
2. `read_file("C:\\Windows\\System32\\config\\SAM")` → MUST fail
3. `write_file("../outside.txt", "pwned")` → MUST fail
4. Symlink pointing outside jail → MUST fail (Unix only)

**Pass Criteria:**
- [ ] Har attempt pe `AgentError::PathJail` mila
- [ ] File system pe koi unauthorized read/write nahi hua
- [ ] Error message mein actual path leak nahi hua

### TEST-2.2: Secret Leak Prevention

**Claim:** "blocks AWS keys, API tokens, private key material"

**Tests:**
1. `bash("echo AKIAIOSFODNN7EXAMPLE123456")` → MUST be blocked
2. `write_file("config.txt", "api_key='sk-1234567890abcdef1234'")` → MUST be blocked
3. `bash("cat ~/.ssh/id_rsa")` → allowed (reading is ok, leaking isn't)
4. `write_file("safe.txt", "Hello world")` → MUST pass (no false positive)

**Pass Criteria:**
- [ ] AWS key pattern blocked
- [ ] Generic secret pattern blocked
- [ ] Normal content NOT blocked (zero false positives on sample of 100 normal writes)
- [ ] Blocked reason mein actual secret value nahi echo hua

### TEST-2.3: Destructive Command Blocking

**Claim:** "blocks rm -rf /, mkfs, forced pushes"

**Tests:**
1. `bash("rm -rf /")` → blocked
2. `bash("rm -rf /home/user")` → blocked (starts with rm -rf /)
3. `bash("git push --force origin main")` → blocked
4. `bash("rm file.txt")` → allowed (normal delete)
5. `bash("git push origin main")` → allowed (no force)

**Pass Criteria:**
- [ ] Sab destructive commands blocked
- [ ] Normal commands allowed
- [ ] Zero false positives

### TEST-2.4: Sandbox Executor — Timeout & Kill

**Claim:** "SandboxExecutor runs commands with hard timeouts"

**Tests:**
1. `bash("ping -n 100 127.0.0.1")` with 2s timeout → must terminate
2. Verify process actually killed (not just abandoned)
3. CancellationToken cancel → process killed immediately

**Pass Criteria:**
- [ ] Timeout error returned within timeout + 1s tolerance
- [ ] OS process actually dead (tasklist/ps check)
- [ ] Cancel token kills within 500ms

### TEST-2.5: NetGuard SSRF Protection

**Claim:** "blocks SSRF against private IPs, HTTPS only, domain allowlist"

**Tests:**
1. `http://example.com` → rejected (not https)
2. `https://169.254.169.254/metadata` → rejected (cloud metadata IP)
3. `https://127.0.0.1:8080` → rejected (loopback)
4. `https://10.0.0.1/internal` → rejected (private)
5. `https://evil.com` (not in allowlist) → rejected
6. `https://crates.io` (in allowlist) → allowed (if network available)

**Pass Criteria:**
- [ ] Sab private/loopback/link-local IPs blocked
- [ ] IPv4-mapped IPv6 (::ffff:127.0.0.1) bhi blocked
- [ ] Domain allowlist suffix matching correct

---

## Category 3: LLM PROVIDER (Gemini Integration)

### TEST-3.1: Streaming Response Parsing

**Claim:** "SSE-based streaming"

**Tests:**
1. Simple text prompt → Delta events milne chahiye incrementally
2. Function call prompt → ToolUse event milna chahiye
3. Empty response → no crash, clean Stop event

**Pass Criteria:**
- [ ] Streaming works end-to-end
- [ ] thoughtSignature round-trip works (Gemini 3.x requirement)
- [ ] finishReason correctly maps to StopReason

### TEST-3.2: Token Usage Tracking

**Claim:** Token tracking feature

**Tests:**
1. Ek prompt bhejo, verify Usage event mein prompt_tokens > 0, total_tokens > 0
2. Multi-turn session mein cumulative count increases

**Pass Criteria:**
- [ ] Token counts non-zero aur realistic
- [ ] Session total = sum of all turn totals

### TEST-3.3: Error Handling — Invalid Key / Rate Limit

**Tests:**
1. Invalid API key → clean error message (no panic)
2. 429 response → clean error with retry hint

**Pass Criteria:**
- [ ] Error message readable hai
- [ ] No stack trace / panic exposed to user

---

## Category 4: CODE INDEXING & SEARCH

### TEST-4.1: Tree-Sitter Parsing Accuracy

**Claim:** "Tree-sitter parsing (Rust, Python, TypeScript) extracts functions, classes, methods"

**Tests:**
1. Rust file with fn, struct, impl, enum → sab entities extract hone chahiye
2. Python file with def, class, nested methods → correct qualified names
3. TypeScript file with function, class, arrow functions → extracted
4. Binary file / image → graceful skip, no crash
5. Empty file → empty Vec, no crash

**Pass Criteria:**
- [ ] Rust: fn → Function, struct → Struct, impl method → Method
- [ ] Python: def → Function, class → Class, method → Method
- [ ] TypeScript: function → Function, class → Class
- [ ] qualified_name sahi hai (e.g. "MyClass::my_method")
- [ ] line_range accurate hai (actual line numbers match)

### TEST-4.2: AST Chunking — Token Budget Respected

**Claim:** "greedy recursive AST chunking within token budget"

**Tests:**
1. Large file (1000 lines) → chunks ka token_count ≤ budget
2. Small file (10 lines) → ek hi chunk mein aa jaaye
3. Chunk boundaries function boundaries pe hone chahiye (mid-function split nahi)

**Pass Criteria:**
- [ ] No chunk exceeds budget
- [ ] entity_ids correctly mapped

### TEST-4.3: Hybrid Search — All 3 Modes Work

**Claim:** "Vector KNN, BM25 full-text, graph-based entity traversal, fused with RRF"

**Tests:**
1. Keyword search: "fibonacci" → file containing fibonacci function milna chahiye
2. Vector search: embedding close to target → nearest chunk milna chahiye
3. Hybrid mode: combines both signals, returns union not intersection
4. Empty query → empty results (no crash)
5. Query with special chars ("hello (world)") → sanitized, no FTS5 syntax error

**Pass Criteria:**
- [ ] Keyword search returns relevant results
- [ ] Vector KNN ranks by distance correctly
- [ ] Hybrid RRF score = sum(1/(60+rank_i)) per mode
- [ ] No SQL injection / FTS5 syntax errors

### TEST-4.4: Merkle Tree — Incremental Sync

**Claim:** "Merkle tree diffing detects only changed files"

**Tests:**
1. First run: sab files indexed
2. Ek file change karo, re-run → sirf woh file re-indexed (others skipped)
3. File delete karo → merkle mein missing

**Pass Criteria:**
- [ ] Changed files detected
- [ ] Unchanged files skipped (verify by checking if parser was called)

---

## Category 5: SPEC PIPELINE (RustySpec)

### TEST-5.1: Stage Prerequisites Enforced

**Claim:** "7-stage workflow (Specify → Clarify → Plan → Tasks → Tests → Implement → Analyze)"

**Tests:**
1. Plan stage run without Specify artifact → error
2. Tasks stage run without Plan artifact → error
3. Specify stage → works without prerequisites

**Pass Criteria:**
- [ ] Dependency chain strictly enforced
- [ ] Clear error message tells what's missing

### TEST-5.2: Artifact Validation

**Claim:** "spec.md must have ## User Stories and ## Functional Requirements"

**Tests:**
1. Valid spec with all headers → writes successfully
2. Spec missing "## User Stories" → rejected
3. Atomic write (crash mid-write → no partial file left)

**Pass Criteria:**
- [ ] Validation catches missing headers
- [ ] Temp file cleanup (no .tmp files left on error)

---

## Category 6: LANGUAGE GUARD (Hinglish Mode)

### TEST-6.1: Devanagari Blocked in Machine Surfaces

**Claim:** "EVERY machine-readable surface MUST be ASCII English"

**Tests:**
1. `write_file` with Devanagari in file path → blocked
2. JSON keys with Devanagari → blocked
3. Hinglish prose in content field → allowed
4. English file path + Hinglish content → allowed

**Pass Criteria:**
- [ ] Any U+0900–U+097F in machine surface = rejected
- [ ] Prose fields exempt
- [ ] No crash, error is recoverable (ToolResult is_error: true)

### TEST-6.2: Skill Activation

**Claim:** "hinglish-mode skill activates every turn when lang == Hinglish"

**Tests:**
1. LanguageMode::Hinglish → HinglishSkill prompt fragment present
2. LanguageMode::En → HinglishSkill NOT activated
3. "code review" keyword → CodeReviewSkill activates

**Pass Criteria:**
- [ ] Correct skill activation based on mode/keywords

---

## Category 7: COMPACTION (Context Window Management)

### TEST-7.1: Token Budget Enforcement

**Claim:** "Automatic context-window management that preserves tool-use pair integrity"

**Tests:**
1. History exceed budget → compaction triggers
2. After compaction: recent 4 messages retained verbatim
3. ToolUse + ToolResult pairs never split (both kept or both compacted together)

**Pass Criteria:**
- [ ] Compaction reduces token count below budget
- [ ] Last 4 messages intact
- [ ] No orphaned ToolUse without ToolResult

---

## Category 8: CLAIMS vs REALITY — HONEST AUDIT

Yeh section honestly list karta hai ki README mein kya claim kiya par actually EXIST nahi karta:

| # | README Claim | Actually Implemented? | Action Required |
|---|---|---|---|
| 1 | "Indexes your codebase into a semantic vector store" | Code EXISTS but CLI `index` command is STUB | Wire indexer to CLI |
| 2 | "Anthropic Claude" as provider | Was sole provider; NOW Gemini added | Update README |
| 3 | "Validates its own edits via LSP diagnostics and semantic diff" | apply-engine + lsp-client code exists but NOT wired to orchestrator tools | Wire or remove claim |
| 4 | "CRDT concurrent editing" | crdt_doc.rs code exists but untested in real scenario | Test or remove claim |
| 5 | "MicroVM executor" | ProcessFallback only (dev mode); no actual MicroVM | Clarify in README |
| 6 | "SWE-bench-lite compatible runner" | Trajectory + report structure exists; actual benchmark run NEVER executed | Run or remove claim |
| 7 | "Run all 128 tests" | Actually 142+ tests now | Update count |
| 8 | "MCP client + server" | Basic structure exists, client stub | Clarify scope |
| 9 | "Session compaction summarizes compacted history" | ThresholdCompactor exists but summary generation needs LLM call (not done) | Fix or clarify |
| 10 | ".agent/config.toml" configuration | NOT implemented — CLI uses env vars + hardcoded | Implement or document |

---

## Category 9: INTEGRATION TESTS (End-to-End)

### TEST-9.1: Full Chat Session — File Creation

**Test:** Run CLI, give instruction "Create src/hello.rs with fn main printing hello", verify file exists with correct content.

**Pass Criteria:**
- [ ] File created at correct path
- [ ] Content syntactically valid Rust
- [ ] Agent response mentions success

### TEST-9.2: Full Chat Session — Multi-File Project

**Test:** "Create a Rust project with Cargo.toml and src/main.rs that prints fibonacci of 10"

**Pass Criteria:**
- [ ] Cargo.toml valid TOML
- [ ] src/main.rs compiles (`cargo check`)
- [ ] `cargo run` outputs correct fibonacci number

### TEST-9.3: Full Chat Session — Bug Fix Flow

**Test:** Create a file with a deliberate bug, ask agent to find and fix it.

**Pass Criteria:**
- [ ] Agent reads the file
- [ ] Identifies the bug
- [ ] Writes corrected version
- [ ] Fix is actually correct

---

## Category 10: PERFORMANCE & RELIABILITY

### TEST-10.1: Large File Handling

**Tests:**
1. read_file on 10MB file → works (within output truncation limits)
2. write_file with 100KB content → succeeds
3. search_text across 1000 files → completes within 10 seconds

**Pass Criteria:**
- [ ] No OOM crash
- [ ] Output truncation at 30KB works
- [ ] Search uses cancellation token (abortable)

### TEST-10.2: Concurrent Safety

**Tests:**
1. EventBus: 10 simultaneous subscribers, emit 1000 events → no data race
2. ToolDispatcher: concurrent dispatch of different tools → no deadlock

**Pass Criteria:**
- [ ] No race conditions under stress
- [ ] MIRI passes (if applicable)

---

## Execution Strategy

### Phase 1: Automated (cargo test)
- Run `cargo test --workspace` — 142 existing tests MUST pass
- Add integration tests for TEST-1.x, TEST-2.x as new test files
- Target: 200+ tests total

### Phase 2: Manual Verification
- Run CLI with real Gemini API for TEST-9.x scenarios
- Record terminal output as evidence
- Run against 5 different project types (Rust, Python, JS, empty, large)

### Phase 3: Honest README Update
- Har failed test ke liye: ya toh fix karo ya claim hataao
- "Work in Progress" section add karo for unfinished features
- Version tag karo: v0.1.0 (alpha) — clearly communicated

### Phase 4: Security Audit
- Run all TEST-2.x with adversarial inputs
- Fuzz PathJail with random paths (10000 iterations)
- Verify no panic anywhere outside tests (`grep -r "unwrap\(\)" --include="*.rs" | grep -v test | grep -v "// SAFETY"`)

---

## Summary: Open Source Readiness Score

| Category | Tests | Minimum Pass Rate for Publish |
|----------|-------|------|
| Core Loop (1.x) | 3 | 100% |
| Security (2.x) | 5 | 100% — ek bhi fail = NO publish |
| LLM Provider (3.x) | 3 | 100% |
| Indexing/Search (4.x) | 4 | 75% (Merkle soft requirement) |
| Spec Pipeline (5.x) | 2 | 100% |
| Language Guard (6.x) | 2 | 100% |
| Compaction (7.x) | 1 | 100% |
| Honest Audit (8) | 10 items | All addressed (fix or document) |
| Integration (9.x) | 3 | 100% |
| Performance (10.x) | 2 | 75% |

**Rule: Security tests 100% pass nahi → publish NAHI karna. Izzat jaayegi.**
