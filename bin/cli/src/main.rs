//! `cli` (L5): clap-based CLI entrypoint.
//!
//! Subcommands: chat, index, spec, search, eval.
//! Ctrl-C cancels the root CancellationToken → graceful drain.

use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;

use agent_core::{Orchestrator, ToolDispatcher};
use harness::{HookEngine, SkillRegistry};
use llm_client::GeminiProvider;
use llm_client::LlmProvider;
use runtime_core::EventBus;

#[derive(Parser)]
#[command(name = "rust-agent", version, about = "Autonomous Rust AI Coding Agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat with the agent
    Chat {
        /// Workspace directory (default: current directory).
        /// The agent will read/write files in this folder.
        #[arg(short = 'w', long = "workspace")]
        workspace: Option<String>,
    },
    /// Start IPC server for editor integration (VS Code extension connects here)
    Serve {
        /// TCP port to listen on (default: 9527)
        #[arg(short = 'p', long = "port", default_value = "9527")]
        port: u16,
    },
    /// Index the codebase (Merkle diff → parse → chunk → embed → store)
    Index,
    /// Run a RustySpec pipeline stage
    Spec {
        /// Stage to run: specify, clarify, plan, tasks, tests, implement, analyze
        stage: String,
    },
    /// Search the indexed codebase
    Search {
        /// Search query
        query: String,
        /// Number of results (default: 10)
        #[arg(short = 'k', default_value = "10")]
        top_k: usize,
    },
    /// Run evaluation suites
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },
}

#[derive(Subcommand)]
enum EvalAction {
    /// Run an evaluation suite
    Run {
        #[arg(long, default_value = "swebench-lite")]
        suite: String,
        #[arg(long, default_value = "2")]
        max_concurrent: usize,
    },
    /// Set a baseline from a run
    Baseline {
        run_id: String,
    },
    /// Diff two runs (or against baseline)
    Diff {
        run_a: String,
        #[arg(default_value = "")]
        run_b: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let cancel = CancellationToken::new();

    // Ctrl-C handler.
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\nInterrupted. Shutting down...");
        cancel_clone.cancel();
    });

    let result = run(cli, cancel).await;
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli, _cancel: CancellationToken) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Chat { workspace } => {
            run_chat(_cancel, workspace).await?;
        }
        Commands::Serve { port } => {
            run_serve(_cancel, port).await?;
        }
        Commands::Index => {
            run_index().await?;
        }
        Commands::Spec { stage } => {
            run_spec(&stage).await?;
        }
        Commands::Search { query, top_k } => {
            run_search(&query, top_k).await?;
        }
        Commands::Eval { action } => match action {
            EvalAction::Run { suite, max_concurrent } => {
                run_eval_run(&suite, max_concurrent).await?;
            }
            EvalAction::Baseline { run_id } => {
                println!("Setting baseline: {run_id}");
                println!("(Copy the run's .jsonl as .agent/evals/baseline.jsonl)");
            }
            EvalAction::Diff { run_a, run_b } => {
                run_eval_diff(&run_a, &run_b)?;
            }
        },
    }
    Ok(())
}

/// Interactive chat loop powered by the configured LLM provider.
async fn run_chat(cancel: CancellationToken, workspace: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve workspace: --workspace flag > current directory.
    let project_root = match workspace {
        Some(ref dir) => std::path::PathBuf::from(dir),
        None => std::env::current_dir()?,
    };
    let project_root = std::fs::canonicalize(&project_root).unwrap_or(project_root);

    // Resolve provider from LLM_PROVIDER env var (default: gemini).
    // Supported: gemini, openai, anthropic, mistral, deepseek, ollama
    let provider_name = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "gemini".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_default();
    // API key resolution: LLM_API_KEY (universal) takes precedence, then the
    // key matching THIS provider only (so a stray key for provider X is never
    // sent to provider Y).
    let provider_key_var = match provider_name.to_lowercase().as_str() {
        "gemini" | "google" => "GEMINI_API_KEY",
        "openai" | "gpt" => "OPENAI_API_KEY",
        "anthropic" | "claude" => "ANTHROPIC_API_KEY",
        "mistral" => "MISTRAL_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        _ => "LLM_API_KEY",
    };
    let api_key = std::env::var("LLM_API_KEY")
        .or_else(|_| std::env::var(provider_key_var))
        .unwrap_or_default();

    let (provider, display_model): (Arc<dyn LlmProvider>, String) = match provider_name.to_lowercase().as_str() {
        "gemini" | "google" => {
            let m = if model.is_empty() { "gemini-3.5-flash".to_string() } else { model };
            let p = GeminiProvider::new(&api_key, &m);
            let p = if std::env::var("GEMINI_USE_AI_STUDIO").unwrap_or_default() == "1" {
                p.with_base_url("https://generativelanguage.googleapis.com/v1beta")
            } else {
                p
            };
            (Arc::new(p), m)
        }
        "openai" | "gpt" => {
            let m = if model.is_empty() { "gpt-5.5".to_string() } else { model };
            let base = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            (Arc::new(llm_client::OpenAiCompatProvider::new(&api_key, &m, &base)), m)
        }
        "anthropic" | "claude" => {
            let m = if model.is_empty() { "claude-fable-5".to_string() } else { model };
            (Arc::new(llm_client::AnthropicProvider::new(&api_key, &m)), m)
        }
        "mistral" => {
            let m = if model.is_empty() { "mistral-medium-3.5".to_string() } else { model };
            (Arc::new(llm_client::OpenAiCompatProvider::new(&api_key, &m, "https://api.mistral.ai/v1")), m)
        }
        "deepseek" => {
            let m = if model.is_empty() { "deepseek-v4-pro".to_string() } else { model };
            (Arc::new(llm_client::OpenAiCompatProvider::new(&api_key, &m, "https://api.deepseek.com")), m)
        }
        "ollama" | "local" => {
            let m = if model.is_empty() { "llama3.3".to_string() } else { model };
            let base = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
            (Arc::new(llm_client::OpenAiCompatProvider::new("", &m, &base)), m)
        }
        other => {
            eprintln!("Unknown provider: '{other}'. Supported: gemini, openai, anthropic, mistral, deepseek, ollama");
            std::process::exit(1);
        }
    };

    if api_key.is_empty() && provider_name != "ollama" && provider_name != "local" {
        eprintln!("Warning: No API key found. Set LLM_API_KEY or provider-specific key (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).");
    }
    // Build hooks. SchemaLangGuard is added only in Hinglish mode to enforce
    // that machine surfaces (paths, commands) stay ASCII while prose can be Hinglish.
    let mut hook_list: Vec<Arc<dyn harness::Hook>> = vec![
        Arc::new(harness::SecretLeakHook::new()),
        Arc::new(harness::DestructiveCommandHook::new()),
    ];
    let lang_mode = std::env::var("LLM_LANG").unwrap_or_else(|_| "hinglish".to_string());
    if lang_mode.eq_ignore_ascii_case("hinglish") {
        hook_list.push(Arc::new(harness::SchemaLangGuard::new()));
    }
    let hooks = Arc::new(HookEngine::new(hook_list));
    // Wire the built-in tools + parallel sub-agent (uses the same provider).
    let mut all_tools = agent_core::default_tools_with_subagent(provider.clone());

    // Load MCP servers from .agent/mcp.json (if present) and add their tools.
    // Format: { "servers": [ { "name": "...", "command": "...", "args": [...] } ] }
    let mcp_config_path = project_root.join(".agent").join("mcp.json");
    let mut mcp_count = 0;
    if mcp_config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&mcp_config_path) {
            if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(servers) = cfg.get("servers").and_then(|s| s.as_array()) {
                    for server in servers {
                        let name = server.get("name").and_then(|v| v.as_str()).unwrap_or("mcp");
                        let command = server.get("command").and_then(|v| v.as_str()).unwrap_or("");
                        let args: Vec<String> = server.get("args")
                            .and_then(|v| v.as_array())
                            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                            .unwrap_or_default();
                        let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                        if command.is_empty() {
                            continue;
                        }
                        match mcp::McpClient::connect(command, &arg_refs, name).await {
                            Ok(client) => {
                                let client = Arc::new(client);
                                let tools = agent_core::mcp_tools(client);
                                mcp_count += tools.len();
                                all_tools.extend(tools);
                                eprintln!("  MCP server '{name}' connected ({} tools)", mcp_count);
                            }
                            Err(e) => eprintln!("  MCP server '{name}' failed: {e}"),
                        }
                    }
                }
            }
        }
    }

    let dispatcher = Arc::new(ToolDispatcher::new(all_tools, hooks));
    let skills = Arc::new(SkillRegistry::load(None).unwrap());
    let event_bus = EventBus::default();

    // Subscribe before handing the bus to the orchestrator, so we can surface
    // tool activity live (Kiro-style progress feedback).
    let mut events = event_bus.subscribe();
    let session_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let session_tokens_clone = session_tokens.clone();
    tokio::spawn(async move {
        use agent_types::AgentEvent;
        while let Ok(event) = events.recv().await {
            match event {
                AgentEvent::TurnStarted => {
                    eprint!("  [thinking...");
                    use std::io::Write;
                    std::io::stderr().flush().ok();
                }
                AgentEvent::Thinking { text } => {
                    // Show thought summary live (truncated to one line)
                    let line = text.lines().next().unwrap_or("").chars().take(80).collect::<String>();
                    eprint!("\r  [thinking: {:<80}]", line);
                    use std::io::Write;
                    std::io::stderr().flush().ok();
                }
                AgentEvent::ToolInvoked { name } => {
                    eprintln!("\r  -> running tool: {name}                                                      ");
                }
                AgentEvent::ToolCompleted { name } => eprintln!("  <- done: {name}"),
                AgentEvent::TokenUsage { prompt_tokens, completion_tokens, total_tokens } => {
                    session_tokens_clone.fetch_add(
                        total_tokens as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    eprintln!(
                        "  [tokens: prompt={prompt_tokens}, output={completion_tokens}, total={total_tokens} | session: {}]",
                        session_tokens_clone.load(std::sync::atomic::Ordering::Relaxed)
                    );
                }
                AgentEvent::TurnEnded => {
                    eprint!("\r                                                                                \r");
                    use std::io::Write;
                    std::io::stderr().flush().ok();
                }
            }
        }
    });

    // Load persistent state (SOUL/HEARTBEAT/MEMORY). Enables cross-session memory.
    let persistent = state_store::PersistentState::load(&project_root).ok();
    let mut memory_context = String::new();
    if let Some(ref p) = persistent {
        let mem = std::fs::read_to_string(project_root.join(".agent").join("MEMORY.md")).unwrap_or_default();
        if !mem.trim().is_empty() {
            // Include the most recent ~2000 chars of long-term memory.
            let tail = if mem.len() > 2000 { &mem[mem.len() - 2000..] } else { &mem };
            memory_context = format!("\n\n## Long-term memory (from previous sessions):\n{tail}\n");
        }
        // Show pending tasks if any.
        let tasks = p.heartbeat_tasks();
        if !tasks.is_empty() {
            memory_context.push_str("\n## Pending tasks:\n");
            for (done, desc) in &tasks {
                memory_context.push_str(&format!("- [{}] {}\n", if *done { "x" } else { " " }, desc));
            }
        }
    }

    let base_prompt = format!(
        "You are an autonomous AI coding agent. Your primary job is to WRITE CODE using tools.\n\
         RULES:\n\
         - When the user asks you to build/create/make something, IMMEDIATELY use write_file to create NEW files. Do NOT just describe what you would do.\n\
         - To MODIFY an existing file, prefer edit_file (str_replace) over write_file — it saves tokens and avoids errors. Only use write_file for new files or full rewrites.\n\
         - Write COMPLETE, working code. Never leave placeholders or TODOs.\n\
         - For large files: write the full file content in a single write_file call. Do not split across turns.\n\
         - Keep explanations SHORT (2-3 lines max) AFTER writing the files.\n\
         - If a task needs multiple files, write ALL of them in the same turn using multiple write_file calls.\n\
         - SELF-HEAL: After editing code, run the check_code tool to verify it compiles. If there are errors, read them, fix the code with edit_file, and check again. Repeat until clean.\n\
         - Always use tools. Never refuse to write code.{memory_context}"
    );

    let mut orchestrator = Orchestrator::new(
        provider,
        dispatcher,
        skills,
        event_bus,
        cancel.clone(),
        if lang_mode.eq_ignore_ascii_case("hinglish") {
            agent_types::LanguageMode::Hinglish
        } else {
            agent_types::LanguageMode::En
        },
    )
    .with_project_root(project_root.clone())
    .with_system_prompt(base_prompt);

    println!("========================================");
    println!(" Rust AI Coding Agent");
    println!(" provider: {provider_name}");
    println!(" model: {display_model}");
    println!(" workspace: {}", project_root.display());
    println!(" tools: read_file, write_file, edit_file, list_files, search_text,");
    println!("        bash, web_fetch, check_code, dispatch_subagent");
    if !memory_context.is_empty() {
        println!(" memory: loaded from .agent/MEMORY.md");
    }
    // Detect if workspace is a git repo (for checkpoint/undo support).
    let git_available = is_git_repo(&project_root);
    if git_available {
        println!(" checkpoints: enabled (/undo to revert last turn)");
    }
    println!(" Commands: /remember <text>, /undo, /quit");
    println!("========================================");
    println!();

    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);

    loop {
        if cancel.is_cancelled() {
            break;
        }

        eprint!("You> ");
        // Flush stderr since eprint doesn't auto-flush
        use std::io::Write;
        std::io::stderr().flush().ok();

        let mut line = String::new();
        use tokio::io::AsyncBufReadExt;
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }

        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }

        // Special commands
        if input == "/quit" || input == "/exit" {
            println!("Bye!");
            break;
        }

        // /remember <text> — save to long-term memory (persists across sessions)
        if let Some(mem_text) = input.strip_prefix("/remember ") {
            if let Some(ref p) = persistent {
                match p.append_memory(mem_text.trim()) {
                    Ok(()) => println!("Saved to memory (.agent/MEMORY.md)\n"),
                    Err(e) => eprintln!("Failed to save memory: {e}\n"),
                }
            } else {
                eprintln!("Memory not available in this workspace.\n");
            }
            continue;
        }

        // /undo — revert file changes from the last agent turn (git checkpoint)
        if input == "/undo" {
            if git_available {
                match git_undo_last_checkpoint(&project_root) {
                    Ok(msg) => println!("{msg}\n"),
                    Err(e) => eprintln!("Undo failed: {e}\n"),
                }
            } else {
                eprintln!("Undo needs a git repository. Run `git init` in the workspace.\n");
            }
            continue;
        }

        // Create a checkpoint BEFORE the turn so /undo can revert it.
        if git_available {
            let _ = git_checkpoint(&project_root, &input);
        }

        match orchestrator.run_turn(input).await {
            Ok(response) => {
                println!("\nAgent> {response}\n");
            }
            Err(e) => {
                eprintln!("\n[Error: {e}]\n");
            }
        }
    }

    Ok(())
}

// ===========================================================================
// INDEX command: Merkle diff → parse → chunk → store (keyword-only, no embeddings)
// ===========================================================================

async fn run_index() -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let agent_dir = cwd.join(".agent");
    std::fs::create_dir_all(&agent_dir)?;

    let merkle_path = agent_dir.join("index.merkle");
    let db_path = agent_dir.join("index.db");

    println!("Indexing codebase at: {}", cwd.display());

    // 1. Build current Merkle tree.
    let current_tree = indexer::MerkleTree::build(&cwd)?;
    println!("  Files found: {}", current_tree.nodes.len());

    // 2. Diff against previous tree (if exists).
    let changed_paths = if merkle_path.exists() {
        let old_tree = indexer::MerkleTree::load(&merkle_path)?;
        let diff = old_tree.diff(&current_tree);
        println!("  Changed since last index: {} files", diff.len());
        diff
    } else {
        println!("  First index — processing all files.");
        current_tree.nodes.keys().cloned().collect::<Vec<_>>()
    };

    if changed_paths.is_empty() {
        println!("Index up to date. 0 files changed.");
        return Ok(());
    }

    // 3. Open vector store.
    let store = vecstore::VecStore::open(&db_path)?;

    // 4. Setup embedder (if API key available → real embeddings; else → keyword only).
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
    let embedder = if !api_key.is_empty() {
        let base_url = if std::env::var("GEMINI_USE_AI_STUDIO").unwrap_or_default() == "1" {
            "https://generativelanguage.googleapis.com/v1beta".to_string()
        } else {
            "https://aiplatform.googleapis.com/v1/publishers/google".to_string()
        };
        println!("  Embedding mode: SEMANTIC (text-embedding-004)");
        Some(llm_client::GeminiEmbedder::new(api_key, base_url))
    } else {
        println!("  Embedding mode: KEYWORD ONLY (set GEMINI_API_KEY for semantic search)");
        None
    };

    // 5. Parse + chunk + embed + insert each changed file.
    let mut total_chunks = 0u64;
    let mut errors = 0u64;
    for path in &changed_paths {
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue, // skip binary/unreadable
        };

        let entities = indexer::parse(path, &source)?;
        if entities.is_empty() {
            continue; // unsupported language
        }

        let chunks = indexer::chunk(&entities, &source, 512, 1);
        let rel = path.strip_prefix(&cwd).unwrap_or(path);
        eprint!("  {} ({} chunks)...", rel.display(), chunks.len());

        let mut inserts: Vec<vecstore::ChunkInsert> = Vec::new();
        for c in &chunks {
            let embedding = if let Some(ref emb) = embedder {
                // Real embedding from Gemini
                match emb.embed(&c.text).await {
                    Ok(v) => v,
                    Err(e) => {
                        errors += 1;
                        if errors <= 3 {
                            eprintln!("\n    Warning: embed failed: {e}");
                        }
                        vec![0.0; 768] // fallback
                    }
                }
            } else {
                vec![0.0; 768] // keyword-only placeholder
            };

            inserts.push(vecstore::ChunkInsert {
                file_path: rel.to_string_lossy().to_string(),
                start_line: c.start_line,
                end_line: c.end_line,
                text: c.text.clone(),
                token_count: c.token_count,
                embedding,
            });
        }

        store.upsert_file(
            &rel.to_string_lossy(),
            std::fs::metadata(path)
                .map(|m| {
                    m.modified()
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64
                })
                .unwrap_or(0),
            "",
        )?;
        store.insert_chunks(&inserts)?;
        total_chunks += inserts.len() as u64;
        eprintln!(" ok");
    }

    // 6. Save updated Merkle tree.
    current_tree.save(&merkle_path)?;

    let (chunks, _vecs, _fts) = store.chunk_counts()?;
    println!("\nIndex complete. {} files processed, {} new chunks (total in DB: {}).",
        changed_paths.len(), total_chunks, chunks);
    if errors > 0 {
        println!("  ({errors} embedding errors — those chunks use keyword-only mode)");
    }

    Ok(())
}

// ===========================================================================
// SEARCH command: keyword (BM25) search over the indexed codebase
// ===========================================================================

async fn run_search(query: &str, top_k: usize) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let db_path = cwd.join(".agent").join("index.db");

    if !db_path.exists() {
        eprintln!("Error: No index found. Run `cli index` first.");
        std::process::exit(1);
    }

    let store = vecstore::VecStore::open(&db_path)?;

    // Try semantic search if API key is available, fallback to keyword.
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
    let (hits, mode_name) = if !api_key.is_empty() {
        let base_url = if std::env::var("GEMINI_USE_AI_STUDIO").unwrap_or_default() == "1" {
            "https://generativelanguage.googleapis.com/v1beta"
        } else {
            "https://aiplatform.googleapis.com/v1/publishers/google"
        };
        let embedder = llm_client::GeminiEmbedder::new(&api_key, base_url);

        // Await the query embedding directly (no nested block_on — that panics
        // inside an already-running tokio runtime).
        match embedder.embed(query).await {
            Ok(query_emb) => {
                let hits = vecstore::search(
                    &store, query, Some(&query_emb), &[],
                    vecstore::SearchMode::Hybrid, top_k,
                )?;
                (hits, "hybrid (semantic + keyword)")
            }
            Err(_) => {
                // Fallback to keyword if embedding fails
                let hits = vecstore::search(
                    &store, query, None, &[],
                    vecstore::SearchMode::Keyword, top_k,
                )?;
                (hits, "keyword (embedding failed, fallback)")
            }
        }
    } else {
        let hits = vecstore::search(
            &store, query, None, &[],
            vecstore::SearchMode::Keyword, top_k,
        )?;
        (hits, "keyword only")
    };

    if hits.is_empty() {
        println!("No results for: \"{query}\" [mode: {mode_name}]");
        return Ok(());
    }

    println!("Search results for: \"{query}\" (top {top_k}, mode: {mode_name})\n");
    for (i, hit) in hits.iter().enumerate() {
        println!("{}. {} (lines {}-{}, score: {:.3})",
            i + 1, hit.file_path, hit.start_line, hit.end_line, hit.score);
        let preview: String = hit.text.lines().take(2).collect::<Vec<_>>().join("\n");
        println!("   {preview}");
        println!();
    }

    Ok(())
}

// ===========================================================================
// SPEC command: run a RustySpec pipeline stage
// ===========================================================================

async fn run_spec(stage_str: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;

    let stage = match stage_str.to_lowercase().as_str() {
        "specify" => spec_pipeline::Stage::Specify,
        "clarify" => spec_pipeline::Stage::Clarify,
        "plan" => spec_pipeline::Stage::Plan,
        "tasks" => spec_pipeline::Stage::Tasks,
        "tests" => spec_pipeline::Stage::Tests,
        "implement" => spec_pipeline::Stage::Implement,
        "analyze" => spec_pipeline::Stage::Analyze,
        _ => {
            eprintln!("Unknown stage: '{stage_str}'. Valid: specify, clarify, plan, tasks, tests, implement, analyze");
            std::process::exit(1);
        }
    };

    // Use "default" session for now.
    let pipeline = spec_pipeline::Pipeline::new(&cwd, "default")?;

    // Check prerequisites.
    if let Err(e) = pipeline.check_prerequisites(stage) {
        eprintln!("Prerequisites not met: {e}");
        eprintln!("Run earlier stages first.");
        std::process::exit(1);
    }

    // Build prompt.
    let user_context = if stage == spec_pipeline::Stage::Specify {
        // For specify, read from stdin or ask user.
        println!("Describe what you want to build (end with empty line):");
        let mut input = String::new();
        let stdin = std::io::stdin();
        loop {
            let mut line = String::new();
            use std::io::BufRead;
            stdin.lock().read_line(&mut line)?;
            if line.trim().is_empty() {
                break;
            }
            input.push_str(&line);
        }
        input
    } else {
        format!("Continue from prior artifacts for stage: {stage_str}")
    };

    let prompt = pipeline.build_prompt(stage, &user_context)?;

    // Send to Gemini and get response.
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        eprintln!("Error: GEMINI_API_KEY not set.");
        std::process::exit(1);
    }
    let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-3.5-flash".to_string());
    let provider = GeminiProvider::new(&api_key, &model);

    let cancel = CancellationToken::new();
    let messages = vec![agent_types::Message {
        role: agent_types::Role::User,
        content: vec![agent_types::ContentBlock::Text(prompt)],
        token_estimate: 0,
    }];

    println!("Running stage: {stage_str}...");
    let mut rx = provider.stream(&messages, &[], &cancel).await
        .map_err(|e| format!("LLM error: {e}"))?;

    let mut response_text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            llm_client::SseEvent::Delta(d) => response_text.push_str(&d),
            llm_client::SseEvent::Stop { .. } => break,
            llm_client::SseEvent::Error(e) => {
                eprintln!("LLM error: {e}");
                std::process::exit(1);
            }
            _ => {}
        }
    }

    // Write artifact.
    let artifact_path = pipeline.write_artifact(stage, &response_text).await?;
    println!("Artifact written: {}", artifact_path.display());
    println!("\n--- Preview (first 20 lines) ---");
    for line in response_text.lines().take(20) {
        println!("{line}");
    }

    Ok(())
}

// ===========================================================================
// EVAL commands
// ===========================================================================

async fn run_eval_run(suite: &str, _max_concurrent: usize) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let cases_dir = cwd.join(".agent").join("evals").join("cases");

    if !cases_dir.exists() {
        eprintln!("No eval cases found at: {}", cases_dir.display());
        eprintln!("Create .agent/evals/cases/*.toml files with EvalCase format.");
        std::process::exit(1);
    }

    let cases = evals::swebench::load_cases(&cases_dir)?;
    println!("Loaded {} eval cases from suite '{suite}'", cases.len());

    if cases.is_empty() {
        println!("No cases to run.");
        return Ok(());
    }

    let run_id = chrono_stub_now();
    let results_path = cwd.join(".agent").join("evals").join("results").join(format!("{run_id}.jsonl"));

    println!("Run ID: {run_id}");
    println!("Results will be written to: {}", results_path.display());
    println!();

    // Run each case ONCE, collect outcomes.
    let mut outcomes: Vec<evals::swebench::EvalOutcome> = Vec::new();
    for case in &cases {
        println!("Running case: {} ...", case.id);
        let start = std::time::Instant::now();
        let passed = run_check_cmd(&case.check_cmd, &case.repo_fixture, case.timeout_secs);
        let elapsed = start.elapsed().as_millis() as u64;

        let outcome = evals::swebench::EvalOutcome {
            case_id: case.id.clone(),
            passed,
            turns: 0,
            tool_calls: 0,
            tokens_in: 0,
            tokens_out: 0,
            wall_time_ms: elapsed,
            error: if passed { None } else { Some("check_cmd failed".into()) },
        };

        let status = if passed { "PASS" } else { "FAIL" };
        println!("  {status} ({elapsed}ms)");
        evals::swebench::append_outcome(&results_path, &outcome)?;
        outcomes.push(outcome);
    }

    // Print summary from the SAME outcomes (no re-execution).
    println!();
    let report = evals::report::EvalReport::new(outcomes);
    report.print_summary();

    Ok(())
}

fn run_eval_diff(run_a: &str, run_b: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let results_dir = cwd.join(".agent").join("evals").join("results");

    let load_outcomes = |run_id: &str| -> Result<Vec<evals::swebench::EvalOutcome>, Box<dyn std::error::Error>> {
        let path = results_dir.join(format!("{run_id}.jsonl"));
        if !path.exists() {
            return Err(format!("Run not found: {}", path.display()).into());
        }
        let content = std::fs::read_to_string(&path)?;
        let outcomes: Vec<evals::swebench::EvalOutcome> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        Ok(outcomes)
    };

    let baseline = load_outcomes(run_a)?;
    let current = load_outcomes(run_b)?;

    let diff = evals::trajectory::diff_runs(&baseline, &current);

    println!("=== Eval Diff: {run_a} vs {run_b} ===\n");

    if diff.hard_regressions.is_empty() && diff.soft_regressions.is_empty() {
        println!("No regressions detected. All clear!");
    } else {
        if !diff.hard_regressions.is_empty() {
            println!("HARD REGRESSIONS (pass -> fail):");
            for r in &diff.hard_regressions {
                println!("  FAIL: {r}");
            }
            println!();
        }
        if !diff.soft_regressions.is_empty() {
            println!("Soft regressions (performance):");
            for r in &diff.soft_regressions {
                println!("  WARN: {r}");
            }
        }
    }

    if diff.has_hard_regression() {
        std::process::exit(1); // CI gate: non-zero exit on hard regression
    }

    Ok(())
}

/// Run a check command in a directory, return true if exit code 0.
fn run_check_cmd(cmd: &str, cwd: &std::path::Path, _timeout_secs: u64) -> bool {
    use std::process::Command;
    // Cross-platform shell selection.
    #[cfg(windows)]
    let result = Command::new("cmd.exe")
        .args(["/C", cmd])
        .current_dir(cwd)
        .output();
    #[cfg(not(windows))]
    let result = Command::new("sh")
        .args(["-c", cmd])
        .current_dir(cwd)
        .output();
    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Simple timestamp for run IDs (no chrono dependency).
fn chrono_stub_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("run-{secs}")
}

// ===========================================================================
// SERVE command: IPC server for editor integration
// ===========================================================================

/// Start the IPC server that a VS Code extension (or any editor plugin) can
/// connect to. Uses newline-delimited JSON PatchMessages over TCP loopback.
/// This is the backend for future editor integration.
async fn run_serve(cancel: CancellationToken, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!(" Rust AI Coding Agent — IPC Server Mode");
    println!(" Listening on: 127.0.0.1:{port}");
    println!(" Protocol: newline-delimited JSON (PatchMessage)");
    println!(" For editor integration (VS Code extension).");
    println!(" Ctrl-C to stop.");
    println!("========================================");

    let server = apply_engine::IpcServer::bind(port).await?;
    server.run().await?;

    println!("Server running. Waiting for editor connections...");

    // Keep alive until cancelled.
    cancel.cancelled().await;
    println!("\nShutting down IPC server.");
    Ok(())
}

// ===========================================================================
// Git checkpoints (/undo support)
// ===========================================================================

/// Check whether `dir` is inside a git working tree.
fn is_git_repo(dir: &std::path::Path) -> bool {
    std::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(dir)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Create a checkpoint commit of the current state BEFORE an agent turn.
/// Uses a dedicated commit so `/undo` can restore the pre-turn state without
/// touching the user's own commit history destructively.
fn git_checkpoint(dir: &std::path::Path, user_msg: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    // Stage everything (including untracked) and make a checkpoint commit.
    // If there's nothing to commit, git commit returns non-zero — that's fine.
    Command::new("git").args(["add", "-A"]).current_dir(dir).output()?;
    let short_msg: String = user_msg.chars().take(50).collect();
    Command::new("git")
        .args(["commit", "-m", &format!("[agent-checkpoint] {short_msg}"), "--no-verify"])
        .current_dir(dir)
        .output()?;
    Ok(())
}

/// Revert the changes made since the last agent checkpoint. Restores the
/// working tree to the checkpoint state (the pre-turn snapshot).
fn git_undo_last_checkpoint(dir: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    use std::process::Command;

    // Find the most recent checkpoint commit.
    let log = Command::new("git")
        .args(["log", "--grep=\\[agent-checkpoint\\]", "--format=%H %s", "-n", "1"])
        .current_dir(dir)
        .output()?;
    let log_str = String::from_utf8_lossy(&log.stdout);
    let line = log_str.trim();
    if line.is_empty() {
        return Ok("No agent checkpoint found to undo.".to_string());
    }

    // The checkpoint commit IS the pre-turn state. Reset the working tree to it,
    // keeping the checkpoint as the current state (so files match pre-turn).
    let hash = line.split_whitespace().next().unwrap_or("");
    if hash.is_empty() {
        return Ok("Could not parse checkpoint.".to_string());
    }

    // Hard reset to the checkpoint commit — restores files to pre-turn snapshot.
    let out = Command::new("git")
        .args(["reset", "--hard", hash])
        .current_dir(dir)
        .output()?;
    if out.status.success() {
        Ok(format!("Reverted to checkpoint {}. Agent's last changes undone.", &hash[..hash.len().min(8)]))
    } else {
        Err(format!("git reset failed: {}", String::from_utf8_lossy(&out.stderr)).into())
    }
}
