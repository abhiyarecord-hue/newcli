//! `cli` (L5): clap-based CLI entrypoint.
//!
//! Subcommands: chat, index, spec, search, eval.
//! Ctrl-C cancels the root CancellationToken → graceful drain.

mod ui;

use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;

use agent_core::{Orchestrator, ToolDispatcher};
use harness::{HookEngine, SkillRegistry};
use llm_client::GeminiProvider;
use llm_client::LlmProvider;
use runtime_core::EventBus;

#[derive(Parser)]
#[command(name = "srijandev", version, about = "Srijan Dev — AI-Powered Autonomous Coding Agent")]
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
        /// Workspace directory (default: current directory).
        #[arg(short = 'w', long = "workspace")]
        workspace: Option<String>,
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
        Commands::Spec { stage, workspace } => {
            run_spec(&stage, workspace).await?;
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

/// Resolve provider name + model + API key from the environment.
/// Supported: gemini, openai, anthropic, mistral, deepseek, ollama.
fn resolve_provider_config() -> (String, String, String) {
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
    (provider_name, model, api_key)
}

/// Construct a boxed [`LlmProvider`] from resolved config. Shared by the
/// interactive chat loop and the RustySpec pipeline so every entry point
/// respects `LLM_PROVIDER`/`LLM_API_KEY` instead of hardcoding one backend.
fn build_provider(
    provider_name: &str,
    model: String,
    api_key: &str,
) -> Result<(Arc<dyn LlmProvider>, String), Box<dyn std::error::Error>> {
    let (provider, display_model): (Arc<dyn LlmProvider>, String) = match provider_name.to_lowercase().as_str() {
        "gemini" | "google" => {
            let m = if model.is_empty() { "gemini-3.5-flash".to_string() } else { model };
            let p = GeminiProvider::new(api_key, &m);
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
            (Arc::new(llm_client::OpenAiCompatProvider::new(api_key, &m, &base)), m)
        }
        "anthropic" | "claude" => {
            let m = if model.is_empty() { "claude-fable-5".to_string() } else { model };
            (Arc::new(llm_client::AnthropicProvider::new(api_key, &m)), m)
        }
        "mistral" => {
            let m = if model.is_empty() { "mistral-medium-3.5".to_string() } else { model };
            (Arc::new(llm_client::OpenAiCompatProvider::new(api_key, &m, "https://api.mistral.ai/v1")), m)
        }
        "deepseek" => {
            let m = if model.is_empty() { "deepseek-v4-pro".to_string() } else { model };
            (Arc::new(llm_client::OpenAiCompatProvider::new(api_key, &m, "https://api.deepseek.com")), m)
        }
        "ollama" | "local" => {
            let m = if model.is_empty() { "llama3.3".to_string() } else { model };
            let base = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
            (Arc::new(llm_client::OpenAiCompatProvider::new("", &m, &base)), m)
        }
        other => {
            return Err(format!(
                "Unknown provider: '{other}'. Supported: gemini, openai, anthropic, mistral, deepseek, ollama"
            ).into());
        }
    };
    Ok((provider, display_model))
}

/// Interactive chat loop powered by the configured LLM provider.
async fn run_chat(cancel: CancellationToken, workspace: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve workspace: --workspace flag > current directory.
    let project_root = match workspace {
        Some(ref dir) => std::path::PathBuf::from(dir),
        None => std::env::current_dir()?,
    };
    let project_root = std::fs::canonicalize(&project_root).unwrap_or(project_root);

    let (provider_name, model, api_key) = resolve_provider_config();
    let (provider, display_model) = match build_provider(&provider_name, model, &api_key) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if api_key.is_empty() && provider_name != "ollama" && provider_name != "local" {
        eprintln!("Warning: No API key found. Set LLM_API_KEY or provider-specific key (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY).");
    }

    // Ask the user to pick Vibe (this free-flow loop) or RustySpec (the
    // structured 7-stage workflow) before building the rest of the session.
    ui::mode_select();
    let mut mode_line = String::new();
    std::io::stdin().read_line(&mut mode_line).ok();
    if mode_line.trim() == "2" {
        return run_rustyspec_session(cancel, project_root, provider, provider_name, display_model, api_key).await;
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
                                let server_tool_count = tools.len();
                                all_tools.extend(tools);
                                eprintln!("  MCP server '{name}' connected ({server_tool_count} tools)");
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

    // Subscribe before handing the bus to the orchestrator. A completed usage
    // snapshot is sent back to the chat loop so response and accounting render
    // in a deterministic order.
    let mut events = event_bus.subscribe();
    let (usage_tx, mut usage_rx) = tokio::sync::mpsc::unbounded_channel::<ui::UsageStats>();
    tokio::spawn(async move {
        use agent_types::AgentEvent;
        let mut stats = ui::UsageStats::default();
        while let Ok(event) = events.recv().await {
            match event {
                AgentEvent::TurnStarted => {
                    stats.start_turn();
                    ui::turn_started();
                }
                AgentEvent::ApiCallStarted => {
                    stats.api_call();
                    ui::api_call(stats.turn_calls);
                }
                AgentEvent::Thinking { text } => ui::thinking(&text),
                AgentEvent::ToolInvoked { name } => ui::tool_started(&name),
                AgentEvent::ToolCompleted { name } => ui::tool_done(&name),
                AgentEvent::TokenUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                } => stats.add_tokens(prompt_tokens, completion_tokens, total_tokens),
                AgentEvent::TurnEnded => {
                    ui::turn_ended();
                    let _ = usage_tx.send(stats.clone());
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
            // Include the most recent 2000 Unicode scalar values of long-term
            // memory without slicing through a UTF-8 code point.
            let tail_start = mem
                .char_indices()
                .rev()
                .nth(1999)
                .map(|(index, _)| index)
                .unwrap_or(0);
            let tail = &mem[tail_start..];
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

    // Load chat history from previous sessions.
    let chat_history = state_store::ChatHistory::open(&project_root)
        .ok();
    let prior_history = chat_history
        .as_ref()
        .and_then(|h| h.load(Some(60)).ok())
        .unwrap_or_default();
    let prior_count = prior_history.len();

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
    .with_system_prompt(base_prompt)
    .with_history(prior_history);

    // Track how many messages were in history before each turn so we can
    // persist only the new ones.
    let mut history_persisted_up_to = prior_count;

    // Detect git for checkpoint/undo support. Only an exact checkpoint created
    // for the current turn may be used by `/undo`.
    let git_available = is_git_repo(&project_root);
    let mut last_checkpoint: Option<GitCheckpoint> = None;

    ui::banner(
        &provider_name,
        &display_model,
        &project_root.display().to_string(),
        !api_key.is_empty() || matches!(provider_name.as_str(), "ollama" | "local"),
        !memory_context.is_empty() || prior_count > 0,
        git_available,
    );

    if prior_count > 0 {
        eprintln!("  {} restored {} messages from previous session\n",
            "\x1b[38;5;45m↻\x1b[0m", prior_count);
    }

    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);

    loop {
        if cancel.is_cancelled() {
            break;
        }

        ui::prompt_start();

        let mut line = String::new();
        use tokio::io::AsyncBufReadExt;
        let read_result = reader.read_line(&mut line).await;
        ui::prompt_end();
        match read_result {
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

        // /clear — reset chat history for this workspace
        if input == "/clear" {
            if let Some(ref h) = chat_history {
                let _ = h.clear();
            }
            history_persisted_up_to = 0;
            println!("Chat history cleared. Starting fresh.\n");
            continue;
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
                match last_checkpoint.as_ref() {
                    Some(checkpoint) => match git_undo_last_checkpoint(&project_root, checkpoint) {
                        Ok(msg) => {
                            last_checkpoint = None;
                            println!("{msg}\n");
                        }
                        Err(e) => eprintln!("Undo failed: {e}\n"),
                    },
                    None => eprintln!(
                        "No successful checkpoint exists for the last agent turn; refusing undo.\n"
                    ),
                }
            } else {
                eprintln!("Undo needs a git repository. Run `git init` in the workspace.\n");
            }
            continue;
        }

        // Create a checkpoint BEFORE the turn so /undo can revert exactly this
        // turn. A failed checkpoint explicitly disables undo rather than falling
        // back to an older commit.
        if git_available {
            match git_checkpoint(&project_root, &input) {
                Ok(checkpoint) => last_checkpoint = Some(checkpoint),
                Err(e) => {
                    last_checkpoint = None;
                    eprintln!("Warning: checkpoint failed; /undo disabled for this turn: {e}");
                }
            }
        }

        let turn_result = orchestrator.run_turn(input).await;

        // Persist new messages added during this turn.
        if let Some(ref history) = chat_history {
            let new_msgs = orchestrator.history_since(history_persisted_up_to);
            if !new_msgs.is_empty() {
                let _ = history.append(new_msgs);
                history_persisted_up_to += new_msgs.len();
            }
        }

        let usage = usage_rx.recv().await.unwrap_or_default();
        match turn_result {
            Ok(response) => ui::answer(&response),
            Err(e) => ui::error(&e.to_string()),
        }
        ui::usage(&usage);
        println!();
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
        current_tree
            .nodes
            .keys()
            .filter(|rel| cwd.join(rel).is_file())
            .cloned()
            .collect::<Vec<_>>()
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

    // 5. Delete stale rows, then parse + chunk + embed each changed file.
    let mut total_chunks = 0u64;
    let mut errors = 0u64;
    for rel in &changed_paths {
        let path = cwd.join(rel);
        let rel_str = rel.to_string_lossy().to_string();

        // A changed file must replace its old rows; a deleted/unsupported/
        // unreadable file must not remain searchable from a stale index.
        store.delete_file(&rel_str)?;
        if !path.is_file() {
            continue;
        }

        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  Skipping unreadable {}: {e}", rel.display());
                continue;
            }
        };

        let entities = indexer::parse(&path, &source)?;
        if entities.is_empty() {
            continue; // unsupported language or no indexable entities
        }

        let chunks = indexer::chunk(&entities, &source, 512, 1);
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
                file_path: rel_str.clone(),
                start_line: c.start_line,
                end_line: c.end_line,
                text: c.text.clone(),
                token_count: c.token_count,
                embedding,
            });
        }

        store.upsert_file(
            &rel_str,
            std::fs::metadata(&path)
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

async fn run_spec(stage_str: &str, workspace: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let project_root = match workspace {
        Some(ref dir) => std::path::PathBuf::from(dir),
        None => std::env::current_dir()?,
    };
    let project_root = std::fs::canonicalize(&project_root).unwrap_or(project_root);

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
    let pipeline = spec_pipeline::Pipeline::new(&project_root, "default")?;

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

    // Resolve the provider the same way `chat` does — honors LLM_PROVIDER /
    // LLM_API_KEY instead of hardcoding Gemini.
    let (provider_name, model, api_key) = resolve_provider_config();
    let (provider, display_model) = build_provider(&provider_name, model, &api_key)?;
    if api_key.is_empty() && provider_name != "ollama" && provider_name != "local" {
        eprintln!("Warning: No API key found for provider '{provider_name}'.");
    }
    println!("Running stage: {stage_str} (provider: {provider_name}, model: {display_model})...");

    if stage == spec_pipeline::Stage::Implement {
        // The Implement stage must produce real files, not a text dump. Run a
        // full orchestrator turn with the actual tool set so write_file/edit_file
        // execute against the workspace, then log a short summary artifact.
        run_spec_implement(&project_root, provider, prompt, pipeline).await?;
        return Ok(());
    }

    let cancel = CancellationToken::new();
    let response_text = stream_provider_text(provider, prompt, &cancel).await
        .map_err(|e| format!("LLM error: {e}"))?;

    // Write artifact.
    let artifact_path = pipeline.write_artifact(stage, &response_text).await?;
    println!("Artifact written: {}", artifact_path.display());
    println!("\n--- Preview (first 20 lines) ---");
    for line in response_text.lines().take(20) {
        println!("{line}");
    }

    Ok(())
}

/// Interactive RustySpec session: guides the user through the 7 stages in
/// order, reusing the provider resolved for this process (honors
/// LLM_PROVIDER/LLM_API_KEY, same as Vibe chat).
async fn run_rustyspec_session(
    cancel: CancellationToken,
    project_root: std::path::PathBuf,
    provider: Arc<dyn LlmProvider>,
    provider_name: String,
    display_model: String,
    api_key: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let stages = spec_pipeline::Stage::all();
    let session_id = "default";
    let pipeline = spec_pipeline::Pipeline::new(&project_root, session_id)?;

    ui::banner(
        &provider_name,
        &display_model,
        &project_root.display().to_string(),
        !api_key.is_empty() || matches!(provider_name.as_str(), "ollama" | "local"),
        false,
        is_git_repo(&project_root),
    );
    println!("  \x1b[1;38;5;214mRustySpec mode\x1b[0m — structured 7-stage workflow");
    println!("  Stages: specify → clarify → plan → tasks → tests → implement → analyze");
    println!("  Commands: /status  /chat  /rerun <stage>  /quit\n");

    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);

    loop {
        if cancel.is_cancelled() {
            break;
        }

        // Find the next stage whose artifact doesn't exist yet, to guide the
        // user through the pipeline in order.
        let next_stage = stages
            .iter()
            .find(|s| !pipeline.session_dir().join(s.artifact()).exists());

        match next_stage {
            Some(stage) => eprint!("\x1b[1;36m❯\x1b[0m next stage [{stage:?}] — run it? (y/n/status/chat/quit): "),
            None => eprint!("\x1b[1;36m❯\x1b[0m all stages complete. (status/chat/rerun <stage>/quit): "),
        }
        use std::io::Write;
        std::io::stderr().flush().ok();

        let mut line = String::new();
        use tokio::io::AsyncBufReadExt;
        if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
            break;
        }
        let input = line.trim().to_string();
        let input_lower = input.to_lowercase();

        if input_lower == "/quit" || input_lower == "quit" || input_lower == "/exit" {
            println!("Bye!");
            break;
        }
        if input_lower == "/status" || input_lower == "status" {
            for s in stages {
                let done = pipeline.session_dir().join(s.artifact()).exists();
                println!("  [{}] {s:?}", if done { "x" } else { " " });
            }
            println!();
            continue;
        }

        // /chat — drop into a Vibe-style follow-up session with the same
        // workspace/provider/tools. This is how bugs found after Implement
        // ("start button doesn't work, fix it") get fixed: the agent reads and
        // edits real files here instead of being stuck picking stages.
        if input_lower == "/chat" || input_lower == "chat" {
            println!("Switching to free-flow chat for this workspace. Type /back to return to RustySpec.\n");
            run_rustyspec_followup_chat(&cancel, &project_root, provider.clone(), &mut reader).await?;
            println!();
            continue;
        }

        // /rerun <stage> — delete an existing artifact and redo that stage
        // (e.g. re-run Implement after Plan/Tasks changed).
        if let Some(rest) = input_lower.strip_prefix("/rerun ").or_else(|| input_lower.strip_prefix("rerun ")) {
            let target = stages.iter().find(|s| format!("{:?}", s).to_lowercase() == rest.trim());
            match target {
                Some(stage) => {
                    let path = pipeline.session_dir().join(stage.artifact());
                    if path.is_dir() {
                        let _ = tokio::fs::remove_dir_all(&path).await;
                    } else {
                        let _ = tokio::fs::remove_file(&path).await;
                    }
                    println!("Cleared {stage:?} artifact. It will run again next.\n");
                }
                None => eprintln!("Unknown stage '{rest}'. Valid: specify, clarify, plan, tasks, tests, implement, analyze\n"),
            }
            continue;
        }

        let stage = match next_stage {
            Some(s) => *s,
            None => continue,
        };
        if input_lower != "y" && input_lower != "yes" {
            continue;
        }

        let user_context = if stage == spec_pipeline::Stage::Specify {
            println!("Describe what you want to build (end with empty line):");
            let mut ctx = String::new();
            loop {
                let mut l = String::new();
                if reader.read_line(&mut l).await.unwrap_or(0) == 0 || l.trim().is_empty() {
                    break;
                }
                ctx.push_str(&l);
            }
            ctx
        } else {
            format!("Continue from prior artifacts for stage: {stage:?}")
        };

        let prompt = match pipeline.build_prompt(stage, &user_context) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Prerequisites not met: {e}\n");
                continue;
            }
        };

        println!("Running stage: {stage:?}...");

        if stage == spec_pipeline::Stage::Implement {
            let pipeline_for_impl = spec_pipeline::Pipeline::new(&project_root, session_id)?;
            if let Err(e) = run_spec_implement(&project_root, provider.clone(), prompt, pipeline_for_impl).await {
                eprintln!("Implement stage failed: {e}\n");
            }
            println!();
            continue;
        }

        let response_text = match stream_provider_text(provider.clone(), prompt, &cancel).await {
            Ok(t) => t,
            Err(e) => {
                eprintln!("LLM error: {e}\n");
                continue;
            }
        };

        match pipeline.write_artifact(stage, &response_text).await {
            Ok(path) => println!("Artifact written: {}\n", path.display()),
            Err(e) => eprintln!("Failed to write artifact: {e}\n"),
        }
    }

    Ok(())
}

/// A short free-flow chat loop reachable from inside RustySpec via `/chat`,
/// so bugs discovered after Implement ("start button doesn't work") can be
/// fixed with real tool calls without leaving the spec session. Returns to
/// the caller on `/back`, EOF, or cancellation.
async fn run_rustyspec_followup_chat(
    cancel: &CancellationToken,
    project_root: &std::path::Path,
    provider: Arc<dyn LlmProvider>,
    reader: &mut tokio::io::BufReader<tokio::io::Stdin>,
) -> Result<(), Box<dyn std::error::Error>> {
    let hook_list: Vec<Arc<dyn harness::Hook>> = vec![
        Arc::new(harness::SecretLeakHook::new()),
        Arc::new(harness::DestructiveCommandHook::new()),
    ];
    let hooks = Arc::new(HookEngine::new(hook_list));
    let tools = agent_core::default_tools_with_subagent(provider.clone());
    let dispatcher = Arc::new(ToolDispatcher::new(tools, hooks));
    let skills = Arc::new(SkillRegistry::load(None).unwrap());
    let event_bus = EventBus::default();

    let mut events = event_bus.subscribe();
    let (usage_tx, mut usage_rx) = tokio::sync::mpsc::unbounded_channel::<ui::UsageStats>();
    tokio::spawn(async move {
        use agent_types::AgentEvent;
        let mut stats = ui::UsageStats::default();
        while let Ok(event) = events.recv().await {
            match event {
                AgentEvent::TurnStarted => {
                    stats.start_turn();
                    ui::turn_started();
                }
                AgentEvent::ApiCallStarted => {
                    stats.api_call();
                    ui::api_call(stats.turn_calls);
                }
                AgentEvent::Thinking { text } => ui::thinking(&text),
                AgentEvent::ToolInvoked { name } => ui::tool_started(&name),
                AgentEvent::ToolCompleted { name } => ui::tool_done(&name),
                AgentEvent::TokenUsage { prompt_tokens, completion_tokens, total_tokens } => {
                    stats.add_tokens(prompt_tokens, completion_tokens, total_tokens);
                }
                AgentEvent::TurnEnded => {
                    ui::turn_ended();
                    let _ = usage_tx.send(stats.clone());
                }
            }
        }
    });

    let system_prompt = "You are an autonomous AI coding agent helping fix or extend a project \
        that was scaffolded by the RustySpec pipeline. WRITE CODE using tools — when the user \
        reports a bug, IMMEDIATELY read the relevant files, find the problem, and fix it with \
        edit_file/write_file. Never just describe a fix; make it. After editing, run check_code \
        to verify it compiles/parses.";

    let chat_history = state_store::ChatHistory::open(project_root).ok();
    let prior_history = chat_history
        .as_ref()
        .and_then(|h| h.load(Some(60)).ok())
        .unwrap_or_default();
    let prior_count = prior_history.len();

    let mut orchestrator = Orchestrator::new(
        provider,
        dispatcher,
        skills,
        event_bus,
        cancel.clone(),
        agent_types::LanguageMode::En,
    )
    .with_project_root(project_root.to_path_buf())
    .with_system_prompt(system_prompt)
    .with_history(prior_history);
    let mut history_persisted_up_to = prior_count;

    loop {
        if cancel.is_cancelled() {
            break;
        }
        ui::prompt_start();
        let mut line = String::new();
        use tokio::io::AsyncBufReadExt;
        let read_result = reader.read_line(&mut line).await;
        ui::prompt_end();
        match read_result {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "/back" || input == "/quit" || input == "/exit" {
            break;
        }

        let turn_result = orchestrator.run_turn(input).await;

        if let Some(ref history) = chat_history {
            let new_msgs = orchestrator.history_since(history_persisted_up_to);
            if !new_msgs.is_empty() {
                let _ = history.append(new_msgs);
                history_persisted_up_to += new_msgs.len();
            }
        }

        let usage = usage_rx.recv().await.unwrap_or_default();
        match turn_result {
            Ok(response) => ui::answer(&response),
            Err(e) => ui::error(&e.to_string()),
        }
        ui::usage(&usage);
        println!();
    }

    Ok(())
}

/// Stream a single-turn completion from `provider` and collect the full text.
async fn stream_provider_text(
    provider: Arc<dyn LlmProvider>,
    prompt: String,
    cancel: &CancellationToken,
) -> Result<String, Box<dyn std::error::Error>> {
    let messages = vec![agent_types::Message {
        role: agent_types::Role::User,
        content: vec![agent_types::ContentBlock::Text(prompt)],
        token_estimate: 0,
    }];
    let mut rx = provider.stream(&messages, &[], cancel).await
        .map_err(|e| format!("LLM error: {e}"))?;

    let mut text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            llm_client::SseEvent::Delta(d) => text.push_str(&d),
            llm_client::SseEvent::Stop { .. } => break,
            llm_client::SseEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }
    Ok(text)
}

/// Run the Implement stage through the real agent loop: the same tool set,
/// hooks, and dispatcher as `chat`, so `write_file`/`edit_file`/`bash` actually
/// modify the workspace per the task list, instead of producing a text-only
/// artifact.
async fn run_spec_implement(
    project_root: &std::path::Path,
    provider: Arc<dyn LlmProvider>,
    prompt: String,
    pipeline: spec_pipeline::Pipeline,
) -> Result<(), Box<dyn std::error::Error>> {
    let hook_list: Vec<Arc<dyn harness::Hook>> = vec![
        Arc::new(harness::SecretLeakHook::new()),
        Arc::new(harness::DestructiveCommandHook::new()),
    ];
    let hooks = Arc::new(HookEngine::new(hook_list));
    let tools = agent_core::default_tools_with_subagent(provider.clone());
    let dispatcher = Arc::new(ToolDispatcher::new(tools, hooks));
    let skills = Arc::new(SkillRegistry::load(None).unwrap());
    let event_bus = EventBus::default();
    let cancel = CancellationToken::new();

    let mut events = event_bus.subscribe();
    tokio::spawn(async move {
        use agent_types::AgentEvent;
        while let Ok(event) = events.recv().await {
            match event {
                AgentEvent::TurnStarted => ui::turn_started(),
                AgentEvent::ApiCallStarted => ui::api_call(1),
                AgentEvent::ToolInvoked { name } => ui::tool_started(&name),
                AgentEvent::ToolCompleted { name } => ui::tool_done(&name),
                AgentEvent::TurnEnded => ui::turn_ended(),
                _ => {}
            }
        }
    });

    let system_prompt = "You are implementing a RustySpec Implement stage. \
        The task list and prior artifacts are given in the user message. \
        CRITICAL RULES:\n\
        1. You MUST call the write_file tool for EVERY file. NEVER paste code in your text response.\n\
        2. Each write_file call MUST have a non-empty 'content' field containing the COMPLETE file.\n\
        3. Do NOT describe what you will write — just call write_file immediately.\n\
        4. Create ALL files listed in the task plan in a single turn.\n\
        5. After writing all files, call check_code to verify.\n\
        6. If check_code shows errors, use edit_file to fix them.\n\
        NEVER say 'I will create' or 'here is the code' — USE THE TOOL.";

    let mut orchestrator = Orchestrator::new(
        provider,
        dispatcher,
        skills,
        event_bus,
        cancel.clone(),
        agent_types::LanguageMode::En,
    )
    .with_project_root(project_root.to_path_buf())
    .with_system_prompt(system_prompt);

    let mut response = orchestrator.run_turn(prompt).await?;

    // If the model responded with text containing code blocks but made zero
    // tool calls (common with weaker function-calling models), nudge it to
    // retry using actual tools. Try up to 2 nudges before giving up.
    for attempt in 0..2 {
        let has_code_blocks = response.contains("```");
        // Check if any file was written by looking at the orchestrator history
        // for ToolResult blocks with "Wrote" or "Updated" in the output.
        let wrote_files = orchestrator.history().iter().any(|m| {
            m.content.iter().any(|b| {
                if let agent_types::ContentBlock::ToolResult { output, is_error, .. } = b {
                    !is_error && (output.contains("Wrote ") || output.contains("Updated "))
                } else {
                    false
                }
            })
        });

        if wrote_files || !has_code_blocks {
            break; // Model used tools correctly, or no code to write
        }

        eprintln!(
            "  \x1b[33m⟳ Model pasted code in text instead of calling write_file. Retrying (attempt {})...\x1b[0m",
            attempt + 2
        );
        let nudge = "You pasted code in your text response but did NOT call write_file. \
            That does NOT create files. You MUST call the write_file tool with the full \
            file content for EACH file. Do it now — call write_file for every file \
            that needs to be created.".to_string();
        response = orchestrator.run_turn(nudge).await?;
    }

    // Record a short summary artifact (the Implement stage's artifact slot is
    // a directory; write a log file inside it rather than treating the
    // directory path itself as a file).
    let log_dir = pipeline.session_dir().join("code");
    tokio::fs::create_dir_all(&log_dir).await?;
    let log_path = log_dir.join("IMPLEMENTATION_LOG.md");
    tokio::fs::write(&log_path, format!(
        "# Implementation Log\n\n{response}\n"
    )).await?;

    println!("\nImplementation complete. Summary logged at: {}", log_path.display());

    // Verify that files were actually created in the workspace (not just
    // described in text). Count non-.agent files to detect DeepSeek-style
    // models that paste code in prose instead of calling write_file.
    let mut file_count = 0u32;
    let ignore_dirs = [".agent", ".git", "node_modules", "target"];
    if let Ok(mut rd) = tokio::fs::read_dir(project_root).await {
        while let Ok(Some(entry)) = rd.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if ignore_dirs.contains(&name.as_str()) {
                continue;
            }
            if entry.file_type().await.map(|t| t.is_file()).unwrap_or(false) {
                file_count += 1;
            } else if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
                // Count at least one file in subdirs
                if let Ok(mut sub) = tokio::fs::read_dir(entry.path()).await {
                    while let Ok(Some(se)) = sub.next_entry().await {
                        if se.file_type().await.map(|t| t.is_file()).unwrap_or(false) {
                            file_count += 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    if file_count == 0 {
        eprintln!("\n\x1b[1;31m⚠ WARNING:\x1b[0m No project files were created in the workspace!");
        eprintln!("  The model may have described code in text instead of calling write_file.");
        eprintln!("  Try: /rerun implement");
    } else {
        println!("  {} project file(s) verified in workspace.", file_count);
    }

    println!("\n--- Agent summary ---\n{response}");

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

/// A pre-turn snapshot plus the repository state needed to restore it without
/// leaving the user's branch pointed at an agent-generated commit.
#[derive(Clone, Debug)]
struct GitCheckpoint {
    commit: String,
    head: String,
    index_tree: String,
}

fn git_output(
    dir: &std::path::Path,
    args: &[&str],
) -> Result<String, Box<dyn std::error::Error>> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git {} failed: {}", args.join(" "), stderr.trim()).into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Snapshot the complete non-ignored working tree in a commit reachable through
/// `reference`, while restoring the user's original staging area afterwards.
fn git_snapshot(
    dir: &std::path::Path,
    message: &str,
    reference: &str,
) -> Result<GitCheckpoint, Box<dyn std::error::Error>> {
    let head = git_output(dir, &["rev-parse", "--verify", "HEAD"])?;
    let index_tree = git_output(dir, &["write-tree"])?;

    let snapshot_result = (|| -> Result<String, Box<dyn std::error::Error>> {
        git_output(dir, &["add", "-A"])?;
        let tree = git_output(dir, &["write-tree"])?;
        let commit = git_output(
            dir,
            &["commit-tree", &tree, "-p", &head, "-m", message],
        )?;
        git_output(dir, &["update-ref", reference, &commit])?;
        Ok(commit)
    })();

    // `git add -A` is only used to construct the snapshot tree. Never leave it
    // behind as a staging-area side effect, including on failure paths.
    let restore_result = git_output(dir, &["read-tree", &index_tree]);
    match (snapshot_result, restore_result) {
        (Ok(commit), Ok(_)) => Ok(GitCheckpoint {
            commit,
            head,
            index_tree,
        }),
        (Err(snapshot_error), Ok(_)) => Err(snapshot_error),
        (Ok(_), Err(restore_error)) => Err(restore_error),
        (Err(snapshot_error), Err(restore_error)) => Err(format!(
            "{snapshot_error}; additionally failed to restore Git index: {restore_error}"
        )
        .into()),
    }
}

/// Create an exact pre-turn checkpoint without adding a commit to the user's
/// branch history. The object is retained under a dedicated agent ref.
fn git_checkpoint(
    dir: &std::path::Path,
    user_msg: &str,
) -> Result<GitCheckpoint, Box<dyn std::error::Error>> {
    let short_msg: String = user_msg.chars().take(50).collect();
    git_snapshot(
        dir,
        &format!("agent checkpoint: {short_msg}"),
        "refs/agent/checkpoints/last",
    )
}

/// Restore the exact pre-turn snapshot. Before any hard reset, save the current
/// tree under a unique backup ref so concurrent/manual user edits remain
/// recoverable even though the working tree is restored to its pre-turn state.
fn git_undo_last_checkpoint(
    dir: &std::path::Path,
    checkpoint: &GitCheckpoint,
) -> Result<String, Box<dyn std::error::Error>> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let backup_ref = format!("refs/agent/backups/undo-{unique}-{}", std::process::id());
    let backup = git_snapshot(dir, "agent undo safety backup", &backup_ref)?;

    let restore_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        // First make post-turn untracked files part of Git's tracked snapshot;
        // the following reset can then remove files absent from the checkpoint.
        git_output(dir, &["reset", "--hard", &backup.commit])?;
        git_output(dir, &["reset", "--hard", &checkpoint.commit])?;
        // Preserve the user's current branch/commits, then restore the staging
        // area exactly as it was before the agent turn.
        git_output(dir, &["reset", "--soft", &backup.head])?;
        git_output(dir, &["read-tree", &checkpoint.index_tree])?;
        Ok(())
    })();

    if let Err(error) = restore_result {
        // Best-effort rollback to the state captured immediately before undo.
        let _ = git_output(dir, &["reset", "--hard", &backup.commit]);
        let _ = git_output(dir, &["reset", "--soft", &backup.head]);
        let _ = git_output(dir, &["read-tree", &backup.index_tree]);
        return Err(format!(
            "{error}. Pre-undo work is preserved at {backup_ref} ({})",
            backup.commit
        )
        .into());
    }

    Ok(format!(
        "Reverted the last agent turn to checkpoint {}. Pre-undo work is recoverable at {backup_ref}.",
        &checkpoint.commit[..checkpoint.commit.len().min(8)]
    ))
}
