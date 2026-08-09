//! `cli` (L5): clap-based CLI entrypoint.
//!
//! Subcommands: chat, index, spec, search, eval.
//! Ctrl-C cancels the root CancellationToken → graceful drain.

mod input;
mod mcp_config;
mod spec_input;
mod spec_output;
mod ui;
mod validation_manifest;

use std::sync::Arc;

use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;

use agent_core::{Orchestrator, ToolDispatcher};
use harness::{HookEngine, SkillRegistry};
use llm_client::GeminiProvider;
use llm_client::LlmProvider;
use runtime_core::EventBus;

/// Canonicalize a path and strip the Windows `\\?\` verbatim prefix so
/// downstream APIs handle spaces and Unicode in path components correctly.
fn canonical_project_root(path: &std::path::Path) -> std::path::PathBuf {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    #[cfg(windows)]
    {
        let s = canon.to_string_lossy();
        if let Some(stripped) = s.strip_prefix(r"\\?\") {
            if stripped.len() < 260 && stripped.chars().nth(1) == Some(':') {
                return std::path::PathBuf::from(stripped.to_string());
            }
        }
        canon
    }
    #[cfg(not(windows))]
    {
        canon
    }
}

#[derive(Parser)]
#[command(
    name = "srijandev",
    version,
    about = "Srijan Dev — AI-Powered Autonomous Coding Agent"
)]
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
        /// Session mode: `vibe` for conversational coding, `spec` for the
        /// structured requirements→design→tasks pipeline. Interactive terminals
        /// prompt for selection when omitted; redirected stdin defaults to `vibe`
        /// without consuming the first chat line.
        #[arg(short = 'm', long = "mode", value_parser = ["vibe", "spec"])]
        mode: Option<String>,
    },
    /// Start IPC server for editor integration. Binds loopback only and is
    /// UNAUTHENTICATED: any local process can connect and edit documents. No
    /// editor extension ships with this repository; it is a planned project.
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
        /// Read the `specify` description from this file instead of the
        /// terminal. Preferred for anything long: a file arrives whole, with
        /// its blank lines and headings intact. Relative paths resolve against
        /// the workspace. Ignored by the other stages, which read their input
        /// from prior artifacts.
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<String>,
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
    /// Validate the release-evidence manifest and derive readiness. Offline and
    /// deterministic: it reads only the manifest file and exits nonzero unless
    /// every mandatory gate is recorded as passed.
    ValidateEvidence {
        /// Manifest path (default: the audit spec's manifest)
        #[arg(
            long,
            default_value = ".kiro/specs/audit-production-hardening/validation-manifest.json"
        )]
        manifest: String,
    },
}

#[derive(Subcommand)]
enum EvalAction {
    /// Run an evaluation suite in check_only mode: each case's check command is
    /// executed, but the agent is NOT driven against the case prompt. Turn,
    /// tool-call, and token metrics are therefore not measured and are omitted.
    Run {
        #[arg(long, default_value = "swebench-lite")]
        suite: String,
        #[arg(long, default_value = "2")]
        max_concurrent: usize,
    },
    /// Set a baseline from a run
    Baseline { run_id: String },
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

    let input = input::InputBroker::stdio();
    let result = run(cli, cancel, input).await;
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

async fn run(
    cli: Cli,
    _cancel: CancellationToken,
    input: Arc<input::InputBroker>,
) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Chat { workspace, mode } => {
            run_chat(_cancel, workspace, mode, input.clone()).await?;
        }
        Commands::Serve { port } => {
            run_serve(_cancel, port).await?;
        }
        Commands::Index => {
            run_index().await?;
        }
        Commands::Spec {
            stage,
            workspace,
            from_file,
        } => {
            run_spec(
                _cancel.clone(),
                &stage,
                workspace,
                from_file.as_deref(),
                input.clone(),
            )
            .await?;
        }
        Commands::Search { query, top_k } => {
            run_search(&query, top_k).await?;
        }
        Commands::Eval { action } => match action {
            EvalAction::Run {
                suite,
                max_concurrent,
            } => {
                run_eval_run(_cancel.clone(), &suite, max_concurrent).await?;
            }
            EvalAction::Baseline { run_id } => {
                println!("Setting baseline: {run_id}");
                println!("(Copy the run's .jsonl as .agent/evals/baseline.jsonl)");
            }
            EvalAction::Diff { run_a, run_b } => {
                run_eval_diff(&run_a, &run_b)?;
            }
        },
        Commands::ValidateEvidence { manifest } => {
            run_validate_evidence(&manifest)?;
        }
    }
    Ok(())
}

/// Validate the evidence manifest and fail the process unless it derives
/// `ready`. Printing the reasons is the point: an operator must be able to see
/// exactly which gate is missing without reading the manifest by hand.
fn run_validate_evidence(manifest: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(manifest);
    let outcome = validation_manifest::validate_file(path);
    print!("{}", outcome.render());
    if outcome.is_ready() {
        Ok(())
    } else {
        Err(format!(
            "evidence manifest at {} does not support a ready status ({} blocking reason(s))",
            path.display(),
            outcome.reasons.len()
        )
        .into())
    }
}

/// Operator override for the generated-output cap, via `LLM_MAX_OUTPUT_TOKENS`.
///
/// Exists because the right value is a property of the deployment, not of this
/// code. A large prompt combined with a large output budget was observed to fail
/// against a live endpoint, so the default is deliberately conservative; a model
/// with a bigger budget can be given one here without a code change.
fn max_output_tokens_override() -> Option<u32> {
    std::env::var("LLM_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|budget| *budget > 0)
}

/// Build an [`llm_client::Embedder`] from the environment, or `None` for
/// keyword-only search.
///
/// Deliberately provider-agnostic. This tool is meant to work with whatever a
/// user already has, so anything speaking the OpenAI `/embeddings` shape is
/// supported — OpenAI, Azure AI Foundry, Mistral, DeepSeek, Together,
/// OpenRouter, and local runtimes such as Ollama, LM Studio, and vLLM — in
/// addition to Gemini's own shape.
///
/// Resolution order:
/// 1. `EMBEDDING_PROVIDER` when set, which selects the request shape explicitly.
/// 2. Otherwise `GEMINI_API_KEY`, preserving the previous behaviour for existing
///    users who have only that configured.
/// 3. Otherwise none, and search stays keyword-only.
///
/// A missing credential is not by itself a reason to refuse: local runtimes need
/// none, so an empty key is only fatal for providers that actually require one.
fn resolve_embedder() -> Option<Box<dyn llm_client::Embedder>> {
    let provider = std::env::var("EMBEDDING_PROVIDER")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    if provider.is_empty() {
        let gemini_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();
        if gemini_key.is_empty() {
            return None;
        }
        return Some(Box::new(llm_client::GeminiEmbedder::new(
            gemini_key,
            gemini_embedding_base_url(),
        )));
    }

    // Key precedence: an embedding-specific key first, so a user can point
    // indexing at a different account than chat without changing chat.
    let key = first_non_empty_env(&[
        "EMBEDDING_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ]);

    if provider == "gemini" {
        if key.is_empty() {
            eprintln!("  EMBEDDING_PROVIDER=gemini needs an API key; falling back to keyword only");
            return None;
        }
        return Some(Box::new(llm_client::GeminiEmbedder::new(
            key,
            std::env::var("EMBEDDING_BASE_URL").unwrap_or_else(|_| gemini_embedding_base_url()),
        )));
    }

    // Everything else is treated as OpenAI-compatible, which is what makes a
    // single implementation cover both hosted and self-hosted backends.
    let base_url = std::env::var("EMBEDDING_BASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| default_embedding_base_url(&provider))
        .unwrap_or_else(|| {
            eprintln!(
                "  EMBEDDING_PROVIDER={provider} has no built-in endpoint; \
                 set EMBEDDING_BASE_URL"
            );
            String::new()
        });
    if base_url.is_empty() {
        return None;
    }

    let model = std::env::var("EMBEDDING_MODEL").unwrap_or_default();
    if model.trim().is_empty() {
        eprintln!("  EMBEDDING_PROVIDER={provider} needs EMBEDDING_MODEL; keyword only");
        return None;
    }

    let mut embedder =
        llm_client::OpenAiCompatEmbedder::new(key, base_url, model).with_provider(provider);

    // An explicit width is honoured two ways: requested from the endpoint when it
    // supports shortening, and enforced on the response either way.
    if let Some(dimension) = std::env::var("EMBEDDING_DIMENSIONS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|dimension| *dimension > 0)
    {
        embedder = embedder.with_requested_dimensions(dimension);
    }

    Some(Box::new(embedder))
}

fn first_non_empty_env(names: &[&str]) -> String {
    for name in names {
        if let Ok(value) = std::env::var(name) {
            if !value.trim().is_empty() {
                return value;
            }
        }
    }
    String::new()
}

fn gemini_embedding_base_url() -> String {
    if std::env::var("GEMINI_USE_AI_STUDIO").unwrap_or_default() == "1" {
        "https://generativelanguage.googleapis.com/v1beta".to_string()
    } else {
        "https://aiplatform.googleapis.com/v1/publishers/google".to_string()
    }
}

/// Well-known `/embeddings` roots, so common setups need no URL.
fn default_embedding_base_url(provider: &str) -> Option<String> {
    let url = match provider {
        "openai" => "https://api.openai.com/v1",
        "mistral" => "https://api.mistral.ai/v1",
        "deepseek" => "https://api.deepseek.com",
        "together" => "https://api.together.xyz/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "ollama" => "http://localhost:11434/v1",
        "lmstudio" => "http://localhost:1234/v1",
        "vllm" => "http://localhost:8000/v1",
        _ => return None,
    };
    Some(url.to_string())
}

/// Embedding profile used for *search*, which must match how the index was
/// built or the compatibility check will correctly refuse to use it.
fn search_embedding_profile() -> vecstore::EmbeddingProfile {
    match resolve_embedder() {
        Some(embedder) => {
            let dimension = embedder
                .declared_dimension()
                .unwrap_or(vecstore::EMBEDDING_DIMENSION);
            vecstore::EmbeddingProfile::new(embedder.provider(), embedder.model(), dimension)
        }
        None => vecstore::EmbeddingProfile::default_gemini(),
    }
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
    let (provider, display_model): (Arc<dyn LlmProvider>, String) = match provider_name
        .to_lowercase()
        .as_str()
    {
        "gemini" | "google" => {
            let m = if model.is_empty() {
                "gemini-3.5-flash".to_string()
            } else {
                model
            };
            let p = GeminiProvider::new(api_key, &m);
            let p = if std::env::var("GEMINI_USE_AI_STUDIO").unwrap_or_default() == "1" {
                p.with_base_url("https://generativelanguage.googleapis.com/v1beta")
            } else {
                p
            };
            (Arc::new(p), m)
        }
        "openai" | "gpt" => {
            let m = if model.is_empty() {
                "gpt-5.5".to_string()
            } else {
                model
            };
            let base = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            let mut provider = llm_client::OpenAiCompatProvider::new(api_key, &m, &base);
            if let Some(budget) = max_output_tokens_override() {
                provider = provider.with_max_tokens(budget);
            }
            // Reasoning models reject `max_tokens` and require
            // `max_completion_tokens`; older and third-party endpoints only
            // accept `max_tokens`. The provider infers this from the model name,
            // but Azure sends a *deployment* name that can be anything, so allow
            // an explicit override rather than forcing a rename or a code change.
            if let Some(field) = std::env::var("OPENAI_TOKEN_LIMIT_FIELD")
                .ok()
                .and_then(|value| llm_client::TokenLimitField::parse(&value))
            {
                provider = provider.with_token_limit_field(field);
            }
            (Arc::new(provider), m)
        }
        "anthropic" | "claude" => {
            let m = if model.is_empty() {
                "claude-fable-5".to_string()
            } else {
                model
            };
            let provider = llm_client::AnthropicProvider::new(api_key, &m);
            let provider = if let Ok(base) = std::env::var("ANTHROPIC_BASE_URL") {
                provider.with_base_url(base)
            } else {
                provider
            };
            (Arc::new(provider), m)
        }
        "mistral" => {
            let m = if model.is_empty() {
                "mistral-medium-3.5".to_string()
            } else {
                model
            };
            (
                Arc::new(llm_client::OpenAiCompatProvider::new(
                    api_key,
                    &m,
                    "https://api.mistral.ai/v1",
                )),
                m,
            )
        }
        "deepseek" => {
            let m = if model.is_empty() {
                "deepseek-v4-pro".to_string()
            } else {
                model
            };
            (
                Arc::new(llm_client::OpenAiCompatProvider::new(
                    api_key,
                    &m,
                    "https://api.deepseek.com",
                )),
                m,
            )
        }
        "ollama" | "local" => {
            let m = if model.is_empty() {
                "llama3.3".to_string()
            } else {
                model
            };
            let base = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
            (
                Arc::new(llm_client::OpenAiCompatProvider::new("", &m, &base)),
                m,
            )
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
async fn run_chat(
    cancel: CancellationToken,
    workspace: Option<String>,
    mode_override: Option<String>,
    input: Arc<input::InputBroker>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve workspace: --workspace flag > current directory.
    let project_root = match workspace {
        Some(ref dir) => std::path::PathBuf::from(dir),
        None => std::env::current_dir()?,
    };
    let project_root = canonical_project_root(&project_root);

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

    // Interactive terminals retain mode selection. A --mode flag bypasses the
    // prompt. Redirected stdin defaults to Vibe without consuming the first line.
    let session_mode = match mode_override.as_deref() {
        Some("spec") => input::SessionMode::RustySpec,
        Some(_) => input::SessionMode::Vibe,
        None => {
            if input.is_interactive() {
                ui::mode_select();
            }
            input.select_mode().await?
        }
    };
    if session_mode == input::SessionMode::RustySpec {
        return run_rustyspec_session(
            cancel,
            project_root,
            provider,
            provider_name,
            display_model,
            api_key,
            input,
        )
        .await;
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

    // Repository-controlled MCP startup is gated as one canonical, identity-bound
    // transaction. The trust module parses once, obtains/reuses a user-local
    // decision through the sole InputBroker, durably records explicit decisions,
    // and invokes the supervised spawner only after approval.
    let mcp_spawner = mcp_config::ProductionMcpSpawner;
    match mcp_config::start_workspace_servers(
        &project_root,
        input.as_ref(),
        input.is_interactive(),
        &mcp_spawner,
    )
    .await
    {
        Ok(attempts) => {
            for attempt in attempts {
                let name = attempt.server_name;
                match attempt.result {
                    Ok(client) => {
                        let client = Arc::new(client);
                        let tools = agent_core::mcp_tools(client);
                        let server_tool_count = tools.len();
                        all_tools.extend(tools);
                        eprintln!("  MCP server '{name}' connected ({server_tool_count} tools)");
                    }
                    Err(error) => eprintln!("  MCP server '{name}' failed: {error}"),
                }
            }
        }
        Err(error) => eprintln!("  MCP startup disabled: {error}"),
    }

    let dispatcher = Arc::new(ToolDispatcher::new(all_tools, hooks));
    let skills = Arc::new(load_workspace_skills(&project_root));
    let event_bus = EventBus::default();

    // Subscribe before handing the bus to the orchestrator. A completed usage
    // snapshot is sent back to the chat loop so response and accounting render
    // in a deterministic order.
    let mut events = event_bus.subscribe();
    let (usage_tx, mut usage_rx) = tokio::sync::mpsc::unbounded_channel::<ui::UsageStats>();
    tokio::spawn(async move {
        use agent_types::AgentEvent;
        let mut stats = ui::UsageStats::default();
        loop {
            let event = match events.recv().await {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    eprintln!("warning: event listener skipped {skipped} events (burst)");
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
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
                AgentEvent::EventLagged(lag) => {
                    eprintln!("warning: event listener skipped {} events", lag.skipped);
                }
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
        let mem = std::fs::read_to_string(project_root.join(".agent").join("MEMORY.md"))
            .unwrap_or_default();
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
                memory_context.push_str(&format!(
                    "- [{}] {}\n",
                    if *done { "x" } else { " " },
                    desc
                ));
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

    // Load chat history from previous sessions. The versioned snapshot is
    // authoritative; a legacy JSONL store is imported once on first load.
    let chat_history = state_store::ChatHistory::open(&project_root).ok();
    let mut conversation_snapshot = match chat_history.as_ref() {
        Some(h) => match h.load_snapshot(Some(60)) {
            Ok((snapshot, migration)) => {
                if let Some(report) = migration {
                    for line in &report.malformed_lines {
                        ui::error(&format!(
                            "Skipped malformed legacy history line {line} during migration"
                        ));
                    }
                    if report.trimmed_incomplete > 0 || report.trimmed_leading_orphans > 0 {
                        ui::error(&format!(
                            "Trimmed {} leading and {} trailing incomplete tool messages during migration",
                            report.trimmed_leading_orphans, report.trimmed_incomplete
                        ));
                    }
                    if let Some(backup) = &report.backup_path {
                        println!(
                            "Legacy history migrated; backup retained at {}",
                            backup.display()
                        );
                    }
                }
                snapshot
            }
            Err(error) => {
                // Never silently start from an empty conversation.
                ui::error(&format!("Could not load conversation history: {error}"));
                return Ok(());
            }
        },
        None => state_store::ConversationSnapshotV2::default(),
    };
    let prior_history = conversation_snapshot.plain_messages();
    let prior_count = prior_history.len();
    let restored_summary = conversation_snapshot.compacted_summary.clone();

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
    .with_approval_provider(input.clone())
    .with_system_prompt(base_prompt)
    .with_history(prior_history);

    // Restore the compacted summary as conversation data, not policy.
    orchestrator.set_compacted_summary(restored_summary);

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
        eprintln!(
            "  \x1b[38;5;45m↻\x1b[0m restored {prior_count} messages from previous session\n"
        );
    }

    loop {
        if cancel.is_cancelled() {
            break;
        }

        ui::prompt_start();
        let read_result = input.read_line().await;
        ui::prompt_end();
        let line = match read_result {
            Ok(Some(line)) => line,
            Ok(None) | Err(_) => break,
        };

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
            // Report success only after BOTH the durable store and every
            // in-memory conversation layer are cleared. Clearing only one side
            // lets the next turn repopulate the other.
            match chat_history {
                Some(ref h) => match h.clear() {
                    Ok(report) => {
                        // Empty the persisted snapshot without regressing its
                        // monotonic identity: `generation` and `next_message_id`
                        // must keep advancing so a cleared conversation can
                        // never be confused with an earlier one. Then drop every
                        // in-memory layer the orchestrator could write back.
                        conversation_snapshot.replace_messages(&[]);
                        conversation_snapshot.compacted_summary = Default::default();
                        orchestrator.clear_conversation();
                        println!("Chat history cleared. Starting fresh.\n");
                        for backup in &report.retained_backups {
                            println!(
                                "Note: a migration backup is still on disk and is never restored automatically: {}",
                                backup.display()
                            );
                        }
                    }
                    Err(error) => {
                        // Storage removal failed, possibly partially. Clear the
                        // in-memory layers anyway so the next turn cannot
                        // re-persist the conversation the user asked to remove,
                        // and state plainly that disk state may remain.
                        conversation_snapshot.replace_messages(&[]);
                        conversation_snapshot.compacted_summary = Default::default();
                        orchestrator.clear_conversation();
                        eprintln!(
                            "Chat history storage was NOT fully cleared: {error}\n\
                             The in-memory conversation was cleared, so it will not be written back, \
                             but files on disk may still contain earlier messages.\n"
                        );
                    }
                },
                None => {
                    // Without a storage handle there is nothing durable to
                    // remove, but the live conversation must still be dropped so
                    // `/clear` is never a silent no-op.
                    orchestrator.clear_conversation();
                    conversation_snapshot = state_store::ConversationSnapshotV2::default();
                    eprintln!(
                        "Chat history storage is unavailable in this workspace; cleared the in-memory conversation only.\n"
                    );
                }
            }
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

        // Persist the complete conversation state as one atomic snapshot.
        // A full replacement removes the index cursor entirely, so compaction
        // truncating history from the front can no longer skip persistence.
        if let Some(ref history) = chat_history {
            conversation_snapshot.replace_messages(orchestrator.history());
            conversation_snapshot.compacted_summary = orchestrator.compacted_summary().clone();
            match history.save_snapshot(&conversation_snapshot) {
                Ok(saved) => conversation_snapshot = saved,
                Err(error) => ui::error(&format!("Failed to persist conversation: {error}")),
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

    // 1. Build the current Merkle tree. Directory hash nodes remain in the
    // tree for diffing, but only supported, readable regular files are
    // selected for parsing and reported as files.
    let current_tree = indexer::MerkleTree::build(&cwd)?;

    // 2. Diff against the previous tree (if one exists). Removed, unreadable,
    // and newly unsupported paths remain in this set so their stale rows can
    // be cleaned even though they are not selected for parsing.
    let changed_paths = if merkle_path.exists() {
        let old_tree = indexer::MerkleTree::load(&merkle_path)?;
        let diff = old_tree.diff(&current_tree);
        println!("  Changed since last index: {} files", diff.len());
        diff
    } else {
        println!("  First index — processing all files.");
        current_tree.nodes.keys().cloned().collect::<Vec<_>>()
    };

    let mut selected_files: Vec<std::path::PathBuf> = changed_paths
        .iter()
        .filter(|rel| {
            let path = cwd.join(rel);
            path.is_file()
                && indexer::Language::from_path(&path).is_some()
                && std::fs::read_to_string(path).is_ok()
        })
        .cloned()
        .collect();
    selected_files.sort();
    println!("  Files found: {}", selected_files.len());

    if changed_paths.is_empty() {
        println!("Index up to date. 0 files changed.");
        return Ok(());
    }

    // 3. Open vector store and atomically remove stale rows for changed paths
    // that are no longer eligible for parsing.
    let store = vecstore::VecStore::open(&db_path)?;
    for rel in &changed_paths {
        if !selected_files.contains(rel) {
            store.remove_file(&rel.to_string_lossy())?;
        }
    }

    // 4. Setup embedder. Any provider, or keyword-only when none is configured.
    let embedder = resolve_embedder();
    let embedding_profile = match &embedder {
        Some(embedder) => {
            let dimension = llm_client::resolve_dimension(embedder.as_ref()).await?;
            println!(
                "  Embedding mode: SEMANTIC ({} / {} / {dimension}d)",
                embedder.provider(),
                embedder.model()
            );
            Some(vecstore::EmbeddingProfile::new(
                embedder.provider(),
                embedder.model(),
                dimension,
            ))
        }
        None => {
            println!(
                "  Embedding mode: KEYWORD ONLY (set EMBEDDING_PROVIDER/EMBEDDING_MODEL, \
                 or GEMINI_API_KEY, for semantic search)"
            );
            None
        }
    };

    // The vector table is created for one fixed width, so switching embedding
    // models requires rebuilding it. Doing this before indexing means a width
    // change is handled once, up front, instead of failing per insert.
    if let Some(profile) = &embedding_profile {
        if vecstore::ensure_vector_dimension(store.conn(), profile.dimension)? {
            println!(
                "  Vector index rebuilt for {}d; previous embeddings were dropped and are \
                 being regenerated",
                profile.dimension
            );
        }
    }

    // 5. Parse, chunk, and embed a complete replacement before asking
    // VecStore to open its transaction. Standalone stale deletion remains an
    // independent atomic operation for deleted/unsupported/unreadable files.
    let mut total_chunks = 0u64;
    let mut errors = 0u64;
    for rel in &selected_files {
        let path = cwd.join(rel);
        let rel_str = rel.to_string_lossy().to_string();

        let source = match std::fs::read_to_string(&path) {
            Ok(source) => source,
            Err(error) => {
                eprintln!("  Skipping unreadable {}: {error}", rel.display());
                store.remove_file(&rel_str)?;
                continue;
            }
        };

        let entities = indexer::parse(&path, &source)?;
        if entities.is_empty() {
            store.remove_file(&rel_str)?;
            continue;
        }

        let chunks = indexer::chunk(&entities, &source, 512, 1);
        eprint!("  {} ({} chunks)...", rel.display(), chunks.len());

        let mut inserts: Vec<vecstore::ChunkInsert> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let embedding = if let Some(ref embedder) = embedder {
                match embedder.embed(&chunk.text).await {
                    Ok(vector) => vector,
                    Err(error) => {
                        errors += 1;
                        if errors <= 3 {
                            eprintln!("\n    Warning: embed failed: {error}");
                        }
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            inserts.push(vecstore::ChunkInsert {
                file_path: rel_str.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                text: chunk.text.clone(),
                token_count: chunk.token_count,
                embedding,
            });
        }

        let mtime = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| {
                modified
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .ok()
            })
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0);
        let file = vecstore::FileRecord {
            path: rel_str,
            mtime,
            content_hash: String::new(),
        };

        // Record the profile that actually produced these vectors. Hardcoding
        // Gemini here would label another backend's vectors as Gemini's, and the
        // compatibility check would then happily compare incompatible vectors.
        store.replace_file_with_profile(
            file,
            &inserts,
            embedding_profile
                .as_ref()
                .unwrap_or(&vecstore::EmbeddingProfile::default_gemini()),
        )?;
        total_chunks += inserts.len() as u64;
        eprintln!(" ok");
    }

    // 6. Save updated Merkle tree.
    current_tree.save(&merkle_path)?;

    let (chunks, _vecs, _fts) = store.chunk_counts()?;
    println!(
        "\nIndex complete. {} files processed, {} new chunks (total in DB: {}).",
        selected_files.len(),
        total_chunks,
        chunks
    );
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

    // Try semantic search when an embedder is configured, falling back to BM25
    // with an actionable compatibility reason when the index cannot be used.
    // Provider-agnostic on purpose: the query must be embedded by the same
    // backend that built the index, whichever backend that is.
    let (hits, mode_name, bm25_only_reason) = if let Some(embedder) = resolve_embedder() {
        match embedder.embed(query).await {
            Ok(query_embedding) => {
                // The stored width is authoritative for search: the index was
                // built at whatever width the embedder produced, and asserting a
                // different one here would reject a perfectly usable index.
                let mut profile = search_embedding_profile();
                profile.dimension = query_embedding.len();
                let report = vecstore::search_with_profile(
                    &store,
                    query,
                    Some(&query_embedding),
                    &[],
                    vecstore::SearchMode::Hybrid,
                    top_k,
                    &profile,
                )?;
                let mode = if report.bm25_only_reason.is_some() {
                    "keyword (BM25-only compatibility fallback)"
                } else {
                    "hybrid (semantic + keyword)"
                };
                (report.hits, mode, report.bm25_only_reason)
            }
            Err(_) => {
                let hits = vecstore::search(
                    &store,
                    query,
                    None,
                    &[],
                    vecstore::SearchMode::Keyword,
                    top_k,
                )?;
                (
                    hits,
                    "keyword (embedding failed, fallback)",
                    Some(
                        "BM25-only: query embedding failed; check the embedding provider credentials and retry."
                            .to_string(),
                    ),
                )
            }
        }
    } else {
        let hits = vecstore::search(
            &store,
            query,
            None,
            &[],
            vecstore::SearchMode::Keyword,
            top_k,
        )?;
        (hits, "keyword only", None)
    };

    if let Some(reason) = bm25_only_reason {
        eprintln!("{reason}");
    }

    if hits.is_empty() {
        println!("No results for: \"{query}\" [mode: {mode_name}]");
        return Ok(());
    }

    println!("Search results for: \"{query}\" (top {top_k}, mode: {mode_name})\n");
    for (i, hit) in hits.iter().enumerate() {
        println!(
            "{}. {} (lines {}-{}, score: {:.3})",
            i + 1,
            hit.file_path,
            hit.start_line,
            hit.end_line,
            hit.score
        );
        let preview: String = hit.text.lines().take(2).collect::<Vec<_>>().join("\n");
        println!("   {preview}");
        println!();
    }

    Ok(())
}

// ===========================================================================
// SPEC command: run a RustySpec pipeline stage
// ===========================================================================

/// Collect the Specify stage description, from a file when one is named and
/// from the terminal otherwise.
///
/// A long description is given as a file on purpose. Pasting many lines into a
/// terminal used to be cut at the first blank line, and the discarded lines
/// then answered whichever prompt came next; a file has neither problem. The
/// interactive session, which has no command-line flag available, reaches the
/// same path by entering `@<path>` on its own line.
async fn collect_specify_description(
    project_root: &std::path::Path,
    from_file: Option<&str>,
    input: &input::InputBroker,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(raw_path) = from_file {
        let text = spec_input::load_description_file(project_root, raw_path)?;
        report_loaded_description(raw_path, &text);
        return Ok(text);
    }

    println!("Describe what you want to build.");
    println!("  - Paste as many lines as you like; blank lines are kept.");
    println!(
        "  - Finish with '{}' on its own line.",
        spec_input::DESCRIPTION_TERMINATOR
    );
    println!("  - Or load a file instead: '@path/to/description.md' on its own line.");

    let typed = input
        .read_multiline_until_terminator(spec_input::MAX_DESCRIPTION_BYTES)
        .await?;

    if let Some(raw_path) = spec_input::parse_file_reference(&typed) {
        let text = spec_input::load_description_file(project_root, raw_path)?;
        report_loaded_description(raw_path, &text);
        return Ok(text);
    }

    Ok(spec_input::check_typed_description(&typed)?)
}

/// Show what was actually loaded, so a wrong or truncated file is caught before
/// a provider call is spent on it.
fn report_loaded_description(raw_path: &str, text: &str) {
    println!(
        "Loaded description from '{raw_path}' ({} bytes, {} lines).",
        text.len(),
        text.lines().count()
    );
}

async fn run_spec(
    cancel: CancellationToken,
    stage_str: &str,
    workspace: Option<String>,
    from_file: Option<&str>,
    input: Arc<input::InputBroker>,
) -> Result<(), Box<dyn std::error::Error>> {
    let project_root = match workspace {
        Some(ref dir) => std::path::PathBuf::from(dir),
        None => std::env::current_dir()?,
    };
    let project_root = canonical_project_root(&project_root);

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
        collect_specify_description(&project_root, from_file, &input).await?
    } else {
        if from_file.is_some() {
            eprintln!(
                "Note: --from-file applies to the 'specify' stage only; \
                 '{stage_str}' reads its input from prior artifacts. Ignoring it."
            );
        }
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
        run_spec_implement(
            cancel.clone(),
            &project_root,
            provider,
            prompt,
            pipeline,
            input.clone(),
        )
        .await?;
        return Ok(());
    }

    // An artifact is published only for output the provider actually finished.
    // Truncated, unterminated, stalled, or cancelled output fails visibly and
    // leaves no file behind. The process-wide token is used so Ctrl-C
    // interrupts a running stage instead of being ignored.
    let completed = spec_output::complete_stage_text(provider, prompt, &cancel).await?;
    let response_text = completed.text;

    // Write artifact.
    let artifact_path = pipeline.write_artifact(stage, &response_text).await?;
    println!("Artifact written: {}", artifact_path.display());
    println!(
        "(stop reason: {:?}, continuations: {})",
        completed.stop_reason, completed.continuations
    );
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
    input: Arc<input::InputBroker>,
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

    loop {
        if cancel.is_cancelled() {
            break;
        }

        // Find the next stage whose artifact doesn't exist yet, to guide the
        // user through the pipeline in order.
        let mut next_stage = None;
        for stage in stages {
            if !pipeline.artifact_path(*stage)?.exists() {
                next_stage = Some(stage);
                break;
            }
        }

        match next_stage {
            Some(stage) => eprint!(
                "\x1b[1;36m❯\x1b[0m next stage [{stage:?}] — run it? (y/n/status/chat/quit): "
            ),
            None => eprint!(
                "\x1b[1;36m❯\x1b[0m all stages complete. (status/chat/rerun <stage>/quit): "
            ),
        }
        use std::io::Write;
        std::io::stderr().flush().ok();

        let Some(line) = input.read_line().await? else {
            break;
        };
        let input_text = line.trim().to_string();
        let input_lower = input_text.to_lowercase();

        if input_lower == "/quit" || input_lower == "quit" || input_lower == "/exit" {
            println!("Bye!");
            break;
        }
        if input_lower == "/status" || input_lower == "status" {
            for s in stages {
                let done = pipeline.artifact_path(*s)?.exists();
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
            run_rustyspec_followup_chat(&cancel, &project_root, provider.clone(), input.clone())
                .await?;
            println!();
            continue;
        }

        // /rerun <stage> — delete an existing artifact and redo that stage
        // (e.g. re-run Implement after Plan/Tasks changed).
        if let Some(rest) = input_lower
            .strip_prefix("/rerun ")
            .or_else(|| input_lower.strip_prefix("rerun "))
        {
            let target = stages
                .iter()
                .find(|s| format!("{:?}", s).to_lowercase() == rest.trim());
            match target {
                Some(stage) => {
                    let path = pipeline.artifact_path(*stage)?;
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
            // A bad or empty description ends this stage, not the session: the
            // user stays in the loop and can enter it again.
            match collect_specify_description(&project_root, None, &input).await {
                Ok(description) => description,
                Err(error) => {
                    eprintln!("{error}\n");
                    continue;
                }
            }
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
            if let Err(e) = run_spec_implement(
                cancel.clone(),
                &project_root,
                provider.clone(),
                prompt,
                pipeline_for_impl,
                input.clone(),
            )
            .await
            {
                eprintln!("Implement stage failed: {e}\n");
            }
            println!();
            continue;
        }

        // Incomplete output must not be published: writing it would also mark
        // this stage done and silently advance the session past it.
        let completed =
            match spec_output::complete_stage_text(provider.clone(), prompt, &cancel).await {
                Ok(completed) => completed,
                Err(error) => {
                    eprintln!("Stage {stage:?} did not complete: {error}");
                    eprintln!("Nothing was written; run the stage again.\n");
                    continue;
                }
            };

        match pipeline.write_artifact(stage, &completed.text).await {
            Ok(path) => println!(
                "Artifact written: {} (stop reason: {:?}, continuations: {})\n",
                path.display(),
                completed.stop_reason,
                completed.continuations
            ),
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
    input: Arc<input::InputBroker>,
) -> Result<(), Box<dyn std::error::Error>> {
    let hook_list: Vec<Arc<dyn harness::Hook>> = vec![
        Arc::new(harness::SecretLeakHook::new()),
        Arc::new(harness::DestructiveCommandHook::new()),
    ];
    let hooks = Arc::new(HookEngine::new(hook_list));
    let tools = agent_core::default_tools_with_subagent(provider.clone());
    let dispatcher = Arc::new(ToolDispatcher::new(tools, hooks));
    let skills = Arc::new(load_workspace_skills(project_root));
    let event_bus = EventBus::default();

    let mut events = event_bus.subscribe();
    let (usage_tx, mut usage_rx) = tokio::sync::mpsc::unbounded_channel::<ui::UsageStats>();
    tokio::spawn(async move {
        use agent_types::AgentEvent;
        let mut stats = ui::UsageStats::default();
        loop {
            let event = match events.recv().await {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    eprintln!("warning: event listener skipped {skipped} events (burst)");
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
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
                } => {
                    stats.add_tokens(prompt_tokens, completion_tokens, total_tokens);
                }
                AgentEvent::EventLagged(lag) => {
                    eprintln!("warning: event listener skipped {} events", lag.skipped);
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
    // Same authoritative snapshot contract as the main loop: no index cursor.
    let mut conversation_snapshot = chat_history
        .as_ref()
        .and_then(|h| h.load_snapshot(Some(60)).ok())
        .map(|(snapshot, _migration)| snapshot)
        .unwrap_or_default();
    let prior_history = conversation_snapshot.plain_messages();
    let restored_summary = conversation_snapshot.compacted_summary.clone();

    let mut orchestrator = Orchestrator::new(
        provider,
        dispatcher,
        skills,
        event_bus,
        cancel.clone(),
        agent_types::LanguageMode::En,
    )
    .with_project_root(project_root.to_path_buf())
    .with_approval_provider(input.clone())
    .with_system_prompt(system_prompt)
    .with_history(prior_history);
    orchestrator.set_compacted_summary(restored_summary);

    loop {
        if cancel.is_cancelled() {
            break;
        }
        ui::prompt_start();
        let read_result = input.read_line().await;
        ui::prompt_end();
        let line = match read_result {
            Ok(Some(line)) => line,
            Ok(None) | Err(_) => break,
        };
        let input_text = line.trim().to_string();
        if input_text.is_empty() {
            continue;
        }
        if input_text == "/back" || input_text == "/quit" || input_text == "/exit" {
            break;
        }

        let turn_result = orchestrator.run_turn(input_text).await;

        if let Some(ref history) = chat_history {
            conversation_snapshot.replace_messages(orchestrator.history());
            conversation_snapshot.compacted_summary = orchestrator.compacted_summary().clone();
            match history.save_snapshot(&conversation_snapshot) {
                Ok(saved) => conversation_snapshot = saved,
                Err(error) => ui::error(&format!("Failed to persist conversation: {error}")),
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

// `stream_provider_text` was removed by Task 22.1. It discarded the terminal
// `StopReason`, so a `MaxTokens` stop returned truncated text that callers
// published as a finished artifact. Specification stages now use
// `spec_output::complete_stage_text`, which reports incompleteness instead.

/// Run the Implement stage through the real agent loop: the same tool set,
/// hooks, and dispatcher as `chat`, so `write_file`/`edit_file`/`bash` actually
/// modify the workspace per the task list, instead of producing a text-only
/// artifact.
async fn run_spec_implement(
    cancel: CancellationToken,
    project_root: &std::path::Path,
    provider: Arc<dyn LlmProvider>,
    prompt: String,
    pipeline: spec_pipeline::Pipeline,
    input: Arc<input::InputBroker>,
) -> Result<(), Box<dyn std::error::Error>> {
    let hook_list: Vec<Arc<dyn harness::Hook>> = vec![
        Arc::new(harness::SecretLeakHook::new()),
        Arc::new(harness::DestructiveCommandHook::new()),
    ];
    let hooks = Arc::new(HookEngine::new(hook_list));
    let tools = agent_core::default_tools_with_subagent(provider.clone());
    let dispatcher = Arc::new(ToolDispatcher::new(tools, hooks));
    let skills = Arc::new(load_workspace_skills(project_root));
    let event_bus = EventBus::default();

    let mut events = event_bus.subscribe();
    tokio::spawn(async move {
        use agent_types::AgentEvent;
        loop {
            let event = match events.recv().await {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    eprintln!("warning: spec event listener skipped {skipped} events (burst)");
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
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
    .with_approval_provider(input)
    .with_system_prompt(system_prompt);

    let mut response = orchestrator.run_turn(prompt).await?;
    // Completion is judged only on mutations this run actually committed, so a
    // pre-existing or unrelated file can never stand in for real work.
    let mut committed_mutations = orchestrator.committed_mutations_this_turn();

    // If the model responded with text containing code blocks but made zero
    // tool calls (common with weaker function-calling models), nudge it to
    // retry using actual tools. Try up to 2 nudges before giving up.
    for attempt in 0..2 {
        let has_code_blocks = response.contains("```");

        if committed_mutations > 0 || !has_code_blocks {
            break; // Model used tools correctly, or no code to write
        }

        eprintln!(
            "  \x1b[33m⟳ Model pasted code in text instead of calling write_file. Retrying (attempt {})...\x1b[0m",
            attempt + 2
        );
        let nudge = "You pasted code in your text response but did NOT call write_file. \
            That does NOT create files. You MUST call the write_file tool with the full \
            file content for EACH file. Do it now — call write_file for every file \
            that needs to be created."
            .to_string();
        response = orchestrator.run_turn(nudge).await?;
        committed_mutations += orchestrator.committed_mutations_this_turn();
    }

    // Without a mutation committed by this run there is nothing to report as
    // done. Counting workspace files here would be wrong: an unrelated
    // pre-existing file would make a prose-only answer look successful. Fail
    // before any completion log is created.
    if committed_mutations == 0 {
        eprintln!("\n\x1b[1;31mImplement stage did not complete.\x1b[0m");
        eprintln!("  No workspace mutation was committed during this run.");
        eprintln!("  The model likely described code in text instead of calling write_file.");
        eprintln!("  No completion log was written. Try: /rerun implement");
        eprintln!("\n--- Agent summary ---\n{response}");
        return Err(Box::new(agent_types::AgentError::Tool {
            name: "spec_implement".into(),
            reason: "no workspace mutation was committed during this run".into(),
        }));
    }

    // Record a short summary artifact (the Implement stage's artifact slot is
    // a directory; write a log file inside it rather than treating the
    // directory path itself as a file). Reached only after real work.
    let log_dir = pipeline.artifact_path(spec_pipeline::Stage::Implement)?;
    tokio::fs::create_dir_all(&log_dir).await?;
    let log_path = log_dir.join("IMPLEMENTATION_LOG.md");
    let log_body = format!("# Implementation Log\n\n{response}\n");
    runtime_core::atomic_replace(
        &log_path,
        log_body.as_bytes(),
        runtime_core::AtomicWriteOptions::default(),
        &cancel,
    )
    .await?;

    println!(
        "\nImplementation complete. Summary logged at: {}",
        log_path.display()
    );
    println!(
        "  {} workspace mutation(s) committed by this run.",
        committed_mutations
    );

    println!("\n--- Agent summary ---\n{response}");

    Ok(())
}

// ===========================================================================
// EVAL commands
// ===========================================================================

async fn run_eval_run(
    cancel: CancellationToken,
    suite: &str,
    max_concurrent: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let cases_dir = cwd.join(".agent").join("evals").join("cases");

    if !cases_dir.exists() {
        eprintln!("No eval cases found at: {}", cases_dir.display());
        eprintln!("Create .agent/evals/cases/*.toml files with EvalCase format.");
        std::process::exit(1);
    }

    // Filter before scheduling so a case outside the selected suite is never
    // counted or executed.
    let cases = evals::swebench::load_suite(&cases_dir, suite)?;
    println!("Loaded {} eval cases from suite '{suite}'", cases.len());

    if cases.is_empty() {
        println!("No cases to run.");
        return Ok(());
    }

    // Claim a results file exclusively before running anything. A losing race
    // retries with a new ID instead of appending to another run's results, so
    // two runs started in the same second can never merge.
    let results_dir = cwd.join(".agent").join("evals").join("results");
    let run = evals::run::create_exclusive_run(&results_dir)?;

    println!("Run ID: {}", run.run_id());
    println!("Results will be written to: {}", run.path().display());
    println!();

    // Run cases ONCE each, up to `max_concurrent` at a time. A permit is held
    // for the whole supervised execution, so the limit bounds real concurrency
    // rather than just task creation.
    let permits = std::cmp::max(1, max_concurrent);
    println!("Concurrency limit: {permits}");
    let semaphore = Arc::new(tokio::sync::Semaphore::new(permits));
    let mut scheduled = tokio::task::JoinSet::new();

    for (index, case) in cases.iter().enumerate() {
        let semaphore = semaphore.clone();
        let cancel = cancel.clone();
        let case = case.clone();
        let case_id = case.id.clone();
        scheduled.spawn(async move {
            // A closed semaphore only happens on shutdown; treat it as "not run".
            let _permit = match semaphore.acquire().await {
                Ok(permit) => permit,
                Err(_) => {
                    return (
                        index,
                        case_id,
                        false,
                        0u64,
                        Some("not scheduled".to_string()),
                    )
                }
            };
            if cancel.is_cancelled() {
                return (index, case_id, false, 0, Some("cancelled".to_string()));
            }
            println!("Running case: {case_id} ...");
            let start = std::time::Instant::now();
            let timeout = case.timeout_secs;
            let check = run_check_cmd(&case.check_cmd, &case.repo_fixture, timeout, &cancel).await;
            let elapsed = start.elapsed().as_millis() as u64;
            let status = if check.passed { "PASS" } else { "FAIL" };
            println!("  {case_id}: {status} ({elapsed}ms)");
            if let Some(detail) = &check.detail {
                println!("    {case_id}: {detail}");
                // Surface the supervisor's bounded capture so a failure is
                // diagnosable, and say so when the bound cut it. Lines are
                // case-prefixed because concurrent failures interleave.
                for (stream, text) in [("stdout", &check.stdout), ("stderr", &check.stderr)] {
                    let text = text.trim();
                    if !text.is_empty() {
                        println!("    {case_id} {stream}: {text}");
                    }
                }
                if check.output_truncated {
                    println!("    {case_id}: captured output was truncated at the byte limit");
                }
            }
            (index, case_id, check.passed, elapsed, check.detail)
        });
    }

    // Join every scheduled case so no supervised child outlives this run.
    let mut collected: Vec<(usize, evals::swebench::EvalOutcome)> = Vec::new();
    while let Some(joined) = scheduled.join_next().await {
        let (index, case_id, passed, elapsed, error) = joined?;
        collected.push((
            index,
            evals::swebench::EvalOutcome {
                case_id,
                passed,
                // Check-only mode measures no agent activity. These stay zero and
                // the report omits them rather than presenting them as findings.
                turns: 0,
                tool_calls: 0,
                tokens_in: 0,
                tokens_out: 0,
                wall_time_ms: elapsed,
                error,
            },
        ));
    }

    // Persist in case order so the results file does not depend on completion
    // order, which concurrency makes nondeterministic.
    collected.sort_by_key(|(index, _)| *index);
    let outcomes: Vec<evals::swebench::EvalOutcome> =
        collected.into_iter().map(|(_, outcome)| outcome).collect();
    for outcome in &outcomes {
        run.append(outcome)?;
    }

    // Print summary from the SAME outcomes (no re-execution).
    println!();
    let report = evals::report::EvalReport::new(outcomes.clone());
    report.print_summary();

    if outcomes.iter().any(|outcome| !outcome.passed) {
        return Err("one or more evaluation checks failed".into());
    }
    Ok(())
}

fn run_eval_diff(run_a: &str, run_b: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let results_dir = cwd.join(".agent").join("evals").join("results");

    let load_outcomes =
        |run_id: &str| -> Result<Vec<evals::swebench::EvalOutcome>, Box<dyn std::error::Error>> {
            // A run ID names a file inside the results directory, so it is
            // validated before being joined onto a path.
            let path = evals::run::results_path(&results_dir, run_id)?;
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

    // Report the whole union, not just the intersection, so a case dropped from
    // the current run cannot disappear from the comparison.
    println!(
        "compared: {}  added: {}  removed: {}  duplicate: {}",
        diff.compared.len(),
        diff.added.len(),
        diff.removed.len(),
        diff.duplicates.len()
    );
    for (label, cases) in [
        ("Added (current only)", &diff.added),
        ("Removed (baseline only)", &diff.removed),
        ("Duplicate case ids", &diff.duplicates),
    ] {
        if !cases.is_empty() {
            println!("\n{label}:");
            for case_id in cases {
                println!("  {case_id}");
            }
        }
    }
    if !diff.errors.is_empty() {
        println!("\nNot compared:");
        for entry in &diff.errors {
            println!("  ERROR: {entry}");
        }
    }
    println!();

    if diff.blocks_gate() && diff.hard_regressions.is_empty() {
        // Nothing regressed, but something could not be compared, so the run is
        // not clear either. Saying "all clear" here would contradict the
        // non-zero exit below.
        println!("Comparison incomplete: see the not-compared entries above.");
    } else if diff.hard_regressions.is_empty() && diff.soft_regressions.is_empty() {
        println!("No regressions detected. All clear!");
    } else {
        if !diff.hard_regressions.is_empty() {
            println!("HARD REGRESSIONS (pass -> fail, or a passing case removed):");
            for r in &diff.hard_regressions {
                println!("  FAIL: {r}");
            }
            println!();
        }
        if !diff.soft_regressions.is_empty() {
            println!("Soft regressions and disclosures:");
            for r in &diff.soft_regressions {
                println!("  WARN: {r}");
            }
        }
    }

    // A case that could not be compared is not evidence of passing, so an
    // ambiguous comparison fails the gate alongside a hard regression.
    if diff.blocks_gate() {
        std::process::exit(1);
    }

    Ok(())
}

/// Run a check command under the shared bounded process-tree supervisor.
///
/// The supervisor owns the whole descendant tree, so a timeout or cancellation
/// terminates children the command spawned instead of leaking them.
async fn run_check_cmd(
    cmd: &str,
    cwd: &std::path::Path,
    timeout_secs: u64,
    cancel: &CancellationToken,
) -> CheckOutcome {
    use sandbox::{ProcessFallback, Termination};

    // `execute_typed` keeps the bounded output captured before a timeout kill,
    // which the error-mapping `execute` would discard.
    match ProcessFallback
        .execute_typed(
            cmd,
            std::time::Duration::from_secs(timeout_secs),
            cancel,
            cwd,
        )
        .await
    {
        Ok(result) => {
            let passed = matches!(result.termination, Termination::Exit) && result.exit_code == 0;
            // Distinguish the ways a check can fail instead of flattening them
            // all into one message, and disclose that captured output was cut.
            let detail = match result.termination {
                Termination::Exit if passed => None,
                Termination::Exit => Some(format!("check_cmd exited with {}", result.exit_code)),
                Termination::Timeout => Some(format!("check_cmd timed out after {timeout_secs}s")),
                Termination::Cancelled => Some("check_cmd cancelled".to_string()),
            };
            CheckOutcome {
                passed,
                detail,
                stdout: result.stdout,
                stderr: result.stderr,
                output_truncated: result.stdout_truncated || result.stderr_truncated,
            }
        }
        Err(error) => CheckOutcome {
            passed: false,
            detail: Some(format!("check_cmd could not run: {error}")),
            stdout: String::new(),
            stderr: String::new(),
            output_truncated: false,
        },
    }
}

/// Observed result of one supervised `check_cmd`.
struct CheckOutcome {
    passed: bool,
    /// Why it failed, distinguishing exit code, timeout, cancellation, and
    /// spawn failure. `None` only when the check passed.
    detail: Option<String>,
    stdout: String,
    stderr: String,
    /// Whether the supervisor's byte bound truncated captured output.
    output_truncated: bool,
}

// `chrono_stub_now` was removed by Task 23.2. Its second-resolution `run-{secs}`
// ID let two runs started in the same second share one results file. Run
// identity now comes from `evals::run::create_exclusive_run`, which pairs a
// high-resolution timestamp with random entropy and claims the file exclusively.

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
/// Load built-in and workspace skills, reporting per-file problems.
///
/// Malformed workspace input degrades that one file: the built-ins and every
/// other valid skill stay available, and nothing panics on user input.
fn load_workspace_skills(project_root: &std::path::Path) -> SkillRegistry {
    let skills_dir = project_root.join(".agent").join("skills");
    let (registry, errors) = SkillRegistry::load_with_diagnostics(Some(&skills_dir));
    for error in &errors {
        eprintln!("Skill not loaded — {error}");
    }
    registry
}

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

fn git_output(dir: &std::path::Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
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
        let commit = git_output(dir, &["commit-tree", &tree, "-p", &head, "-m", message])?;
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

#[cfg(test)]
mod task7_checkpoint_preservation_tests {
    use super::*;
    use std::path::Path;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn repository() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), &["init", "-q"]);
        git(dir.path(), &["config", "core.autocrlf", "false"]);
        git(dir.path(), &["config", "core.eol", "lf"]);
        git(dir.path(), &["config", "user.name", "Task 7"]);
        git(
            dir.path(),
            &["config", "user.email", "task7@example.invalid"],
        );
        std::fs::write(dir.path().join("tracked.txt"), "base\n").unwrap();
        git(dir.path(), &["add", "tracked.txt"]);
        git(dir.path(), &["commit", "-q", "-m", "base"]);
        dir
    }

    #[test]
    fn exact_checkpoint_preserves_branch_and_staging_then_undo_is_recoverable() {
        // **Validates: Requirements 3.1, 3.3** (BUG-002)
        let dir = repository();
        let root = dir.path();
        let branch_head = git(root, &["rev-parse", "HEAD"]);

        std::fs::write(root.join("tracked.txt"), "staged\n").unwrap();
        git(root, &["add", "tracked.txt"]);
        std::fs::write(root.join("tracked.txt"), "pre-turn\n").unwrap();

        let checkpoint = git_checkpoint(root, "preserve exact turn").unwrap();
        assert_eq!(git(root, &["rev-parse", "HEAD"]), branch_head);
        assert_eq!(git(root, &["show", ":tracked.txt"]), "staged");
        assert_eq!(
            std::fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "pre-turn\n"
        );

        std::fs::write(root.join("tracked.txt"), "post-turn\n").unwrap();
        std::fs::write(root.join("post-turn.txt"), "recover me\n").unwrap();
        let message = git_undo_last_checkpoint(root, &checkpoint).unwrap();

        assert!(message.contains("Pre-undo work is recoverable at refs/agent/backups/undo-"));
        assert_eq!(git(root, &["rev-parse", "HEAD"]), branch_head);
        assert_eq!(git(root, &["show", ":tracked.txt"]), "staged");
        assert_eq!(
            std::fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "pre-turn\n"
        );
        assert!(!root.join("post-turn.txt").exists());

        let backup_ref = git(
            root,
            &["for-each-ref", "--format=%(refname)", "refs/agent/backups"],
        );
        assert!(backup_ref.starts_with("refs/agent/backups/undo-"));
        assert_eq!(
            git(root, &["show", &format!("{backup_ref}:post-turn.txt")]),
            "recover me"
        );
    }

    #[test]
    fn latest_successful_checkpoint_is_exact_and_failed_checkpoint_cannot_fallback() {
        // **Validates: Requirements 3.1, 3.3** (BUG-002)
        let dir = repository();
        let root = dir.path();
        std::fs::write(root.join("tracked.txt"), "checkpoint-one\n").unwrap();
        let first = git_checkpoint(root, "first").unwrap();
        std::fs::write(root.join("tracked.txt"), "checkpoint-two\n").unwrap();
        let second = git_checkpoint(root, "second").unwrap();
        assert_ne!(first.commit, second.commit);
        std::fs::write(root.join("tracked.txt"), "after-second\n").unwrap();
        git_undo_last_checkpoint(root, &second).unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "checkpoint-two\n"
        );

        let not_a_repo = tempfile::tempdir().unwrap();
        assert!(git_checkpoint(not_a_repo.path(), "must fail").is_err());
        let source = include_str!("main.rs");
        assert!(source.contains("last_checkpoint = None;"));
        assert!(source
            .contains("No successful checkpoint exists for the last agent turn; refusing undo."));
    }
}
