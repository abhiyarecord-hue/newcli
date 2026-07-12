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
        Commands::Index => {
            println!("Indexing codebase...");
            println!("(Merkle diff → parse → chunk → embed → store)");
            println!("Index complete. 0 files changed (stub).");
        }
        Commands::Spec { stage } => {
            println!("Running spec stage: {stage}");
            println!("(Spec pipeline stub — full wiring in TASK-9.2)");
        }
        Commands::Search { query, top_k } => {
            println!("Searching for: \"{query}\" (top {top_k})");
            println!("(Search stub — full wiring after vector store integration)");
        }
        Commands::Eval { action } => match action {
            EvalAction::Run { suite, max_concurrent } => {
                println!("Running eval suite '{suite}' with max_concurrent={max_concurrent}");
            }
            EvalAction::Baseline { run_id } => {
                println!("Setting baseline: {run_id}");
            }
            EvalAction::Diff { run_a, run_b } => {
                println!("Diffing runs: {run_a} vs {run_b}");
            }
        },
    }
    Ok(())
}

/// Interactive chat loop powered by Gemini 3.5 Flash via the Orchestrator.
async fn run_chat(cancel: CancellationToken, workspace: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve workspace: --workspace flag > current directory.
    let project_root = match workspace {
        Some(ref dir) => std::path::PathBuf::from(dir),
        None => std::env::current_dir()?,
    };
    // Canonicalize so paths display cleanly.
    let project_root = std::fs::canonicalize(&project_root).unwrap_or(project_root);

    // Resolve the API key: prefer the GEMINI_API_KEY env var, fall back to the
    // built-in test key. The CLI resolves the key here — the llm-client library
    // never reads env vars itself.
    const FALLBACK_TEST_KEY: &str = "";
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| FALLBACK_TEST_KEY.to_string());
    let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-3.5-flash".to_string());

    let provider = Arc::new({
        let p = GeminiProvider::new(api_key, model.clone());
        // If user explicitly wants AI Studio (old free-tier endpoint), set GEMINI_USE_AI_STUDIO=1
        if std::env::var("GEMINI_USE_AI_STUDIO").unwrap_or_default() == "1" {
            p.with_base_url("https://generativelanguage.googleapis.com/v1beta")
        } else {
            p // default is Vertex AI Express Mode
        }
    });
    let hooks = Arc::new(HookEngine::new(vec![
        Arc::new(harness::SecretLeakHook::new()),
        Arc::new(harness::DestructiveCommandHook::new()),
    ]));
    // Wire the built-in tools (read_file, write_file, list_files, search_text, bash).
    let dispatcher = Arc::new(ToolDispatcher::new(agent_core::default_tools(), hooks));
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

    let mut orchestrator = Orchestrator::new(
        provider,
        dispatcher,
        skills,
        event_bus,
        cancel.clone(),
        agent_types::LanguageMode::Hinglish,
    )
    .with_project_root(project_root.clone())
    .with_system_prompt(
        "You are an autonomous AI coding agent. Your primary job is to WRITE CODE using tools.\n\
         RULES:\n\
         - When the user asks you to build/create/make something, IMMEDIATELY use write_file to create the files. Do NOT just describe what you would do.\n\
         - Write COMPLETE, working code. Never leave placeholders or TODOs.\n\
         - For large files: write the full file content in a single write_file call. Do not split across turns.\n\
         - Keep explanations SHORT (2-3 lines max) AFTER writing the files.\n\
         - If a task needs multiple files, write ALL of them in the same turn using multiple write_file calls.\n\
         - Always use tools. Never refuse to write code."
    );

    println!("========================================");
    println!(" Rust AI Coding Agent");
    println!(" model: {model}");
    println!(" workspace: {}", project_root.display());
    println!(" tools: read_file, write_file, list_files, search_text, bash");
    println!(" Type your message. /quit or Ctrl-C to exit.");
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
