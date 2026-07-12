//! Quick test: calls Gemini 2.5 Flash with a simple prompt and prints the streamed response.
//!
//! Run with:
//!   cargo run --example test_gemini -p llm-client

use llm_client::{GeminiProvider, LlmProvider, SseEvent};
use tokio_util::sync::CancellationToken;

// Test API key (will be deleted after testing)
const API_KEY: &str = "";
const MODEL: &str = "gemini-3.5-flash";

#[tokio::main]
async fn main() {
    println!("=== Gemini 3.5 Flash API Test ===\n");

    let provider = GeminiProvider::new(API_KEY, MODEL);
    let cancel = CancellationToken::new();

    // Simple test message
    let messages = vec![agent_types::Message {
        role: agent_types::Role::User,
        content: vec![agent_types::ContentBlock::Text(
            "Hello! Tell me a short joke about programming in Rust. Keep it to 2-3 sentences."
                .to_string(),
        )],
        token_estimate: 0,
    }];

    println!("Sending request to Gemini 3.5 Flash...\n");

    match provider.stream(&messages, &[], &cancel).await {
        Ok(mut rx) => {
            print!("Response: ");
            let mut got_response = false;
            while let Some(event) = rx.recv().await {
                match event {
                    SseEvent::Delta(text) => {
                        print!("{text}");
                        got_response = true;
                    }
                    SseEvent::Stop { reason } => {
                        println!("\n\n[Stream ended — reason: {reason:?}]");
                        break;
                    }
                    SseEvent::ToolUse { name, .. } => {
                        println!("\n[Tool call: {name}]");
                    }
                    SseEvent::Usage { total_tokens, .. } => {
                        println!("\n[Tokens used: {total_tokens}]");
                    }
                    SseEvent::Thinking(thought) => {
                        let line = thought.lines().next().unwrap_or("");
                        println!("\n[Thinking: {line}]");
                    }
                    SseEvent::Error(e) => {
                        eprintln!("\n\nERROR: {e}");
                        break;
                    }
                }
            }
            if !got_response {
                println!("(no text received)");
            }
        }
        Err(e) => {
            eprintln!("Failed to start stream: {e}");
        }
    }

    println!("\n=== Test Complete ===");
}
