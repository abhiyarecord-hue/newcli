# Contributing to NewGen CLI

Thanks for your interest in contributing! This is an early-alpha project and contributions are very welcome.

## Getting Started

```bash
git clone https://github.com/abhiyarecord-hue/newcli.git
cd newcli
cargo build --workspace
cargo test --workspace
```

## Requirements

- Rust 1.75+ (edition 2021)
- A C compiler (for tree-sitter grammars and sqlite)

## What We Need Help With

### High Priority
- **Testing LLM providers** — We only test Gemini. If you have OpenAI/Anthropic/Mistral/DeepSeek keys, please test those providers and report results (or fix bugs).
- **Cross-platform testing** — The project was primarily developed on Windows. Linux/macOS test reports are valuable.
- **SWE-bench evaluation** — Help run the benchmark cases in `swe bench test/` and publish results.

### Adding a New LLM Provider
Implement the `LlmProvider` trait in `crates/llm-client/`. See `openai_compat.rs` for the OpenAI-compatible pattern, or `gemini.rs` / `anthropic.rs` for provider-specific formats.

### Adding a New Tool
Implement the `Tool` trait in `crates/agent-core/src/builtin.rs`. All file access must go through `PathJail`. See existing tools for the pattern.

## Code Standards

- No `unwrap()`/`expect()`/`panic!` in library code (outside tests) — return `Result` instead
- All async code uses `tokio` with cancellation tokens
- Run `cargo fmt` and `cargo clippy` before submitting
- Add tests for new functionality

## Pull Request Process

1. Fork and create a feature branch
2. Make your changes with tests
3. Ensure `cargo test --workspace` passes
4. Submit a PR with a clear description

## Reporting Bugs

Open an issue with:
- What you did
- What you expected
- What actually happened
- Your OS and Rust version
