//! OpenAI-compatible streaming provider.
//!
//! A single provider that works with ANY OpenAI-compatible API:
//! - OpenAI (GPT-5.6 Sol, GPT-5.5, GPT-5.4, GPT-5)
//! - Mistral (Medium 3.5, Small 4, Large 3)
//! - DeepSeek (V4-Pro, V4-Flash, V3.1)
//! - Ollama (Llama 3.3, Qwen 3, Mistral local — FREE, offline)
//! - Any other OpenAI-compatible endpoint (Together, Groq, etc.)
//!
//! Users configure via environment variables:
//!   OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

use agent_types::{AgentError, ContentBlock, Message, Result, Role, ToolSchema};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::provider::{LlmProvider, SseEvent, StopReason};
use crate::secret;
use crate::sse::SseParser;

/// Well-known base URLs for popular providers (for documentation/config help).
pub mod endpoints {
    pub const OPENAI: &str = "https://api.openai.com/v1";
    pub const MISTRAL: &str = "https://api.mistral.ai/v1";
    pub const DEEPSEEK: &str = "https://api.deepseek.com";
    pub const OLLAMA: &str = "http://localhost:11434/v1";
}

/// Latest model IDs per provider (July 2026).
pub mod models {
    // OpenAI
    pub const GPT_5_6_SOL: &str = "gpt-5.6-sol";
    pub const GPT_5_5: &str = "gpt-5.5";
    pub const GPT_5_4: &str = "gpt-5.4";
    pub const GPT_5: &str = "gpt-5";

    // Mistral
    pub const MISTRAL_MEDIUM_3_5: &str = "mistral-medium-3.5";
    pub const MISTRAL_SMALL_4: &str = "mistral-small-4";
    pub const MISTRAL_LARGE_3: &str = "mistral-large-3";

    // DeepSeek
    pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";
    pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
    pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
    pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";

    // Ollama (local)
    pub const LLAMA_3_3: &str = "llama3.3";
    pub const QWEN_3: &str = "qwen3";
}

pub struct OpenAiCompatProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    /// `None` means infer from the model name.
    token_limit_field: Option<TokenLimitField>,
}

/// Which field an OpenAI-compatible endpoint accepts for the output budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenLimitField {
    /// Classic field, understood by older OpenAI models and by Ollama,
    /// Mistral, and DeepSeek.
    MaxTokens,
    /// Required by reasoning models, which reject `max_tokens` with
    /// `unsupported_parameter`.
    MaxCompletionTokens,
}

impl TokenLimitField {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }

    /// Infer from a model name. Matching is on the lowercased name so
    /// `GPT-5-Mini` behaves like `gpt-5-mini`.
    pub fn infer(model: &str) -> Self {
        let model = model.to_ascii_lowercase();
        let reasoning = ["gpt-5", "o1", "o3", "o4"]
            .iter()
            .any(|family| model == *family || model.starts_with(&format!("{family}-")));
        if reasoning {
            Self::MaxCompletionTokens
        } else {
            Self::MaxTokens
        }
    }

    /// Parse an explicit override, accepting either field name directly.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "max_tokens" => Some(Self::MaxTokens),
            "max_completion_tokens" => Some(Self::MaxCompletionTokens),
            _ => None,
        }
    }
}

impl OpenAiCompatProvider {
    /// Create a new OpenAI-compatible provider.
    ///
    /// # Examples
    /// ```ignore
    /// // OpenAI
    /// OpenAiCompatProvider::new("sk-...", "gpt-5.5", "https://api.openai.com/v1");
    /// // Mistral
    /// OpenAiCompatProvider::new("key", "mistral-medium-3.5", "https://api.mistral.ai/v1");
    /// // DeepSeek
    /// OpenAiCompatProvider::new("key", "deepseek-v4-pro", "https://api.deepseek.com");
    /// // Ollama (local, no key needed)
    /// OpenAiCompatProvider::new("", "llama3.3", "http://localhost:11434/v1");
    /// ```
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            max_tokens: 16384,
            token_limit_field: None,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Force the output-budget field name instead of inferring it.
    ///
    /// Needed because Azure sends a **deployment name** in `model`, and a
    /// deployment can be called anything. If the deployment name does not look
    /// like the underlying model, inference cannot know which field the endpoint
    /// accepts, so the caller must be able to say so.
    pub fn with_token_limit_field(mut self, field: TokenLimitField) -> Self {
        self.token_limit_field = Some(field);
        self
    }

    /// Which JSON field carries the output-token budget.
    ///
    /// Reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`) **reject** `max_tokens`
    /// outright with `unsupported_parameter` and require
    /// `max_completion_tokens`. Older and non-OpenAI compatible endpoints
    /// (Ollama, Mistral, DeepSeek) only understand `max_tokens`. Sending the
    /// wrong one fails every request, so this is not cosmetic.
    fn token_limit_field(&self) -> &'static str {
        self.token_limit_field
            .unwrap_or_else(|| TokenLimitField::infer(&self.model))
            .as_str()
    }

    fn build_body(&self, messages: &[Message], tools: &[ToolSchema]) -> Value {
        let mut api_messages: Vec<Value> = Vec::new();

        for m in messages {
            match m.role {
                Role::System => {
                    let text = extract_text(&m.content);
                    api_messages.push(json!({
                        "role": "system",
                        "content": text
                    }));
                }
                Role::User => {
                    let text = extract_text(&m.content);
                    api_messages.push(json!({
                        "role": "user",
                        "content": text
                    }));
                }
                Role::Assistant => {
                    let mut msg = json!({"role": "assistant"});
                    let text = extract_text(&m.content);
                    if !text.is_empty() {
                        msg["content"] = json!(text);
                    }
                    // Tool calls in assistant message.
                    let tool_calls: Vec<Value> = m
                        .content
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::ToolUse {
                                id, name, input, ..
                            } = b
                            {
                                Some(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": input.to_string()
                                    }
                                }))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = json!(tool_calls);
                    }
                    api_messages.push(msg);
                }
                Role::Tool => {
                    for b in &m.content {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error: _,
                        } = b
                        {
                            api_messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": output
                            }));
                        }
                    }
                }
            }
        }

        let mut body = json!({
            "model": self.model,
            "messages": api_messages,
            "stream": true,
        });
        body[self.token_limit_field()] = json!(self.max_tokens);

        // Official OpenAI requires this flag to emit the final usage-only
        // streaming chunk. Avoid sending it to stricter compatible endpoints.
        if self.base_url.contains("api.openai.com") {
            body["stream_options"] = json!({"include_usage": true});
        }

        if !tools.is_empty() {
            let tool_defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tool_defs);
        }

        body
    }
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| {
            if let ContentBlock::Text(t) = b {
                Some(t.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiCompatProvider {
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        cancel: &CancellationToken,
    ) -> Result<mpsc::Receiver<SseEvent>> {
        let body = self.build_body(messages, tools);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header("HTTP-Referer", "https://newgen-cli.dev")
            .header("X-Title", "NewGen CLI");

        // Only add auth header if key is non-empty (Ollama doesn't need one).
        if !self.api_key.is_empty() {
            req = req.header("authorization", format!("Bearer {}", self.api_key));
        }

        let resp = req
            .body(serde_json::to_vec(&body).map_err(|e| AgentError::Llm(e.to_string()))?)
            .send()
            .await
            .map_err(|e| AgentError::Llm(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AgentError::Llm(format!("http {status}: {text}")));
        }

        let (tx, rx) = mpsc::channel(64);
        let child = cancel.child_token();
        let endpoint = secret::sanitized_endpoint(&url);
        let redaction_key = self.api_key.clone();

        tokio::spawn(async move {
            let mut stream = resp.bytes_stream();
            let mut parser = SseParser::new();
            // Accumulate tool call arguments across multiple deltas.
            let mut tool_calls: std::collections::HashMap<u32, (String, String, String)> =
                std::collections::HashMap::new(); // index → (id, name, args_json)
            let mut pending_stop: Option<StopReason> = None;

            loop {
                tokio::select! {
                    biased;
                    _ = child.cancelled() => {
                        let _ = tx.send(SseEvent::Cancelled).await;
                        return;
                    },
                    next = stream.next() => match next {
                        None => {
                            // A retained partial frame means the response was
                            // truncated; it must never be reported as complete.
                            if let Err(error) = parser.finish() {
                                let _ = tx.send(SseEvent::Error(error.to_string())).await;
                                return;
                            }
                            // Routers may close without a `[DONE]` marker. Flush
                            // accumulated tool calls before the terminal event so
                            // a complete turn is not silently downgraded.
                            let had_tool_calls = !tool_calls.is_empty();
                            if !emit_pending_tools(&mut tool_calls, &tx).await {
                                return;
                            }
                            let reason = if had_tool_calls {
                                StopReason::ToolUse
                            } else {
                                pending_stop.unwrap_or(StopReason::EndTurn)
                            };
                            let _ = tx.send(SseEvent::Stop { reason }).await;
                            return;
                        }
                        Some(Err(error)) => {
                            // Render only a failure category; reqwest's own
                            // Display includes the request URL.
                            let _ = tx.send(SseEvent::Error(format!(
                                "chat completion stream from {endpoint}: {}",
                                secret::transport_error_kind(&error)
                            ))).await;
                            return;
                        }
                        Some(Ok(bytes)) => {
                            let frames = match parser.feed(&bytes) {
                                Ok(frames) => frames,
                                Err(error) => {
                                    let _ = tx.send(SseEvent::Error(error.to_string())).await;
                                    return;
                                }
                            };
                            for frame in frames {
                                // Frames already buffered in this segment must not
                                // outrace cancellation into a Stop event.
                                if child.is_cancelled() {
                                    let _ = tx.send(SseEvent::Cancelled).await;
                                    return;
                                }

                                let data = frame.data.trim();

                                // Keep-alive and comment-only frames carry no
                                // payload and must not end the turn.
                                if data.is_empty() {
                                    continue;
                                }

                                if data == "[DONE]" {
                                    let had_tool_calls = !tool_calls.is_empty();
                                    if !emit_pending_tools(&mut tool_calls, &tx).await {
                                        return;
                                    }
                                    let reason = if had_tool_calls {
                                        StopReason::ToolUse
                                    } else {
                                        pending_stop.unwrap_or(StopReason::EndTurn)
                                    };
                                    let _ = tx.send(SseEvent::Stop { reason }).await;
                                    return;
                                }

                                let chunk: Value = match serde_json::from_str(data) {
                                    Ok(value) => value,
                                    Err(error) => {
                                        let _ = tx
                                            .send(SseEvent::Error(format!(
                                                "invalid JSON in complete SSE frame: {error}"
                                            )))
                                            .await;
                                        return;
                                    }
                                };

                                if let Some(error) = chunk.get("error") {
                                    let message = error
                                        .get("message")
                                        .and_then(Value::as_str)
                                        .unwrap_or("provider returned an error");
                                    // Provider-controlled text is untrusted and
                                    // may echo the credential, so redact and bound it.
                                    let _ = tx.send(SseEvent::Error(
                                        secret::bounded_redacted_body(message, &redaction_key),
                                    )).await;
                                    return;
                                }

                                // Usage may arrive in a dedicated chunk with no
                                // choices/delta, so process it first.
                                if let Some(usage) = chunk.get("usage") {
                                    let prompt = usage
                                        .get("prompt_tokens")
                                        .and_then(Value::as_u64)
                                        .unwrap_or(0) as u32;
                                    let completion = usage
                                        .get("completion_tokens")
                                        .and_then(Value::as_u64)
                                        .unwrap_or(0) as u32;
                                    let total = usage
                                        .get("total_tokens")
                                        .and_then(Value::as_u64)
                                        .unwrap_or(0) as u32;
                                    if total > 0 {
                                        let _ = tx.send(SseEvent::Usage {
                                            prompt_tokens: prompt,
                                            completion_tokens: completion,
                                            total_tokens: total,
                                        }).await;
                                    }
                                }

                                let choice = chunk.get("choices").and_then(|c| c.get(0));
                                if let Some(delta) = choice.and_then(|c| c.get("delta")) {
                                    if let Some(content) = delta.get("content").and_then(Value::as_str) {
                                        if !content.is_empty()
                                            && tx.send(SseEvent::Delta(content.to_string())).await.is_err()
                                        {
                                            return;
                                        }
                                    }

                                    if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                                        for tc in tcs {
                                            let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                                            let entry = tool_calls.entry(idx).or_insert_with(|| {
                                                let id = tc.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                                                let name = tc.get("function")
                                                    .and_then(|f| f.get("name"))
                                                    .and_then(Value::as_str)
                                                    .unwrap_or("")
                                                    .to_string();
                                                (id, name, String::new())
                                            });
                                            if let Some(args) = tc.get("function")
                                                .and_then(|f| f.get("arguments"))
                                                .and_then(Value::as_str)
                                            {
                                                entry.2.push_str(args);
                                            }
                                        }
                                    }
                                }

                                if let Some(reason) = choice
                                    .and_then(|c| c.get("finish_reason"))
                                    .and_then(Value::as_str)
                                {
                                    pending_stop = Some(match reason {
                                        "tool_calls" | "function_call" => StopReason::ToolUse,
                                        "length" => StopReason::MaxTokens,
                                        _ => StopReason::EndTurn,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// Emit accumulated tool calls as SseEvent::ToolUse in provider index order.
///
/// Returns `false` when an accumulated argument buffer is not complete JSON. A
/// truncated argument stream must not be presented as a complete tool turn, so
/// the caller emits an error instead of a terminal `Stop`.
async fn emit_pending_tools(
    tool_calls: &mut std::collections::HashMap<u32, (String, String, String)>,
    tx: &mpsc::Sender<SseEvent>,
) -> bool {
    let mut sorted: Vec<(u32, (String, String, String))> = tool_calls.drain().collect();
    sorted.sort_by_key(|(idx, _)| *idx);
    for (_idx, (id, name, args_json)) in sorted {
        // A tool called with no arguments legitimately accumulates nothing.
        let input: Value = if args_json.trim().is_empty() {
            json!({})
        } else {
            match serde_json::from_str(&args_json) {
                Ok(input) => input,
                Err(error) => {
                    let _ = tx
                        .send(SseEvent::Error(format!(
                            "incomplete tool call arguments for '{name}': {error}"
                        )))
                        .await;
                    return false;
                }
            }
        };
        let _ = tx
            .send(SseEvent::ToolUse {
                id,
                name,
                input,
                provider_metadata: None,
            })
            .await;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::testing::spawn_http_capture;
    use std::time::Duration;

    async fn collect(mut events: mpsc::Receiver<SseEvent>) -> Vec<SseEvent> {
        let mut collected = Vec::new();
        while let Some(event) = events.recv().await {
            collected.push(event);
        }
        collected
    }

    async fn stream_fixture(body: Vec<u8>, cancel: &CancellationToken) -> Vec<SseEvent> {
        let (base_url, server) =
            spawn_http_capture("200 OK", "text/event-stream", body, Duration::from_secs(5)).await;
        let provider = OpenAiCompatProvider::new("", "fixture", base_url);
        let events = provider.stream(&[], &[], cancel).await.unwrap();
        let collected = collect(events).await;
        let _ = server.await;
        collected
    }

    #[tokio::test]
    async fn stream_close_without_done_still_flushes_tool_calls() {
        // **Validates: Requirements 2.36, 3.10**
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",",
            "\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n"
        );

        let events = stream_fixture(body.as_bytes().to_vec(), &CancellationToken::new()).await;
        assert!(matches!(
            &events[0],
            SseEvent::ToolUse { id, name, input, .. }
                if id == "call-1" && name == "read_file" && input["path"] == "a.rs"
        ));
        assert!(matches!(
            events.last().unwrap(),
            SseEvent::Stop {
                reason: StopReason::ToolUse
            }
        ));
    }

    #[tokio::test]
    async fn truncated_stream_is_an_error_not_a_completion() {
        // **Validates: Requirements 2.36, 2.37**
        // The final frame has no terminating blank line, so the response was cut.
        let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}".to_vec();

        let events = stream_fixture(body, &CancellationToken::new()).await;
        assert!(!events
            .iter()
            .any(|event| matches!(event, SseEvent::Stop { .. })));
        assert!(matches!(
            events.last().unwrap(),
            SseEvent::Error(message) if message.contains("incomplete")
        ));
    }

    #[tokio::test]
    async fn cancellation_emits_exactly_one_terminal_cancelled_and_never_stop() {
        // **Validates: Requirements 2.36**
        let cancel = CancellationToken::new();
        cancel.cancel();
        let body =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"ignored\"}}]}\n\ndata: [DONE]\n\n"
                .to_vec();

        let events = stream_fixture(body, &cancel).await;
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, SseEvent::Cancelled))
                .count(),
            1
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, SseEvent::Stop { .. })));
    }

    #[tokio::test]
    async fn keep_alive_frames_do_not_end_the_turn() {
        // **Validates: Requirements 3.10**
        let body = concat!(
            "event: ping\n\n",
            ": comment\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"kept\"}}]}\n\n",
            "data: [DONE]\n\n"
        );

        let events = stream_fixture(body.as_bytes().to_vec(), &CancellationToken::new()).await;
        assert!(!events
            .iter()
            .any(|event| matches!(event, SseEvent::Error(_))));
        assert!(events
            .iter()
            .any(|event| matches!(event, SseEvent::Delta(text) if text == "kept")));
        assert!(matches!(
            events.last().unwrap(),
            SseEvent::Stop {
                reason: StopReason::EndTurn
            }
        ));
    }

    #[tokio::test]
    async fn truncated_tool_arguments_are_not_presented_as_a_complete_tool_turn() {
        // **Validates: Requirements 2.36**
        // Arguments stop mid-JSON and the router closes without `[DONE]`.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",",
            "\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n"
        );

        let events = stream_fixture(body.as_bytes().to_vec(), &CancellationToken::new()).await;
        assert!(!events
            .iter()
            .any(|event| matches!(event, SseEvent::Stop { .. })));
        assert!(!events
            .iter()
            .any(|event| matches!(event, SseEvent::ToolUse { .. })));
        assert!(matches!(
            events.last().unwrap(),
            SseEvent::Error(message) if message.contains("incomplete tool call arguments")
        ));
    }

    #[tokio::test]
    async fn in_stream_provider_errors_are_redacted_and_bounded() {
        // **Validates: Requirements 2.25**
        let secret = "OPENAI_STREAM_SECRET";
        let long_detail = "detail ".repeat(2048);
        let chunk = json!({
            "error": { "message": format!("rejected {secret} {long_detail}") }
        });
        let body = format!("data: {chunk}\n\n");

        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "text/event-stream",
            body.into_bytes(),
            Duration::from_secs(5),
        )
        .await;
        let provider = OpenAiCompatProvider::new(secret, "fixture", base_url);
        let events = collect(
            provider
                .stream(&[], &[], &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        let _ = server.await;

        let message = match events.last().unwrap() {
            SseEvent::Error(message) => message,
            other => panic!("expected error event, got {other:?}"),
        };
        assert!(!message.contains(secret));
        assert!(message.contains("[redacted]"));
        assert!(message.ends_with("...[truncated]"));
    }

    #[test]
    fn foreign_provider_metadata_never_reaches_the_openai_request() {
        // **Validates: Requirements 2.38, 3.10**
        let assistant = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "read_file".into(),
                input: json!({"path": "src/lib.rs"}),
                provider_metadata: Some(json!({ "thought_signature": "gemini-only-value" })),
            }],
            token_estimate: 0,
        };

        let body = OpenAiCompatProvider::new("unused", "fixture", "https://example.com/v1")
            .build_body(&[assistant], &[]);
        let rendered = body.to_string();
        assert!(!rendered.contains("gemini-only-value"));
        assert!(!rendered.contains("thought_signature"));
        assert!(!rendered.contains("thoughtSignature"));
        // Executable arguments are still serialized for the provider.
        assert!(rendered.contains("src/lib.rs"));
    }

    #[test]
    fn body_builds_with_system_and_tools() {
        let p = OpenAiCompatProvider::new("key", "gpt-5.5", "https://api.openai.com/v1");
        let msgs = vec![
            Message {
                role: Role::System,
                content: vec![ContentBlock::Text("be helpful".into())],
                token_estimate: 0,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text("hi".into())],
                token_estimate: 0,
            },
        ];
        let tools = vec![ToolSchema {
            name: "read_file".into(),
            description: "read a file".into(),
            input_schema: json!({"type": "object"}),
            effects: Default::default(),
        }];
        let body = p.build_body(&msgs, &tools);
        assert_eq!(body["model"], "gpt-5.5");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert!(body["tools"].is_array());
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn tool_result_maps_correctly() {
        let p = OpenAiCompatProvider::new("k", "m", "http://x");
        let msgs = vec![Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_123".into(),
                output: "file content here".into(),
                is_error: false,
            }],
            token_estimate: 0,
        }];
        let body = p.build_body(&msgs, &[]);
        assert_eq!(body["messages"][0]["role"], "tool");
        assert_eq!(body["messages"][0]["tool_call_id"], "call_123");
        assert_eq!(body["messages"][0]["content"], "file content here");
    }

    #[test]
    fn no_auth_header_for_empty_key() {
        // Ollama doesn't need auth — empty key should not send header.
        let p = OpenAiCompatProvider::new("", "llama3.3", "http://localhost:11434/v1");
        assert!(p.api_key.is_empty());
    }
}

#[cfg(test)]
mod token_limit_field_tests {
    use super::*;
    use agent_types::{ContentBlock, Message, Role};

    fn user(text: &str) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text(text.into())],
            token_estimate: 0,
        }]
    }

    fn body_for(model: &str) -> Value {
        OpenAiCompatProvider::new("k", model, "https://example.invalid/openai/v1")
            .build_body(&user("hi"), &[])
    }

    #[test]
    fn reasoning_models_use_max_completion_tokens() {
        // Live evidence: an Azure `gpt-5-mini` deployment rejects `max_tokens`
        // with `unsupported_parameter` and requires `max_completion_tokens`, so
        // sending the classic field fails every request rather than degrading.
        for model in [
            "gpt-5",
            "gpt-5-mini",
            "GPT-5-Mini",
            "gpt-5-nano",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "o4-mini",
        ] {
            let body = body_for(model);
            assert!(
                body.get("max_completion_tokens").is_some(),
                "{model} must send max_completion_tokens"
            );
            assert!(
                body.get("max_tokens").is_none(),
                "{model} must not also send max_tokens"
            );
        }
    }

    #[test]
    fn classic_and_third_party_models_keep_max_tokens() {
        // Ollama, Mistral, and DeepSeek only understand `max_tokens`, so the
        // fix must not regress the endpoints that already worked.
        for model in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-35-turbo",
            "mistral-large-latest",
            "deepseek-chat",
            "llama3",
            // Prefix-only lookalikes must not be misread as reasoning models.
            "o1pen-model",
            "gpt-50-legacy",
        ] {
            let body = body_for(model);
            assert!(
                body.get("max_tokens").is_some(),
                "{model} must send max_tokens"
            );
            assert!(
                body.get("max_completion_tokens").is_none(),
                "{model} must not send max_completion_tokens"
            );
        }
    }

    #[test]
    fn explicit_override_wins_over_inference() {
        // Azure puts a *deployment* name in `model`, and a deployment can be
        // named anything, so inference alone cannot be correct in general.
        let body =
            OpenAiCompatProvider::new("k", "my-private-deploy", "https://example.invalid/v1")
                .with_token_limit_field(TokenLimitField::MaxCompletionTokens)
                .build_body(&user("hi"), &[]);
        assert!(body.get("max_completion_tokens").is_some());
        assert!(body.get("max_tokens").is_none());

        let body = OpenAiCompatProvider::new("k", "gpt-5-mini", "https://example.invalid/v1")
            .with_token_limit_field(TokenLimitField::MaxTokens)
            .build_body(&user("hi"), &[]);
        assert!(body.get("max_tokens").is_some());
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn override_parsing_accepts_either_field_name_and_rejects_junk() {
        assert_eq!(
            TokenLimitField::parse("max_tokens"),
            Some(TokenLimitField::MaxTokens)
        );
        assert_eq!(
            TokenLimitField::parse("  MAX_COMPLETION_TOKENS  "),
            Some(TokenLimitField::MaxCompletionTokens)
        );
        assert_eq!(TokenLimitField::parse(""), None);
        assert_eq!(TokenLimitField::parse("max-tokens"), None);
        assert_eq!(TokenLimitField::parse("unlimited"), None);
    }

    #[test]
    fn the_budget_value_is_carried_through_either_field() {
        let body = OpenAiCompatProvider::new("k", "gpt-5-mini", "https://example.invalid/v1")
            .with_max_tokens(4242)
            .build_body(&user("hi"), &[]);
        assert_eq!(body["max_completion_tokens"], 4242);

        let body = OpenAiCompatProvider::new("k", "gpt-4o", "https://example.invalid/v1")
            .with_max_tokens(4242)
            .build_body(&user("hi"), &[]);
        assert_eq!(body["max_tokens"], 4242);
    }
}
