//! OpenAI-compatible streaming provider.
//!
//! A single provider that works with ANY OpenAI-compatible API:
//! - OpenAI (GPT-5.6 Sol/Terra/Luna, GPT-5.5, GPT-5.4, GPT-5)
//! - Mistral (Medium 3.5, Small 4, Large 3)
//! - DeepSeek (V4-Pro, V4-Flash, V3.1)
//! - Ollama (Llama 3.3, Qwen 3, Mistral local — FREE, offline)
//! - Any other OpenAI-compatible endpoint (Together, Groq, etc.)
//!
//! Users configure via environment variables:
//!   OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

use std::sync::atomic::{AtomicU8, Ordering};

use agent_types::{AgentError, ContentBlock, Message, Result, Role, ToolSchema};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::provider::{LlmProvider, SseEvent, StopReason};
use crate::secret;
use crate::sse::SseParser;

/// Default cap on generated output tokens.
///
/// Was 16384, which is provably too aggressive. Measured against a live Azure
/// `gpt-5-mini` deployment: a large prompt with a 16384 budget returned
/// `500 Internal server error`, while the same prompt succeeded at 12288, 8192,
/// 4096, and 2048, and a small prompt succeeded at 16384. The failure is the
/// combination of prompt size and output budget, and it broke the spec pipeline
/// from the `plan` stage onward, because that is where earlier artifacts start
/// being fed back in and the prompt grows.
///
/// 8192 keeps a wide margin below the observed failure point while still leaving
/// room for a substantial code edit. Raise it with `LLM_MAX_OUTPUT_TOKENS` when
/// the model and prompt allow.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 8192;

/// Well-known base URLs for popular providers (for documentation/config help).
pub mod endpoints {
    pub const OPENAI: &str = "https://api.openai.com/v1";

    pub const MISTRAL: &str = "https://api.mistral.ai/v1";
    pub const DEEPSEEK: &str = "https://api.deepseek.com";
    pub const OLLAMA: &str = "http://localhost:11434/v1";
}

/// Model IDs verified against vendor documentation in August 2026.
///
/// These are conveniences, not a whitelist: any model name may be passed
/// through, since a vendor can publish a new one at any time and this list will
/// then be behind. The output-budget field is corrected from the endpoint's own
/// response rather than from this list, so an unlisted name still works.
pub mod models {
    // OpenAI. `gpt-5.6` is an alias that routes to `gpt-5.6-sol`; Terra and
    // Luna are the lower-cost members of the same generation, which matters for
    // an agent that issues many calls per task.
    pub const GPT_5_6: &str = "gpt-5.6";
    pub const GPT_5_6_SOL: &str = "gpt-5.6-sol";
    pub const GPT_5_6_TERRA: &str = "gpt-5.6-terra";
    pub const GPT_5_6_LUNA: &str = "gpt-5.6-luna";
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
    /// Field learned from an endpoint's own rejection, so the correction is
    /// paid for once per process rather than on every request.
    ///
    /// `0` unset, `1` `max_tokens`, `2` `max_completion_tokens`. An explicit
    /// [`Self::with_token_limit_field`] still wins: an operator's statement
    /// about their own deployment outranks a guess.
    learned_token_limit_field: AtomicU8,
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
    ///
    /// A family match is the family name followed by end of string, `-`, or
    /// `.`. The dot matters: the family ships point releases such as `gpt-5.5`
    /// and `gpt-5.6-sol`, and treating only `-` as a separator classified every
    /// one of them as a non-reasoning model, so each request was sent with
    /// `max_tokens` and rejected outright. `gpt-50` and `gpt-5x` are still not
    /// matched, since a longer number is a different family, not a variant.
    pub fn infer(model: &str) -> Self {
        let model = model.to_ascii_lowercase();
        let reasoning = ["gpt-5", "o1", "o3", "o4"].iter().any(|family| {
            model
                .strip_prefix(*family)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with(['-', '.']))
        });
        if reasoning {
            Self::MaxCompletionTokens
        } else {
            Self::MaxTokens
        }
    }

    /// The other field, used to recover when an endpoint rejects the one sent.
    pub fn flipped(self) -> Self {
        match self {
            Self::MaxTokens => Self::MaxCompletionTokens,
            Self::MaxCompletionTokens => Self::MaxTokens,
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
            max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            token_limit_field: None,
            learned_token_limit_field: AtomicU8::new(0),
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
    /// Name inference is a guess, and a guess about a model list that keeps
    /// growing will keep going stale, so a rejection observed from the endpoint
    /// itself is remembered and preferred over inferring again.
    fn effective_token_limit_field(&self) -> TokenLimitField {
        if let Some(explicit) = self.token_limit_field {
            return explicit;
        }
        match self.learned_token_limit_field.load(Ordering::Relaxed) {
            1 => TokenLimitField::MaxTokens,
            2 => TokenLimitField::MaxCompletionTokens,
            _ => TokenLimitField::infer(&self.model),
        }
    }

    fn learn_token_limit_field(&self, field: TokenLimitField) {
        let encoded = match field {
            TokenLimitField::MaxTokens => 1,
            TokenLimitField::MaxCompletionTokens => 2,
        };
        self.learned_token_limit_field
            .store(encoded, Ordering::Relaxed);
    }

    fn build_body(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        token_limit_field: TokenLimitField,
    ) -> Value {
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
        body[token_limit_field.as_str()] = json!(self.max_tokens);

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

/// Does an error body say that the output-budget field just sent is the
/// unsupported parameter?
///
/// Deliberately narrow. It requires the field's own name and a rejection
/// phrase, so an unrelated 400 — a malformed tool schema, an over-long
/// context — is not mistaken for this one and retried pointlessly. The
/// observed OpenAI wording is `Unsupported parameter: 'max_tokens' is not
/// supported with this model. Use 'max_completion_tokens' instead.`
fn indicates_unsupported_token_field(body: &str, sent: TokenLimitField) -> bool {
    let body = body.to_ascii_lowercase();
    if !body.contains(sent.as_str()) {
        return false;
    }
    [
        "unsupported parameter",
        "unsupported_parameter",
        "not supported",
    ]
    .iter()
    .any(|phrase| body.contains(phrase))
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
        let url = format!("{}/chat/completions", self.base_url);
        let mut token_limit_field = self.effective_token_limit_field();

        // At most two attempts, and the second one only after the endpoint
        // itself says the output-budget field is unsupported. Naming a model
        // can never keep pace with a vendor's list, so the endpoint's own
        // verdict is used to correct the guess instead of failing the run.
        let resp = loop {
            let body = self.build_body(messages, tools, token_limit_field);

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
                // `reqwest::Error`'s own Display embeds the request URL and hides the
                // failure category behind a generic "error sending request". That is
                // both a diagnosability problem, since timeout and connect failure
                // look identical, and a hygiene problem, since a provider that
                // carries credentials in the query string would leak them into the
                // message. The sibling Gemini and embedding paths already route
                // through this helper; this one did not.
                .map_err(|error| {
                    secret::transport_error("chat completion request", &url, &error)
                })?;

            if resp.status().is_success() {
                break resp;
            }

            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();

            // Only the output-budget rejection is retried, and only once. Any
            // other non-success status is the server's verdict and is returned
            // as-is: a 401, 404, or 429 must not be disguised as a parameter
            // problem. An explicit operator override is never second-guessed.
            let retryable = self.token_limit_field.is_none()
                && self.learned_token_limit_field.load(Ordering::Relaxed) == 0
                && indicates_unsupported_token_field(&text, token_limit_field);
            if !retryable {
                return Err(AgentError::Llm(format!("http {status}: {text}")));
            }

            token_limit_field = token_limit_field.flipped();
            self.learn_token_limit_field(token_limit_field);
        };

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

    /// Build a body through the same field selection the request path uses, so
    /// these tests keep covering selection and not only serialization.
    fn body_of(
        provider: OpenAiCompatProvider,
        messages: &[Message],
        tools: &[ToolSchema],
    ) -> Value {
        let field = provider.effective_token_limit_field();
        provider.build_body(messages, tools, field)
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

    /// Serve a scripted sequence of raw HTTP responses on loopback, returning
    /// the request bodies received in order.
    ///
    /// The existing single-response helper cannot express a retry, and the
    /// retry's whole point is what the *second* request carries.
    async fn spawn_scripted_http(
        responses: Vec<String>,
    ) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            let mut bodies = Vec::new();
            for response in responses {
                let accepted =
                    tokio::time::timeout(Duration::from_secs(5), listener.accept()).await;
                let (mut socket, _) = match accepted {
                    Ok(Ok(pair)) => pair,
                    // No further request arrived. That is the assertion for the
                    // cases that must NOT retry, so it ends the script quietly.
                    _ => break,
                };

                let mut raw = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let head_end = raw
                        .windows(4)
                        .position(|window| window == b"\r\n\r\n")
                        .map(|pos| pos + 4);
                    if let Some(head_end) = head_end {
                        let head = String::from_utf8_lossy(&raw[..head_end]).to_ascii_lowercase();
                        let declared = head
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .and_then(|value| value.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if raw.len() >= head_end + declared {
                            bodies.push(
                                String::from_utf8_lossy(&raw[head_end..head_end + declared])
                                    .to_string(),
                            );
                            break;
                        }
                    }
                    match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => raw.extend_from_slice(&chunk[..read]),
                    }
                }

                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
                let _ = socket.shutdown().await;
            }
            bodies
        });

        (format!("http://{addr}"), handle)
    }

    fn http_response(status: &str, content_type: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn unsupported_max_tokens_body() -> String {
        // The wording observed from the live endpoint, kept verbatim so the test
        // fails if the detector is narrowed past what servers actually send.
        concat!(
            "{\"error\":{\"message\":\"Unsupported parameter: 'max_tokens' is not supported ",
            "with this model. Use 'max_completion_tokens' instead.\",",
            "\"type\":\"invalid_request_error\",\"param\":\"max_tokens\",",
            "\"code\":\"unsupported_parameter\"}}"
        )
        .to_string()
    }

    fn done_sse() -> String {
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"
            .to_string()
    }

    #[test]
    fn dotted_point_releases_are_recognized_as_reasoning_models() {
        // The regression: `starts_with("gpt-5-")` missed every dotted release,
        // so gpt-5.5 and gpt-5.6-sol were sent max_tokens and rejected.
        for model in [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.5",
            "gpt-5.6",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "GPT-5.6-Luna",
            "o1",
            "o3-mini",
            "o4-mini",
        ] {
            assert_eq!(
                TokenLimitField::infer(model),
                TokenLimitField::MaxCompletionTokens,
                "{model} must use max_completion_tokens"
            );
        }

        // A longer number is a different family, not a variant of this one.
        for model in [
            "gpt-50",
            "gpt-5x",
            "gpt-4.1",
            "mistral-medium-3.5",
            "llama3.3",
            "deepseek-chat",
            "o10-mini",
        ] {
            assert_eq!(
                TokenLimitField::infer(model),
                TokenLimitField::MaxTokens,
                "{model} must use max_tokens"
            );
        }
    }

    #[test]
    fn only_the_output_budget_rejection_is_detected() {
        let sent = TokenLimitField::MaxTokens;
        assert!(indicates_unsupported_token_field(
            &unsupported_max_tokens_body(),
            sent
        ));
        // Names the other field only, so the field we sent was not the problem.
        assert!(!indicates_unsupported_token_field(
            "{\"error\":{\"message\":\"Unsupported parameter: 'max_completion_tokens'\"}}",
            sent
        ));
        // Unrelated 400s must not be retried.
        assert!(!indicates_unsupported_token_field(
            "{\"error\":{\"message\":\"Invalid schema for function 'read_file'\"}}",
            sent
        ));
        assert!(!indicates_unsupported_token_field(
            "{\"error\":{\"message\":\"max_tokens must be a positive integer\"}}",
            sent
        ));
    }

    #[tokio::test]
    async fn an_unsupported_budget_field_is_corrected_once_and_remembered() {
        // A name the list classifies as non-reasoning, so the first attempt is
        // deliberately wrong and the endpoint's verdict has to fix it.
        let (base_url, server) = spawn_scripted_http(vec![
            http_response(
                "400 Bad Request",
                "application/json",
                &unsupported_max_tokens_body(),
            ),
            http_response("200 OK", "text/event-stream", &done_sse()),
            http_response("200 OK", "text/event-stream", &done_sse()),
        ])
        .await;

        let provider = OpenAiCompatProvider::new("", "gpt-4.1", base_url);
        let cancel = CancellationToken::new();

        let events = provider.stream(&[], &[], &cancel).await.unwrap();
        assert!(matches!(
            collect(events).await.last().unwrap(),
            SseEvent::Stop {
                reason: StopReason::EndTurn
            }
        ));

        // The correction is remembered, so a later call starts with the right
        // field instead of paying for the rejection again.
        let events = provider.stream(&[], &[], &cancel).await.unwrap();
        let _ = collect(events).await;

        let bodies = server.await.unwrap();
        assert_eq!(
            bodies.len(),
            3,
            "expected reject, retry, then a second call"
        );
        assert!(bodies[0].contains("\"max_tokens\""), "first: {}", bodies[0]);
        assert!(
            bodies[1].contains("\"max_completion_tokens\""),
            "retry: {}",
            bodies[1]
        );
        assert!(
            bodies[2].contains("\"max_completion_tokens\""),
            "later call: {}",
            bodies[2]
        );
    }

    #[tokio::test]
    async fn other_failures_are_returned_verbatim_without_a_second_attempt() {
        // A 429 is the server's verdict on the request, not a parameter
        // problem. Retrying it with a different field would hide the real
        // cause and double the load.
        let (base_url, server) = spawn_scripted_http(vec![
            http_response(
                "429 Too Many Requests",
                "application/json",
                "{\"error\":\"slow down\"}",
            ),
            http_response("200 OK", "text/event-stream", &done_sse()),
        ])
        .await;

        let provider = OpenAiCompatProvider::new("", "gpt-4.1", base_url);
        let error = provider
            .stream(&[], &[], &CancellationToken::new())
            .await
            .expect_err("a 429 must surface");
        assert!(format!("{error}").contains("429"), "{error}");

        assert_eq!(server.await.unwrap().len(), 1, "must not retry a 429");
    }

    #[tokio::test]
    async fn an_explicit_operator_override_is_never_second_guessed() {
        // The operator has stated what their deployment accepts. Silently
        // sending the other field would override a human decision with a guess.
        let (base_url, server) = spawn_scripted_http(vec![
            http_response(
                "400 Bad Request",
                "application/json",
                &unsupported_max_tokens_body(),
            ),
            http_response("200 OK", "text/event-stream", &done_sse()),
        ])
        .await;

        let provider = OpenAiCompatProvider::new("", "gpt-5.6-sol", base_url)
            .with_token_limit_field(TokenLimitField::MaxTokens);
        let error = provider
            .stream(&[], &[], &CancellationToken::new())
            .await
            .expect_err("the rejection must surface instead of being worked around");
        assert!(
            format!("{error}").contains("unsupported_parameter"),
            "{error}"
        );

        let bodies = server.await.unwrap();
        assert_eq!(bodies.len(), 1, "must not retry against an explicit choice");
        assert!(bodies[0].contains("\"max_tokens\""));
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

        let body = body_of(
            OpenAiCompatProvider::new("unused", "fixture", "https://example.com/v1"),
            &[assistant],
            &[],
        );
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
        let body = body_of(p, &msgs, &tools);
        assert_eq!(body["model"], "gpt-5.5");
        // gpt-5.5 is a reasoning model, so the dotted name must select the
        // reasoning field here too.
        assert!(body.get("max_completion_tokens").is_some());
        assert!(body.get("max_tokens").is_none());
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
        let body = body_of(p, &msgs, &[]);
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

    /// Build through the same field selection the request path uses, so these
    /// tests keep covering the selection and not only the serialization.
    fn body_of(
        provider: OpenAiCompatProvider,
        messages: &[Message],
        tools: &[ToolSchema],
    ) -> Value {
        let field = provider.effective_token_limit_field();
        provider.build_body(messages, tools, field)
    }

    fn body_for(model: &str) -> Value {
        body_of(
            OpenAiCompatProvider::new("k", model, "https://example.invalid/openai/v1"),
            &user("hi"),
            &[],
        )
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
        let body = body_of(
            OpenAiCompatProvider::new("k", "my-private-deploy", "https://example.invalid/v1")
                .with_token_limit_field(TokenLimitField::MaxCompletionTokens),
            &user("hi"),
            &[],
        );
        assert!(body.get("max_completion_tokens").is_some());
        assert!(body.get("max_tokens").is_none());

        let body = body_of(
            OpenAiCompatProvider::new("k", "gpt-5-mini", "https://example.invalid/v1")
                .with_token_limit_field(TokenLimitField::MaxTokens),
            &user("hi"),
            &[],
        );
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
    fn the_default_budget_stays_below_the_observed_failure_point() {
        // 16384 was measured to fail against a live deployment when combined with
        // a large prompt. Pin the default so it cannot drift back up silently.
        // 8192 sits below 12288, the largest budget observed to succeed with a
        // large prompt, and well below the 16384 that failed.
        assert_eq!(DEFAULT_MAX_OUTPUT_TOKENS, 8192);
        let body = body_for("gpt-5-mini");
        assert_eq!(body["max_completion_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
    }

    #[test]
    fn the_budget_value_is_carried_through_either_field() {
        let body = body_of(
            OpenAiCompatProvider::new("k", "gpt-5-mini", "https://example.invalid/v1")
                .with_max_tokens(4242),
            &user("hi"),
            &[],
        );
        assert_eq!(body["max_completion_tokens"], 4242);

        let body = body_of(
            OpenAiCompatProvider::new("k", "gpt-4o", "https://example.invalid/v1")
                .with_max_tokens(4242),
            &user("hi"),
            &[],
        );
        assert_eq!(body["max_tokens"], 4242);
    }
}
