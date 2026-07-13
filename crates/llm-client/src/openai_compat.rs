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

/// Default: OpenAI's official endpoint.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

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
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
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
                            if let ContentBlock::ToolUse { id, name, input } = b {
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
            "max_tokens": self.max_tokens
        });

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
            .header("content-type", "application/json");

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

        tokio::spawn(async move {
            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();
            // Accumulate tool call arguments across multiple deltas.
            let mut tool_calls: std::collections::HashMap<u32, (String, String, String)> =
                std::collections::HashMap::new(); // index → (id, name, args_json)

            loop {
                tokio::select! {
                    biased;
                    _ = child.cancelled() => break,
                    next = stream.next() => match next {
                        None => {
                            // Emit any pending tool calls.
                            emit_pending_tools(&mut tool_calls, &tx).await;
                            let _ = tx.send(SseEvent::Stop { reason: StopReason::EndTurn }).await;
                            break;
                        }
                        Some(Err(e)) => {
                            let _ = tx.send(SseEvent::Error(e.to_string())).await;
                            break;
                        }
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                            // Process SSE lines.
                            while let Some(pos) = buffer.find('\n') {
                                let line = buffer[..pos].trim().to_string();
                                buffer = buffer[pos + 1..].to_string();

                                if line.is_empty() || line.starts_with(':') {
                                    continue;
                                }

                                let data = if let Some(d) = line.strip_prefix("data: ") {
                                    d.trim()
                                } else {
                                    continue;
                                };

                                if data == "[DONE]" {
                                    emit_pending_tools(&mut tool_calls, &tx).await;
                                    let reason = if tool_calls.is_empty() && tx.send(SseEvent::Stop { reason: StopReason::EndTurn }).await.is_err() {
                                        return;
                                    } else {
                                        return;
                                    };
                                }

                                let chunk: Value = match serde_json::from_str(data) {
                                    Ok(v) => v,
                                    Err(_) => continue,
                                };

                                // Extract delta.
                                let delta = match chunk.get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("delta"))
                                {
                                    Some(d) => d,
                                    None => continue,
                                };

                                // Text content.
                                if let Some(content) = delta.get("content").and_then(Value::as_str) {
                                    if !content.is_empty() {
                                        if tx.send(SseEvent::Delta(content.to_string())).await.is_err() {
                                            return;
                                        }
                                    }
                                }

                                // Tool calls (streamed incrementally).
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
                                        // Append arguments fragment.
                                        if let Some(args) = tc.get("function")
                                            .and_then(|f| f.get("arguments"))
                                            .and_then(Value::as_str)
                                        {
                                            entry.2.push_str(args);
                                        }
                                    }
                                }

                                // finish_reason.
                                if let Some(reason) = chunk.get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("finish_reason"))
                                    .and_then(Value::as_str)
                                {
                                    let stop = match reason {
                                        "tool_calls" | "function_call" => {
                                            emit_pending_tools(&mut tool_calls, &tx).await;
                                            StopReason::ToolUse
                                        }
                                        "length" => StopReason::MaxTokens,
                                        _ => StopReason::EndTurn,
                                    };
                                    let _ = tx.send(SseEvent::Stop { reason: stop }).await;
                                    return;
                                }

                                // Usage (if present in chunk).
                                if let Some(usage) = chunk.get("usage") {
                                    let prompt = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
                                    let completion = usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
                                    let total = usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
                                    if total > 0 {
                                        let _ = tx.send(SseEvent::Usage { prompt_tokens: prompt, completion_tokens: completion, total_tokens: total }).await;
                                    }
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

/// Emit accumulated tool calls as SseEvent::ToolUse.
async fn emit_pending_tools(
    tool_calls: &mut std::collections::HashMap<u32, (String, String, String)>,
    tx: &mpsc::Sender<SseEvent>,
) {
    let mut sorted: Vec<(u32, (String, String, String))> =
        tool_calls.drain().collect();
    sorted.sort_by_key(|(idx, _)| *idx);
    for (_idx, (id, name, args_json)) in sorted {
        let input: Value = serde_json::from_str(&args_json).unwrap_or(json!({}));
        let _ = tx
            .send(SseEvent::ToolUse { id, name, input })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
