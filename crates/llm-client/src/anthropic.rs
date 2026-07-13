//! Anthropic-style (Messages API) streaming [`LlmProvider`].
//!
//! Supports all Claude models including:
//! - Claude Fable 5 (Mythos-class, latest — `claude-fable-5`)
//! - Claude Opus 4.8, 4.7
//! - Claude Sonnet 5, 4.6
//! - Claude Haiku 4.5
//!
//! The reader runs on a spawned task: it pipes `reqwest`'s `bytes_stream()`
//! into the [`SseParser`], maps Anthropic SSE frames to [`SseEvent`]s, and
//! sends them over an `mpsc::channel(64)`. Tool-use input arrives as partial
//! `input_json_delta` fragments — the raw string is accumulated and parsed
//! only at `content_block_stop` (parsing early yields wrong code). The task
//! exits when `cancel` fires or the receiver is dropped; a failed `send`
//! (receiver gone) breaks the loop rather than panicking.

use std::collections::HashMap;

use agent_types::{AgentError, ContentBlock, Message, Result, Role, ToolSchema};
use futures_util::StreamExt;
use serde_json::{json, Value};

/// Latest Anthropic model IDs (July 2026).
pub mod models {
    /// Mythos-class: most capable public model. Long-horizon autonomous coding.
    pub const FABLE_5: &str = "claude-fable-5";
    /// Opus class: deep reasoning.
    pub const OPUS_4_8: &str = "claude-opus-4-8";
    pub const OPUS_4_7: &str = "claude-opus-4-7";
    /// Sonnet class: balanced speed + intelligence.
    pub const SONNET_5: &str = "claude-sonnet-5";
    pub const SONNET_4_6: &str = "claude-sonnet-4-6";
    /// Haiku class: fast + cheap.
    pub const HAIKU_4_5: &str = "claude-haiku-4-5";
}
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::provider::{LlmProvider, SseEvent, StopReason};
use crate::sse::{RawSseFrame, SseParser};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    version: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    /// The API key is injected here — the library NEVER reads env vars itself
    /// (TASK-1.2). Callers resolve the key (e.g. the CLI in TASK-9.3).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            version: DEFAULT_VERSION.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Build the Anthropic Messages API request body. `System` messages fold
    /// into the top-level `system` field; `Tool` results ride as `user`
    /// messages carrying `tool_result` blocks (Anthropic's wire shape).
    fn build_body(&self, messages: &[Message], tools: &[ToolSchema]) -> Value {
        let mut system = String::new();
        let mut api_messages: Vec<Value> = Vec::new();

        for m in messages {
            match m.role {
                Role::System => {
                    for b in &m.content {
                        if let ContentBlock::Text(t) = b {
                            if !system.is_empty() {
                                system.push('\n');
                            }
                            system.push_str(t);
                        }
                    }
                }
                Role::User | Role::Tool => api_messages.push(json!({
                    "role": "user",
                    "content": blocks_to_json(&m.content),
                })),
                Role::Assistant => api_messages.push(json!({
                    "role": "assistant",
                    "content": blocks_to_json(&m.content),
                })),
            }
        }

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": true,
            "messages": api_messages,
        });
        if !system.is_empty() {
            body["system"] = json!(system);
        }
        if !tools.is_empty() {
            let ts: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = json!(ts);
        }
        body
    }
}

fn blocks_to_json(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .map(|b| match b {
            ContentBlock::Text(t) => json!({ "type": "text", "text": t }),
            ContentBlock::ToolUse { id, name, input } => json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }),
            ContentBlock::ToolResult {
                tool_use_id,
                output,
                is_error,
            } => json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": output,
                "is_error": is_error,
            }),
        })
        .collect()
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        cancel: &CancellationToken,
    ) -> Result<mpsc::Receiver<SseEvent>> {
        let body = self.build_body(messages, tools);

        let resp = self
            .client
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
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
            let mut parser = SseParser::new();
            let mut state = StreamState::new();

            loop {
                tokio::select! {
                    biased;
                    _ = child.cancelled() => break,
                    next = stream.next() => match next {
                        None => break,
                        Some(Err(e)) => {
                            let _ = tx.send(SseEvent::Error(e.to_string())).await;
                            break;
                        }
                        Some(Ok(bytes)) => {
                            match parser.feed(&bytes) {
                                Ok(frames) => {
                                    for frame in frames {
                                        for ev in state.map_frame(&frame) {
                                            if tx.send(ev).await.is_err() {
                                                return; // receiver dropped
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(SseEvent::Error(e.to_string())).await;
                                    break;
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

/// Accumulator for tool-use content blocks in flight, keyed by block index.
struct ToolAccum {
    id: String,
    name: String,
    json: String,
}

/// Streaming state machine mapping Anthropic frames to [`SseEvent`]s.
pub(crate) struct StreamState {
    tool_blocks: HashMap<u64, ToolAccum>,
    stop_reason: StopReason,
}

impl StreamState {
    pub(crate) fn new() -> Self {
        Self {
            tool_blocks: HashMap::new(),
            stop_reason: StopReason::EndTurn,
        }
    }

    /// Map one raw frame to zero or more high-level events, mutating internal
    /// accumulation state.
    pub(crate) fn map_frame(&mut self, frame: &RawSseFrame) -> Vec<SseEvent> {
        let v: Value = match serde_json::from_str(&frame.data) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let typ = v
            .get("type")
            .and_then(|t| t.as_str())
            .or(frame.event.as_deref())
            .unwrap_or("");

        match typ {
            "content_block_start" => {
                let index = v.get("index").and_then(Value::as_u64).unwrap_or(0);
                let cb = &v["content_block"];
                if cb.get("type").and_then(Value::as_str) == Some("tool_use") {
                    let id = cb.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                    let name = cb
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    self.tool_blocks.insert(
                        index,
                        ToolAccum {
                            id,
                            name,
                            json: String::new(),
                        },
                    );
                }
                Vec::new()
            }
            "content_block_delta" => {
                let index = v.get("index").and_then(Value::as_u64).unwrap_or(0);
                let delta = &v["delta"];
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            return vec![SseEvent::Delta(text.to_string())];
                        }
                        Vec::new()
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) =
                            delta.get("partial_json").and_then(Value::as_str)
                        {
                            if let Some(acc) = self.tool_blocks.get_mut(&index) {
                                acc.json.push_str(partial);
                            }
                        }
                        Vec::new()
                    }
                    _ => Vec::new(),
                }
            }
            "content_block_stop" => {
                let index = v.get("index").and_then(Value::as_u64).unwrap_or(0);
                if let Some(acc) = self.tool_blocks.remove(&index) {
                    let input: Value = if acc.json.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&acc.json).unwrap_or_else(|_| json!({}))
                    };
                    return vec![SseEvent::ToolUse {
                        id: acc.id,
                        name: acc.name,
                        input,
                    }];
                }
                Vec::new()
            }
            "message_delta" => {
                if let Some(reason) = v
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    self.stop_reason = map_stop_reason(reason);
                }
                Vec::new()
            }
            "message_stop" => vec![SseEvent::Stop {
                reason: self.stop_reason,
            }],
            "error" => {
                let msg = v
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or(&frame.data)
                    .to_string();
                vec![SseEvent::Error(msg)]
            }
            _ => Vec::new(),
        }
    }
}

fn map_stop_reason(s: &str) -> StopReason {
    match s {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(event: &str, data: &str) -> RawSseFrame {
        RawSseFrame {
            event: Some(event.to_string()),
            data: data.to_string(),
        }
    }

    #[test]
    fn text_delta_maps_to_delta_event() {
        let mut s = StreamState::new();
        let evs = s.map_frame(&frame(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        ));
        assert_eq!(evs.len(), 1);
        assert!(matches!(&evs[0], SseEvent::Delta(t) if t == "hi"));
    }

    #[test]
    fn tool_use_accumulates_partial_json_until_stop() {
        let mut s = StreamState::new();
        // start
        assert!(s
            .map_frame(&frame(
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"read_file"}}"#,
            ))
            .is_empty());
        // partial fragments — must NOT parse yet
        assert!(s
            .map_frame(&frame(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"sr"}}"#,
            ))
            .is_empty());
        assert!(s
            .map_frame(&frame(
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"c/main.rs\"}"}}"#,
            ))
            .is_empty());
        // stop -> emit the fully-parsed tool use
        let evs = s.map_frame(&frame(
            "content_block_stop",
            r#"{"type":"content_block_stop","index":1}"#,
        ));
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            SseEvent::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "src/main.rs");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn message_delta_then_stop_carries_reason() {
        let mut s = StreamState::new();
        assert!(s
            .map_frame(&frame(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            ))
            .is_empty());
        let evs = s.map_frame(&frame(
            "message_stop",
            r#"{"type":"message_stop"}"#,
        ));
        assert_eq!(evs.len(), 1);
        assert!(matches!(
            &evs[0],
            SseEvent::Stop {
                reason: StopReason::ToolUse
            }
        ));
    }

    #[test]
    fn body_folds_system_and_tool_results() {
        let p = AnthropicProvider::new("k", "claude-x");
        let msgs = vec![
            Message {
                role: Role::System,
                content: vec![ContentBlock::Text("be helpful".into())],
                token_estimate: 0,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    output: "done".into(),
                    is_error: false,
                }],
                token_estimate: 0,
            },
        ];
        let body = p.build_body(&msgs, &[]);
        assert_eq!(body["system"], "be helpful");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "tool_result");
        assert_eq!(body["messages"][0]["content"][0]["tool_use_id"], "tu_1");
    }
}
