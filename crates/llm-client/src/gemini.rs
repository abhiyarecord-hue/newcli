//! Google Gemini (generativelanguage.googleapis.com) streaming [`LlmProvider`].
//!
//! Uses the `streamGenerateContent` endpoint with SSE alt=sse mode.
//! Maps Gemini's chunked JSON responses to our unified [`SseEvent`] stream.

use agent_types::{AgentError, ContentBlock, Message, Result, Role, ToolSchema};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::provider::{LlmProvider, SseEvent, StopReason};

/// Default to Vertex AI Express Mode (uses startup/Cloud billing credits).
/// Set env GEMINI_USE_AI_STUDIO=1 to fall back to the old AI Studio endpoint.
const DEFAULT_BASE_URL: &str = "https://aiplatform.googleapis.com/v1/publishers/google";

pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build the Gemini generateContent request body.
    /// Maps our Message/ContentBlock model to Gemini's `contents` array.
    fn build_body(&self, messages: &[Message], tools: &[ToolSchema]) -> Value {
        let mut system_instruction: Option<Value> = None;
        let mut contents: Vec<Value> = Vec::new();

        for m in messages {
            match m.role {
                Role::System => {
                    // Gemini uses systemInstruction at top level
                    let text = m
                        .content
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text(t) = b {
                                Some(t.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    system_instruction = Some(json!({
                        "parts": [{"text": text}]
                    }));
                }
                Role::User => {
                    let parts = blocks_to_gemini_parts(&m.content);
                    contents.push(json!({
                        "role": "user",
                        "parts": parts
                    }));
                }
                Role::Assistant => {
                    let parts = blocks_to_gemini_parts(&m.content);
                    contents.push(json!({
                        "role": "model",
                        "parts": parts
                    }));
                }
                Role::Tool => {
                    // Tool results go as "function" role response in Gemini.
                    // The functionResponse name MUST match the original
                    // functionCall name — we recover it from the tool_use_id.
                    let parts: Vec<Value> = m
                        .content
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                output,
                                is_error: _,
                            } = b
                            {
                                Some(json!({
                                    "functionResponse": {
                                        "name": fn_name_from_id(tool_use_id),
                                        "response": {
                                            "content": output
                                        }
                                    }
                                }))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !parts.is_empty() {
                        contents.push(json!({
                            "role": "function",
                            "parts": parts
                        }));
                    }
                }
            }
        }

        let mut body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 65536,
                "thinkingConfig": {
                    "includeThoughts": true,
                    "thinkingLevel": "medium"
                }
            }
        });

        if let Some(si) = system_instruction {
            body["systemInstruction"] = si;
        }

        if !tools.is_empty() {
            let function_declarations: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = json!([{
                "functionDeclarations": function_declarations
            }]);
        }

        body
    }

    /// Streaming endpoint URL for Gemini.
    /// Vertex AI Express: .../publishers/google/models/{model}:streamGenerateContent?alt=sse&key=KEY
    /// AI Studio:         .../models/{model}:streamGenerateContent?alt=sse&key=KEY
    fn stream_url(&self) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, self.model, self.api_key
        )
    }
}

/// Reserved key used to carry Gemini's `thoughtSignature` through the
/// provider-agnostic [`ContentBlock::ToolUse`] `input` value.
const THOUGHT_SIGNATURE_KEY: &str = "_thought_signature";

/// Recover the function name from a tool_use_id of the form `gemini_<name>`.
/// Falls back to the id itself if the prefix is absent.
fn fn_name_from_id(tool_use_id: &str) -> &str {
    tool_use_id.strip_prefix("gemini_").unwrap_or(tool_use_id)
}

/// Split a stashed `thoughtSignature` out of a functionCall's args, returning
/// the cleaned args and the signature (if any).
fn split_thought_signature(input: &Value) -> (Value, Option<String>) {
    let mut cleaned = input.clone();
    let mut signature = None;
    if let Some(obj) = cleaned.as_object_mut() {
        if let Some(Value::String(sig)) = obj.remove(THOUGHT_SIGNATURE_KEY) {
            signature = Some(sig);
        }
    }
    (cleaned, signature)
}

fn blocks_to_gemini_parts(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .map(|b| match b {
            ContentBlock::Text(t) => json!({"text": t}),
            ContentBlock::ToolUse { id: _, name, input } => {
                let (args, signature) = split_thought_signature(input);
                let mut part = json!({
                    "functionCall": {
                        "name": name,
                        "args": args
                    }
                });
                if let Some(sig) = signature {
                    part["thoughtSignature"] = Value::String(sig);
                }
                part
            }
            ContentBlock::ToolResult {
                tool_use_id,
                output,
                is_error: _,
            } => json!({
                "functionResponse": {
                    "name": fn_name_from_id(tool_use_id),
                    "response": {"content": output}
                }
            }),
        })
        .collect()
}

#[async_trait::async_trait]
impl LlmProvider for GeminiProvider {
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        cancel: &CancellationToken,
    ) -> Result<mpsc::Receiver<SseEvent>> {
        let body = self.build_body(messages, tools);
        let url = self.stream_url();

        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
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
            let mut state = GeminiStreamState::default();

            loop {
                tokio::select! {
                    biased;
                    _ = child.cancelled() => break,
                    next = stream.next() => match next {
                        None => {
                            // Process any remaining data in buffer
                            process_buffer(&mut buffer, &tx, &mut state).await;
                            // Stream ended — emit stop if not already sent.
                            // If we saw a function call, the turn is a tool-use turn.
                            if !state.got_stop {
                                let reason = if state.saw_tool_use {
                                    StopReason::ToolUse
                                } else {
                                    StopReason::EndTurn
                                };
                                let _ = tx.send(SseEvent::Stop { reason }).await;
                            }
                            break;
                        }
                        Some(Err(e)) => {
                            let _ = tx.send(SseEvent::Error(e.to_string())).await;
                            break;
                        }
                        Some(Ok(bytes)) => {
                            let chunk = String::from_utf8_lossy(&bytes);
                            buffer.push_str(&chunk);
                            process_buffer(&mut buffer, &tx, &mut state).await;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// Cross-chunk streaming state for a single Gemini response.
#[derive(Default)]
struct GeminiStreamState {
    /// Whether any `functionCall` part was seen this turn. Used to decide the
    /// final [`StopReason`] (Gemini reports `finishReason: "STOP"` even for
    /// tool-call turns, so we must track this ourselves).
    saw_tool_use: bool,
    /// Whether a `Stop` event has already been emitted.
    got_stop: bool,
}

/// Process buffered SSE data, extracting complete events separated by "\n\n".
/// Also handles the case where data lines end with "\r\n\r\n".
async fn process_buffer(
    buffer: &mut String,
    tx: &mpsc::Sender<SseEvent>,
    state: &mut GeminiStreamState,
) {
    loop {
        let sep_pos = buffer
            .find("\r\n\r\n")
            .map(|p| (p, 4))
            .or_else(|| buffer.find("\n\n").map(|p| (p, 2)));

        let (pos, sep_len) = match sep_pos {
            Some(v) => v,
            None => break,
        };

        let line = buffer[..pos].to_string();
        *buffer = buffer[pos + sep_len..].to_string();

        // Strip "data: " prefix (SSE format)
        let data = if let Some(stripped) = line.strip_prefix("data: ") {
            stripped
        } else if let Some(stripped) = line.strip_prefix("data:") {
            stripped.trim_start()
        } else {
            continue;
        };

        let v: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let events = parse_gemini_chunk(&v, state);
        for ev in events {
            if matches!(&ev, SseEvent::Stop { .. }) {
                state.got_stop = true;
            }
            if tx.send(ev).await.is_err() {
                return; // receiver dropped
            }
        }
    }
}

/// Parse a single Gemini streaming chunk into our SseEvent(s), updating the
/// cross-chunk `state`.
fn parse_gemini_chunk(v: &Value, state: &mut GeminiStreamState) -> Vec<SseEvent> {
    let mut events = Vec::new();

    // Check for errors
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        events.push(SseEvent::Error(msg.to_string()));
        return events;
    }

    // Extract candidates[0].content.parts
    let parts = v
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    if let Some(parts) = parts {
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    // Check if this is a thought summary part
                    let is_thought = part.get("thought").and_then(Value::as_bool).unwrap_or(false);
                    if is_thought {
                        events.push(SseEvent::Thinking(text.to_string()));
                    } else {
                        events.push(SseEvent::Delta(text.to_string()));
                    }
                }
            }
            if let Some(fc) = part.get("functionCall") {
                state.saw_tool_use = true;
                let name = fc
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let mut args = fc.get("args").cloned().unwrap_or(json!({}));
                // Gemini 3.x thinking models return a `thoughtSignature` that
                // MUST be echoed back with the functionCall on the next turn.
                // Stash it inside the args under a reserved key so it survives
                // the round-trip through the (provider-agnostic) ContentBlock.
                if let Some(sig) = part.get("thoughtSignature").and_then(Value::as_str) {
                    if let Some(obj) = args.as_object_mut() {
                        obj.insert(
                            THOUGHT_SIGNATURE_KEY.to_string(),
                            Value::String(sig.to_string()),
                        );
                    }
                }
                events.push(SseEvent::ToolUse {
                    id: format!("gemini_{}", name),
                    name,
                    input: args,
                });
            }
        }
    }

    // Check finish reason
    let finish_reason = v
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finishReason"))
        .and_then(Value::as_str);

    if let Some(reason) = finish_reason {
        let stop_reason = match reason {
            "MAX_TOKENS" => StopReason::MaxTokens,
            // Gemini reports "STOP" for both plain end-of-turn and tool-call
            // turns, so consult the accumulated state.
            _ if state.saw_tool_use => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        };
        events.push(SseEvent::Stop {
            reason: stop_reason,
        });
    }

    // Extract usage metadata — only emit on the final chunk (when finishReason is present)
    // to avoid spamming the event bus with intermediate token counts.
    if finish_reason.is_some() {
        if let Some(usage) = v.get("usageMetadata") {
            let prompt = usage.get("promptTokenCount").and_then(Value::as_u64).unwrap_or(0) as u32;
            let completion = usage.get("candidatesTokenCount").and_then(Value::as_u64).unwrap_or(0) as u32;
            let total = usage.get("totalTokenCount").and_then(Value::as_u64).unwrap_or(0) as u32;
            if total > 0 {
                events.push(SseEvent::Usage {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                    total_tokens: total,
                });
            }
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_chunk() {
        let mut state = GeminiStreamState::default();
        let chunk = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello!"}],
                    "role": "model"
                }
            }]
        });
        let events = parse_gemini_chunk(&chunk, &mut state);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SseEvent::Delta(t) if t == "Hello!"));
    }

    #[test]
    fn parse_finish_reason() {
        let mut state = GeminiStreamState::default();
        let chunk = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "done"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        });
        let events = parse_gemini_chunk(&chunk, &mut state);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], SseEvent::Delta(t) if t == "done"));
        assert!(matches!(&events[1], SseEvent::Stop { reason: StopReason::EndTurn }));
    }

    #[test]
    fn function_call_yields_tool_use_stop_reason() {
        let mut state = GeminiStreamState::default();
        // Chunk 1: the function call.
        let call = json!({
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "read_file", "args": {"path": "src/main.rs"}}}],
                    "role": "model"
                }
            }]
        });
        let evs = parse_gemini_chunk(&call, &mut state);
        assert!(matches!(&evs[0], SseEvent::ToolUse { name, .. } if name == "read_file"));
        assert!(state.saw_tool_use);

        // Chunk 2: finishReason STOP — must map to ToolUse because a call was seen.
        let stop = json!({
            "candidates": [{ "content": {"parts": []}, "finishReason": "STOP" }]
        });
        let evs = parse_gemini_chunk(&stop, &mut state);
        assert!(matches!(
            evs.last().unwrap(),
            SseEvent::Stop { reason: StopReason::ToolUse }
        ));
    }

    #[test]
    fn parse_error_chunk() {
        let mut state = GeminiStreamState::default();
        let chunk = json!({
            "error": {
                "code": 400,
                "message": "Invalid API key"
            }
        });
        let events = parse_gemini_chunk(&chunk, &mut state);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SseEvent::Error(msg) if msg == "Invalid API key"));
    }

    #[test]
    fn body_builds_correctly() {
        let p = GeminiProvider::new("test-key", "gemini-2.0-flash");
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
        let body = p.build_body(&msgs, &[]);
        assert!(body.get("systemInstruction").is_some());
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
    }
}
