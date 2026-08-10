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
use crate::secret;
use crate::sse::{RawSseFrame, SseParser};

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
            client: crate::http::client(),
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
                    if !parts.is_empty() {
                        contents.push(json!({
                            "role": "user",
                            "parts": parts
                        }));
                    }
                }
                Role::Assistant => {
                    let parts = blocks_to_gemini_parts(&m.content);
                    if !parts.is_empty() {
                        contents.push(json!({
                            "role": "model",
                            "parts": parts
                        }));
                    }
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
            "contents": if contents.is_empty() {
                vec![json!({"role": "user", "parts": [{"text": " "}]})]
            } else {
                contents
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 65536,
                "thinkingConfig": {
                    "includeThoughts": true,
                    "thinkingLevel": "MEDIUM"
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
    ///
    /// The API key is intentionally absent: it travels in the sensitive
    /// `x-goog-api-key` header so it cannot leak through proxies, access logs,
    /// or error text.
    /// Vertex AI Express: .../publishers/google/models/{model}:streamGenerateContent?alt=sse
    /// AI Studio:         .../models/{model}:streamGenerateContent?alt=sse
    fn stream_url(&self) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, self.model
        )
    }
}

/// Provider metadata key used for Gemini's protocol-only thought signature.
const THOUGHT_SIGNATURE_KEY: &str = "thought_signature";

/// Recover the function name from a tool_use_id of the form `gemini_<name>`.
/// Falls back to the id itself if the prefix is absent.
fn fn_name_from_id(tool_use_id: &str) -> &str {
    tool_use_id.strip_prefix("gemini_").unwrap_or(tool_use_id)
}

fn thought_signature(provider_metadata: &Option<Value>) -> Option<&str> {
    provider_metadata
        .as_ref()
        .and_then(|metadata| metadata.get(THOUGHT_SIGNATURE_KEY))
        .and_then(Value::as_str)
}

fn blocks_to_gemini_parts(blocks: &[ContentBlock]) -> Vec<Value> {
    let parts: Vec<Value> = blocks
        .iter()
        .map(|b| match b {
            ContentBlock::Text(t) => json!({"text": if t.is_empty() { " " } else { t.as_str() }}),
            ContentBlock::ToolUse {
                id: _,
                name,
                input,
                provider_metadata,
            } => {
                let mut part = json!({
                    "functionCall": {
                        "name": name,
                        "args": input
                    }
                });
                if let Some(signature) = thought_signature(provider_metadata) {
                    part["thoughtSignature"] = Value::String(signature.to_string());
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
                    "response": {"content": if output.is_empty() { "(empty)" } else { output.as_str() }}
                }
            }),
        })
        .collect();
    // Gemini requires at least one part — if empty, add a placeholder.
    if parts.is_empty() {
        vec![json!({"text": " "})]
    } else {
        parts
    }
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
        let (key_header, key_value) = secret::api_key_header(&self.api_key)?;

        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header(key_header, key_value)
            .body(serde_json::to_vec(&body).map_err(|e| AgentError::Llm(e.to_string()))?)
            .send()
            .await
            .map_err(|error| secret::transport_error("generation request", &url, &error))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = secret::read_bounded_body(resp).await;
            return Err(secret::status_error(
                "generation request",
                &url,
                status,
                &text,
                &self.api_key,
            ));
        }

        let (tx, rx) = mpsc::channel(64);
        let endpoint = secret::sanitized_endpoint(&url);
        let redaction_key = self.api_key.clone();
        let child = cancel.child_token();

        tokio::spawn(async move {
            let mut stream = resp.bytes_stream();
            let mut parser = SseParser::new();
            let mut state = GeminiStreamState {
                redaction_key,
                ..GeminiStreamState::default()
            };

            loop {
                tokio::select! {
                    biased;
                    _ = child.cancelled() => {
                        let _ = tx.send(SseEvent::Cancelled).await;
                        return;
                    }
                    next = stream.next() => match next {
                        None => {
                            let message = match parser.finish() {
                                Ok(()) => "sse stream ended without a terminal event".to_string(),
                                Err(error) => error.to_string(),
                            };
                            let _ = tx.send(SseEvent::Error(message)).await;
                            return;
                        }
                        Some(Err(error)) => {
                            // Render only the sanitized endpoint and a failure
                            // category; reqwest's own Display includes the URL.
                            let _ = tx.send(SseEvent::Error(format!(
                                "generation stream from {endpoint}: {}",
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
                                if process_frame(&frame, &tx, &mut state).await {
                                    return;
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

/// Cross-chunk streaming state for a single Gemini response.
#[derive(Default)]
struct GeminiStreamState {
    /// Whether any `functionCall` part was seen this turn. Used to decide the
    /// final [`StopReason`] (Gemini reports `finishReason: "STOP"` even for
    /// tool-call turns, so we must track this ourselves).
    saw_tool_use: bool,
    /// Credential redacted from provider-supplied error text. Empty in unit
    /// tests that do not exercise credential handling.
    redaction_key: String,
}

/// Decode one complete byte-framed SSE event. Returns true after emitting a
/// terminal event or when the receiver has gone away.
async fn process_frame(
    frame: &RawSseFrame,
    tx: &mpsc::Sender<SseEvent>,
    state: &mut GeminiStreamState,
) -> bool {
    // Keep-alive and comment-only frames carry no payload and must not end the turn.
    if frame.data.trim().is_empty() {
        return false;
    }

    let value: Value = match serde_json::from_str(&frame.data) {
        Ok(value) => value,
        Err(error) => {
            let _ = tx
                .send(SseEvent::Error(format!(
                    "invalid JSON in complete SSE frame: {error}"
                )))
                .await;
            return true;
        }
    };

    for event in parse_gemini_chunk(&value, state) {
        let terminal = matches!(
            event,
            SseEvent::Stop { .. } | SseEvent::Error(_) | SseEvent::Cancelled
        );
        if tx.send(event).await.is_err() || terminal {
            return true;
        }
    }
    false
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
        // Provider-supplied text is untrusted and may echo the credential, so
        // it is redacted and bounded like any other rendered error body.
        events.push(SseEvent::Error(secret::bounded_redacted_body(
            msg,
            &state.redaction_key,
        )));
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
                    let is_thought = part
                        .get("thought")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
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
                let args = fc.get("args").cloned().unwrap_or(json!({}));
                // Keep protocol-only metadata outside executable arguments so
                // legitimate user fields are preserved exactly.
                let provider_metadata = part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(|signature| json!({ THOUGHT_SIGNATURE_KEY: signature }));
                events.push(SseEvent::ToolUse {
                    id: format!("gemini_{}", name),
                    name,
                    input: args,
                    provider_metadata,
                });
            }
        }
    }

    let finish_reason = v
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finishReason"))
        .and_then(Value::as_str);

    // Emit usage before the terminal stop event so consumers can account for
    // the complete request without racing the stream shutdown.
    if finish_reason.is_some() {
        if let Some(usage) = v.get("usageMetadata") {
            let prompt = usage
                .get("promptTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let completion = usage
                .get("candidatesTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let total = usage
                .get("totalTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            if total > 0 {
                events.push(SseEvent::Usage {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                    total_tokens: total,
                });
            }
        }
    }

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

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::testing::{request_head, spawn_http_capture};
    use std::time::Duration;

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
        assert!(matches!(
            &events[1],
            SseEvent::Stop {
                reason: StopReason::EndTurn
            }
        ));
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
            SseEvent::Stop {
                reason: StopReason::ToolUse
            }
        ));
    }

    #[test]
    fn thought_signature_stays_out_of_band_and_user_input_is_unchanged() {
        // **Validates: Requirements 2.38, 3.10**
        let mut state = GeminiStreamState::default();
        let call = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "strict_tool",
                            "args": {"thought_signature": "user-owned", "value": 1}
                        },
                        "thoughtSignature": "provider-owned"
                    }],
                    "role": "model"
                }
            }]
        });

        let events = parse_gemini_chunk(&call, &mut state);
        let (input, metadata) = match &events[0] {
            SseEvent::ToolUse {
                input,
                provider_metadata,
                ..
            } => (input, provider_metadata),
            _ => panic!("expected tool use"),
        };
        assert_eq!(input["thought_signature"], "user-owned");
        assert_eq!(input["value"], 1);
        assert_eq!(
            metadata.as_ref().unwrap()[THOUGHT_SIGNATURE_KEY],
            "provider-owned"
        );

        let parts = blocks_to_gemini_parts(&[ContentBlock::ToolUse {
            id: "gemini_strict_tool".into(),
            name: "strict_tool".into(),
            input: input.clone(),
            provider_metadata: metadata.clone(),
        }]);
        assert_eq!(
            parts[0]["functionCall"]["args"]["thought_signature"],
            "user-owned"
        );
        assert_eq!(parts[0]["thoughtSignature"], "provider-owned");
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
    fn in_stream_provider_errors_are_redacted_and_bounded() {
        // **Validates: Requirements 2.25**
        let mut state = GeminiStreamState {
            redaction_key: "STREAM_SECRET".into(),
            ..GeminiStreamState::default()
        };
        let long_detail = "detail ".repeat(2048);
        let chunk = json!({
            "error": {
                "code": 401,
                "message": format!("rejected STREAM_SECRET {long_detail}")
            }
        });

        let events = parse_gemini_chunk(&chunk, &mut state);
        let message = match &events[0] {
            SseEvent::Error(message) => message,
            other => panic!("expected error event, got {other:?}"),
        };
        assert!(!message.contains("STREAM_SECRET"));
        assert!(message.contains("[redacted]"));
        assert!(message.ends_with("...[truncated]"));
    }

    async fn gemini_stream_fixture(body: Vec<u8>, cancel: &CancellationToken) -> Vec<SseEvent> {
        let (base_url, server) =
            spawn_http_capture("200 OK", "text/event-stream", body, Duration::from_secs(5)).await;
        let provider = GeminiProvider::new("fixture", "fixture").with_base_url(base_url);
        let mut events = provider.stream(&[], &[], cancel).await.unwrap();
        let mut collected = Vec::new();
        while let Some(event) = events.recv().await {
            collected.push(event);
        }
        let _ = server.await;
        collected
    }

    #[tokio::test]
    async fn cancellation_emits_exactly_one_terminal_cancelled_and_never_stop() {
        // **Validates: Requirements 2.36**
        let cancel = CancellationToken::new();
        cancel.cancel();
        let body =
            b"data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n"
                .to_vec();

        let events = gemini_stream_fixture(body, &cancel).await;
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
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"kept\"}]},",
            "\"finishReason\":\"STOP\"}]}\n\n"
        );

        let events =
            gemini_stream_fixture(body.as_bytes().to_vec(), &CancellationToken::new()).await;
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
    async fn truncated_stream_is_an_error_not_a_completion() {
        // **Validates: Requirements 2.36, 2.37**
        // No terminating blank line, so the response was cut mid-frame.
        let body =
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}".to_vec();

        let events = gemini_stream_fixture(body, &CancellationToken::new()).await;
        assert!(!events
            .iter()
            .any(|event| matches!(event, SseEvent::Stop { .. })));
        assert!(matches!(
            events.last().unwrap(),
            SseEvent::Error(message) if message.contains("incomplete")
        ));
    }

    #[test]
    fn persisted_metadata_round_trips_through_history_and_is_re_emitted() {
        // **Validates: Requirements 2.38, 3.10**
        let assistant = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("calling".into()),
                ContentBlock::ToolUse {
                    id: "gemini_strict_tool".into(),
                    name: "strict_tool".into(),
                    input: json!({"_thought_signature": "user-value", "path": "src/lib.rs"}),
                    provider_metadata: Some(json!({ THOUGHT_SIGNATURE_KEY: "provider-value" })),
                },
            ],
            token_estimate: 0,
        };

        // Simulate durable history persistence and reload.
        let persisted = serde_json::to_string(&assistant).unwrap();
        let restored: Message = serde_json::from_str(&persisted).unwrap();

        let body = GeminiProvider::new("unused", "fixture").build_body(&[restored], &[]);
        let parts = &body["contents"][0]["parts"];
        let call = parts
            .as_array()
            .unwrap()
            .iter()
            .find(|part| part.get("functionCall").is_some())
            .expect("function call part");

        // Executable arguments stay byte-identical, including a legitimate
        // user field spelled like the provider's own metadata.
        assert_eq!(
            call["functionCall"]["args"]["_thought_signature"],
            "user-value"
        );
        assert_eq!(call["functionCall"]["args"]["path"], "src/lib.rs");
        assert!(call["functionCall"]["args"]
            .get("thoughtSignature")
            .is_none());
        // The signature rides beside the call, never inside its arguments.
        assert_eq!(call["thoughtSignature"], "provider-value");
    }

    #[test]
    fn stream_url_never_carries_the_credential() {
        let provider = GeminiProvider::new("URL_SECRET", "fixture-model");
        let url = provider.stream_url();
        assert!(!url.contains("URL_SECRET"));
        assert!(!url.contains("key="));
        assert!(url.ends_with("/models/fixture-model:streamGenerateContent?alt=sse"));
    }

    #[tokio::test]
    async fn generation_credential_is_sent_only_as_a_sensitive_header() {
        let secret = "GENERATION_HEADER_SECRET";
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "text/event-stream",
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}]}\n\n"
                .to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let provider = GeminiProvider::new(secret, "fixture").with_base_url(base_url);
        let mut events = provider
            .stream(&[], &[], &CancellationToken::new())
            .await
            .unwrap();
        let mut saw_text = false;
        while let Some(event) = events.recv().await {
            if matches!(&event, SseEvent::Delta(text) if text == "hi") {
                saw_text = true;
            }
        }
        assert!(saw_text);

        let head = request_head(&server.await.unwrap().expect("generation request"));
        let request_line = head.lines().next().unwrap_or_default();
        assert!(!request_line.contains(secret));
        assert!(!request_line.contains("key="));
        assert!(head.to_lowercase().contains("x-goog-api-key:"));
        assert!(head.contains(secret));
    }

    #[tokio::test]
    async fn generation_status_errors_are_sanitized_and_bounded() {
        let secret = "GENERATION_ERROR_SECRET";
        let mut body = format!("rejected key {secret} ");
        body.push_str(&"detail ".repeat(2048));
        let (base_url, server) = spawn_http_capture(
            "401 Unauthorized",
            "application/json",
            body.into_bytes(),
            Duration::from_secs(5),
        )
        .await;

        let provider = GeminiProvider::new(secret, "fixture").with_base_url(base_url);
        let error = provider
            .stream(&[], &[], &CancellationToken::new())
            .await
            .unwrap_err()
            .to_string();
        let _ = server.await.unwrap();

        assert!(!error.contains(secret));
        assert!(error.contains("[redacted]"));
        assert!(error.contains("401"));
        assert!(!error.contains("key="));
        assert!(error.contains("...[truncated]"));
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
