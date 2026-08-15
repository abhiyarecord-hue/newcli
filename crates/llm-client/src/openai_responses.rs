//! OpenAI **Responses** API provider (`POST {base}/responses`).
//!
//! Separate from [`crate::openai_compat`] because the two APIs share nothing at
//! the wire level. Patching one to serve both would break both.
//!
//! This exists because the newest coding models cannot be reached any other way.
//! Verified against a live Azure AI Foundry endpoint:
//!
//! ```text
//! POST /chat/completions  model=gpt-5.3-codex
//!   -> 400 {"error":{"message":"The requested operation is unsupported."}}
//! POST /responses         model=gpt-5.3-codex
//!   -> 200, server-sent events
//! ```
//!
//! Chat Completions is **not** replaced. Ollama, LM Studio, vLLM and llama.cpp
//! speak only that shape, so both providers stay and the choice is made per
//! model.
//!
//! Wire differences, all confirmed from real captured events rather than from
//! documentation:
//!
//! | | Chat Completions | Responses |
//! |---|---|---|
//! | text | `choices[0].delta.content` | `response.output_text.delta` |
//! | end | `finish_reason` | `response.completed` |
//! | truncated | `finish_reason: "length"` | `response.incomplete` + `incomplete_details.reason` |
//! | tool call | `delta.tool_calls[].function` | `response.output_item.added` then `response.function_call_arguments.*` |
//! | tool schema | nested under `function` | flat `{type, name, parameters}` |
//! | tool result | a `role: "tool"` message | a `function_call_output` input item |
//! | system | a `role: "system"` message | the `instructions` field |
//! | budget | `max_completion_tokens` | `max_output_tokens` |

use std::collections::HashMap;

use agent_types::{AgentError, ContentBlock, Message, Result, Role, ToolSchema};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::provider::{LlmProvider, SseEvent, StopReason};
use crate::secret;

/// Default output budget, matching the Chat Completions provider so switching
/// APIs does not silently change how much a model is allowed to produce.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 8192;

/// Does this model name have to go through the Responses API?
///
/// Name-based, and deliberately narrow: it answers "is Chat Completions known to
/// be impossible", not "would Responses be nicer". The codex family returns
/// `400 The requested operation is unsupported` on `/chat/completions`, so
/// routing it there cannot work, and an automatic switch is a correction rather
/// than a preference. Anything else keeps its existing behaviour, and an
/// explicit provider choice always wins over this.
pub fn requires_responses_api(model: &str) -> bool {
    model.to_ascii_lowercase().contains("codex")
}

pub struct OpenAiResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    max_output_tokens: u32,
}

impl OpenAiResponsesProvider {
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: crate::http::client(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
        }
    }

    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }

    /// Translate the conversation into Responses `input` items.
    ///
    /// System messages become `instructions`, which is where this API expects
    /// them; there is no system role in the input list. Tool results become
    /// `function_call_output` items keyed by `call_id`, which is how a result is
    /// tied back to the call that asked for it.
    fn build_body(&self, messages: &[Message], tools: &[ToolSchema]) -> Value {
        let mut instructions: Vec<String> = Vec::new();
        let mut input: Vec<Value> = Vec::new();

        for message in messages {
            match message.role {
                Role::System => {
                    let text = extract_text(&message.content);
                    if !text.is_empty() {
                        instructions.push(text);
                    }
                }
                Role::User => {
                    let text = extract_text(&message.content);
                    if !text.is_empty() {
                        input.push(json!({
                            "role": "user",
                            "content": [{"type": "input_text", "text": text}],
                        }));
                    }
                    // A tool result may be carried on a user-role message.
                    push_tool_results(&message.content, &mut input);
                }
                Role::Assistant => {
                    let text = extract_text(&message.content);
                    if !text.is_empty() {
                        input.push(json!({
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }));
                    }
                    for block in &message.content {
                        if let ContentBlock::ToolUse {
                            id,
                            name,
                            input: arguments,
                            ..
                        } = block
                        {
                            input.push(json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                // Arguments travel as a JSON string, not an object.
                                "arguments": arguments.to_string(),
                            }));
                        }
                    }
                }
                Role::Tool => push_tool_results(&message.content, &mut input),
            }
        }

        let mut body = json!({
            "model": self.model,
            "stream": true,
            "input": input,
            "max_output_tokens": self.max_output_tokens,
        });

        if !instructions.is_empty() {
            body["instructions"] = json!(instructions.join("\n\n"));
        }
        if !tools.is_empty() {
            // Flat, unlike Chat Completions where the schema nests under
            // `function`. Sending the nested shape here is rejected.
            body["tools"] = Value::Array(
                tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        })
                    })
                    .collect(),
            );
        }
        body
    }
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn push_tool_results(blocks: &[ContentBlock], input: &mut Vec<Value>) {
    for block in blocks {
        if let ContentBlock::ToolResult {
            tool_use_id,
            output,
            ..
        } = block
        {
            input.push(json!({
                "type": "function_call_output",
                "call_id": tool_use_id,
                "output": output,
            }));
        }
    }
}

/// A function call being streamed, keyed by the output item that carries it.
struct PendingCall {
    call_id: String,
    name: String,
    arguments: String,
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiResponsesProvider {
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        cancel: &CancellationToken,
    ) -> Result<mpsc::Receiver<SseEvent>> {
        let url = format!("{}/responses", self.base_url);
        let body = self.build_body(messages, tools);

        let mut request = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header("HTTP-Referer", "https://newgen-cli.dev")
            .header("X-Title", "NewGen CLI");
        // An empty key is allowed: a local runtime needs none, and refusing to
        // run without one would exclude exactly those users.
        if !self.api_key.is_empty() {
            request = request.header("authorization", format!("Bearer {}", self.api_key));
        }

        let response = request
            .body(serde_json::to_vec(&body).map_err(|e| AgentError::Llm(e.to_string()))?)
            .send()
            .await
            .map_err(|error| secret::transport_error("responses request", &url, &error))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Llm(format!("http {status}: {text}")));
        }

        let (sender, receiver) = mpsc::channel(64);
        let cancel = cancel.clone();
        let stream_url = url.clone();

        tokio::spawn(async move {
            let mut bytes = response.bytes_stream();
            let mut buffer = String::new();
            let mut pending: HashMap<String, PendingCall> = HashMap::new();
            let mut used_a_tool = false;

            loop {
                let chunk = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        let _ = sender.send(SseEvent::Cancelled).await;
                        return;
                    }
                    chunk = bytes.next() => chunk,
                };

                let Some(chunk) = chunk else { break };
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        // Routed through the same helper as the pre-stream path
                        // so the endpoint is sanitised and, just as importantly,
                        // so `is_transient_transport_error` classifies it and the
                        // caller's retry can act on a dropped connection.
                        let message =
                            secret::transport_error("responses stream", &stream_url, &error)
                                .to_string();
                        let _ = sender.send(SseEvent::Error(message)).await;
                        return;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(at) = buffer.find('\n') {
                    let line = buffer[..at].trim_end_matches('\r').to_string();
                    buffer.drain(..=at);

                    // `event:` lines are ignored on purpose: every payload also
                    // carries its own `type`, so reading one source rather than
                    // two removes a way for them to disagree.
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim();
                    if payload.is_empty() || payload == "[DONE]" {
                        continue;
                    }
                    let Ok(event) = serde_json::from_str::<Value>(payload) else {
                        continue;
                    };

                    if !handle_event(&event, &sender, &mut pending, &mut used_a_tool).await {
                        return;
                    }
                }
            }

            // The stream ended without a terminal event, so any tool call that
            // was still accumulating is emitted rather than lost, and the turn is
            // closed honestly.
            let leftovers: Vec<PendingCall> = pending.into_values().collect();
            for call in leftovers {
                used_a_tool = true;
                if !emit_tool_use(&sender, call).await {
                    return;
                }
            }
            let reason = if used_a_tool {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            };
            let _ = sender.send(SseEvent::Stop { reason }).await;
        });

        Ok(receiver)
    }
}

/// Handle one decoded event. Returns `false` when the stream is finished.
async fn handle_event(
    event: &Value,
    sender: &mpsc::Sender<SseEvent>,
    pending: &mut HashMap<String, PendingCall>,
    used_a_tool: &mut bool,
) -> bool {
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    match kind {
        "response.output_text.delta" => {
            if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                if !delta.is_empty()
                    && sender
                        .send(SseEvent::Delta(delta.to_string()))
                        .await
                        .is_err()
                {
                    return false;
                }
            }
        }
        // Thought summaries, when the deployment is configured to stream them.
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                if !delta.is_empty()
                    && sender
                        .send(SseEvent::Thinking(delta.to_string()))
                        .await
                        .is_err()
                {
                    return false;
                }
            }
        }
        "response.output_item.added" => {
            let item = event.get("item");
            let is_call = item
                .and_then(|item| item.get("type"))
                .and_then(Value::as_str)
                == Some("function_call");
            if let (true, Some(item)) = (is_call, item) {
                if let Some(item_id) = item.get("id").and_then(Value::as_str) {
                    pending.insert(
                        item_id.to_string(),
                        PendingCall {
                            call_id: item
                                .get("call_id")
                                .and_then(Value::as_str)
                                .unwrap_or(item_id)
                                .to_string(),
                            name: item
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            arguments: item
                                .get("arguments")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        },
                    );
                }
            }
        }
        "response.function_call_arguments.delta" => {
            if let (Some(item_id), Some(delta)) = (
                event.get("item_id").and_then(Value::as_str),
                event.get("delta").and_then(Value::as_str),
            ) {
                if let Some(call) = pending.get_mut(item_id) {
                    call.arguments.push_str(delta);
                }
            }
        }
        "response.function_call_arguments.done" => {
            // Authoritative arguments, so the accumulated deltas are replaced
            // rather than appended to.
            if let Some(item_id) = event.get("item_id").and_then(Value::as_str) {
                if let Some(call) = pending.get_mut(item_id) {
                    if let Some(arguments) = event.get("arguments").and_then(Value::as_str) {
                        call.arguments = arguments.to_string();
                    }
                }
                if let Some(call) = pending.remove(item_id) {
                    *used_a_tool = true;
                    if !emit_tool_use(sender, call).await {
                        return false;
                    }
                }
            }
        }
        "response.output_item.done" => {
            // Only reached for a call whose arguments never produced a `done`
            // event; the normal path has already removed it.
            if let Some(item_id) = event
                .get("item")
                .and_then(|item| item.get("id"))
                .and_then(Value::as_str)
            {
                if let Some(call) = pending.remove(item_id) {
                    *used_a_tool = true;
                    if !emit_tool_use(sender, call).await {
                        return false;
                    }
                }
            }
        }
        "response.completed" => {
            send_usage(event, sender).await;
            let reason = if *used_a_tool {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            };
            let _ = sender.send(SseEvent::Stop { reason }).await;
            return false;
        }
        "response.incomplete" => {
            send_usage(event, sender).await;
            let reason = event
                .get("response")
                .and_then(|response| response.get("incomplete_details"))
                .and_then(|details| details.get("reason"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            // Only a budget stop is a MaxTokens stop. Anything else is a real
            // failure and must not be presented as truncated-but-fine output,
            // which the caller would try to continue.
            if reason == "max_output_tokens" {
                let _ = sender
                    .send(SseEvent::Stop {
                        reason: StopReason::MaxTokens,
                    })
                    .await;
            } else {
                let _ = sender
                    .send(SseEvent::Error(format!(
                        "response incomplete: {}",
                        if reason.is_empty() {
                            "reason not reported"
                        } else {
                            reason
                        }
                    )))
                    .await;
            }
            return false;
        }
        "response.failed" | "error" | "response.error" => {
            let message = event
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .or_else(|| event.get("message").and_then(Value::as_str))
                .or_else(|| {
                    event
                        .get("error")
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                })
                .unwrap_or("provider reported a failure without a message");
            let _ = sender.send(SseEvent::Error(message.to_string())).await;
            return false;
        }
        _ => {}
    }
    true
}

async fn emit_tool_use(sender: &mpsc::Sender<SseEvent>, call: PendingCall) -> bool {
    let arguments = call.arguments.trim();
    // An empty argument list is a legitimate call with no parameters.
    let parsed = if arguments.is_empty() {
        json!({})
    } else {
        match serde_json::from_str::<Value>(arguments) {
            Ok(value) => value,
            Err(error) => {
                // Guessing at malformed arguments would execute something the
                // model did not ask for.
                let _ = sender
                    .send(SseEvent::Error(format!(
                        "tool call `{}` had unparsable arguments: {error}",
                        call.name
                    )))
                    .await;
                return false;
            }
        }
    };

    sender
        .send(SseEvent::ToolUse {
            id: call.call_id,
            name: call.name,
            input: parsed,
            provider_metadata: None,
        })
        .await
        .is_ok()
}

async fn send_usage(event: &Value, sender: &mpsc::Sender<SseEvent>) {
    let Some(usage) = event
        .get("response")
        .and_then(|response| response.get("usage"))
    else {
        return;
    };
    let field = |name: &str| usage.get(name).and_then(Value::as_u64).unwrap_or(0) as u32;
    let prompt_tokens = field("input_tokens");
    let completion_tokens = field("output_tokens");
    let total_tokens = match usage.get("total_tokens").and_then(Value::as_u64) {
        Some(total) => total as u32,
        None => prompt_tokens + completion_tokens,
    };
    if prompt_tokens == 0 && completion_tokens == 0 && total_tokens == 0 {
        return;
    }
    let _ = sender
        .send(SseEvent::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        })
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::testing::spawn_http_capture;
    use std::time::Duration;

    fn provider(base_url: String) -> OpenAiResponsesProvider {
        OpenAiResponsesProvider::new("k", "gpt-5.3-codex", base_url)
    }

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text(text.into())],
            token_estimate: 0,
        }
    }

    async fn collect(mut events: mpsc::Receiver<SseEvent>) -> Vec<SseEvent> {
        let mut collected = Vec::new();
        while let Some(event) = events.recv().await {
            collected.push(event);
        }
        collected
    }

    async fn run(body: &str) -> Vec<SseEvent> {
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "text/event-stream",
            body.as_bytes().to_vec(),
            Duration::from_secs(5),
        )
        .await;
        let provider = provider(base_url);
        let events = provider
            .stream(&[user("hi")], &[], &CancellationToken::new())
            .await
            .unwrap();
        let collected = collect(events).await;
        let _ = server.await;
        collected
    }

    /// Text stream in the exact shape captured from a live endpoint.
    const TEXT_STREAM: &str = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n",
        "\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"content_index\":0,\"delta\":\"OK\",\"item_id\":\"msg_1\",\"output_index\":0,\"sequence_number\":4}\n",
        "\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":1100,\"output_tokens\":7,\"total_tokens\":1107}}}\n",
        "\n",
    );

    /// Function-call stream in the exact shape captured from a live endpoint.
    const TOOL_STREAM: &str = concat!(
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"in_progress\",\"arguments\":\"\",\"call_id\":\"call_abc\",\"name\":\"write_file\"},\"output_index\":0}\n",
        "\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"path\\\"\",\"item_id\":\"fc_1\",\"output_index\":0}\n",
        "\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"path\\\":\\\"z.txt\\\",\\\"content\\\":\\\"ZZZ\\\"}\",\"item_id\":\"fc_1\",\"output_index\":0}\n",
        "\n",
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"call_abc\",\"name\":\"write_file\"},\"output_index\":0}\n",
        "\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n",
        "\n",
    );

    #[test]
    fn only_a_model_that_cannot_use_chat_completions_is_rerouted() {
        // The codex family answers 400 on /chat/completions, so rerouting it is
        // a correction. Everything else must keep its existing path.
        for model in [
            "gpt-5.3-codex",
            "gpt-5.2-codex",
            "gpt-5.1-codex-max",
            "codex-mini",
            "GPT-5.3-CODEX",
        ] {
            assert!(requires_responses_api(model), "{model} must be rerouted");
        }
        for model in [
            "gpt-5.6-sol",
            "gpt-5-mini",
            "gpt-4o",
            "llama3.3",
            "deepseek-chat",
            "claude-fable-5",
        ] {
            assert!(
                !requires_responses_api(model),
                "{model} must not be rerouted"
            );
        }
    }

    #[tokio::test]
    async fn text_deltas_usage_and_completion_are_translated() {
        let events = run(TEXT_STREAM).await;
        assert!(
            matches!(&events[0], SseEvent::Delta(text) if text == "OK"),
            "{events:?}"
        );
        assert!(
            events.iter().any(|event| matches!(
                event,
                SseEvent::Usage {
                    prompt_tokens: 1100,
                    completion_tokens: 7,
                    total_tokens: 1107
                }
            )),
            "{events:?}"
        );
        assert!(
            matches!(
                events.last(),
                Some(SseEvent::Stop {
                    reason: StopReason::EndTurn
                })
            ),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn a_function_call_becomes_one_tool_use_with_the_call_id() {
        // `call_id` is what a result must be sent back under, so it, and not the
        // output item id, has to reach the caller.
        let events = run(TOOL_STREAM).await;
        let calls: Vec<&SseEvent> = events
            .iter()
            .filter(|event| matches!(event, SseEvent::ToolUse { .. }))
            .collect();
        assert_eq!(calls.len(), 1, "exactly one call expected: {events:?}");
        match calls[0] {
            SseEvent::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_abc");
                assert_eq!(name, "write_file");
                assert_eq!(input["path"], "z.txt");
                assert_eq!(input["content"], "ZZZ");
            }
            other => panic!("expected a tool call, got {other:?}"),
        }
        assert!(
            matches!(
                events.last(),
                Some(SseEvent::Stop {
                    reason: StopReason::ToolUse
                })
            ),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn a_budget_stop_is_reported_as_max_tokens_so_it_can_be_continued() {
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Rivers are\"}\n\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
        );
        let events = run(body).await;
        assert!(
            matches!(
                events.last(),
                Some(SseEvent::Stop {
                    reason: StopReason::MaxTokens
                })
            ),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn an_incomplete_response_for_any_other_reason_is_an_error() {
        // Reporting it as truncated-but-fine would make the caller try to
        // continue from output that was never valid.
        let body = "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"content_filter\"}}}\n\n";
        let events = run(body).await;
        match events.last() {
            Some(SseEvent::Error(message)) => {
                assert!(message.contains("content_filter"), "{message}")
            }
            other => panic!("expected an error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_failed_response_surfaces_the_provider_message() {
        let body = "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"model overloaded\"}}}\n\n";
        let events = run(body).await;
        match events.last() {
            Some(SseEvent::Error(message)) => {
                assert!(message.contains("model overloaded"), "{message}")
            }
            other => panic!("expected an error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unparsable_tool_arguments_are_refused_rather_than_guessed() {
        // Executing a tool from arguments that are not valid JSON would run
        // something the model did not ask for.
        let body = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"bash\",\"arguments\":\"\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{not json\",\"item_id\":\"fc_1\"}\n\n",
        );
        let events = run(body).await;
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, SseEvent::ToolUse { .. })),
            "no call may be emitted: {events:?}"
        );
        assert!(
            matches!(events.last(), Some(SseEvent::Error(message)) if message.contains("bash")),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn a_call_with_no_arguments_is_still_a_valid_call() {
        let body = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"list_files\",\"arguments\":\"\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"\",\"item_id\":\"fc_1\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let events = run(body).await;
        assert!(
            events.iter().any(|event| matches!(
                event,
                SseEvent::ToolUse { name, input, .. } if name == "list_files" && input.is_object()
            )),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn a_stream_cut_before_its_terminal_event_still_delivers_the_call() {
        // Losing a completed call because the connection ended early would make
        // the turn look like it did nothing.
        let body = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"read_file\",\"arguments\":\"\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"path\\\":\\\"a.txt\\\"}\",\"item_id\":\"fc_1\"}\n\n",
        );
        let events = run(body).await;
        assert!(
            events.iter().any(|event| matches!(
                event,
                SseEvent::ToolUse { id, name, input, .. }
                    if id == "call_9" && name == "read_file" && input["path"] == "a.txt"
            )),
            "{events:?}"
        );
        assert!(
            matches!(
                events.last(),
                Some(SseEvent::Stop {
                    reason: StopReason::ToolUse
                })
            ),
            "{events:?}"
        );
    }

    #[test]
    fn the_conversation_is_mapped_onto_the_responses_shape() {
        let messages = vec![
            Message {
                role: Role::System,
                content: vec![ContentBlock::Text("be terse".into())],
                token_estimate: 0,
            },
            user("write z.txt"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text("on it".into()),
                    ContentBlock::ToolUse {
                        id: "call_abc".into(),
                        name: "write_file".into(),
                        input: json!({"path": "z.txt"}),
                        provider_metadata: None,
                    },
                ],
                token_estimate: 0,
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_abc".into(),
                    output: "written".into(),
                    is_error: false,
                }],
                token_estimate: 0,
            },
        ];
        let tools = vec![ToolSchema {
            name: "write_file".into(),
            description: "Write a file".into(),
            input_schema: json!({"type": "object"}),
            effects: Default::default(),
        }];

        let body = provider("https://example.invalid/v1".into()).build_body(&messages, &tools);

        // A system message becomes `instructions`; there is no system role here.
        assert_eq!(body["instructions"], "be terse");
        assert!(body["input"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["role"] != "system"));

        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        // Arguments travel as a JSON string, not as an object.
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_abc");
        assert_eq!(input[2]["arguments"], "{\"path\":\"z.txt\"}");
        // A result is tied to its call by call_id.
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_abc");
        assert_eq!(input[3]["output"], "written");

        // Flat tool schema: nesting under `function`, as Chat Completions does,
        // is rejected by this API.
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "write_file");
        assert!(body["tools"][0]["parameters"].is_object());
        assert!(body["tools"][0]["function"].is_null());

        // Budget field name differs from Chat Completions.
        assert_eq!(body["max_output_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
        assert!(body["max_completion_tokens"].is_null());
        assert!(body["messages"].is_null());
        assert_eq!(body["stream"], true);
    }

    #[tokio::test]
    async fn a_non_success_status_is_returned_verbatim() {
        let (base_url, server) = spawn_http_capture(
            "400 Bad Request",
            "application/json",
            br#"{"error":{"message":"The requested operation is unsupported."}}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;
        let error = provider(base_url)
            .stream(&[user("hi")], &[], &CancellationToken::new())
            .await
            .expect_err("a 400 must surface");
        let message = format!("{error}");
        assert!(message.contains("400"), "{message}");
        assert!(message.contains("unsupported"), "{message}");
        let _ = server.await;
    }
}
