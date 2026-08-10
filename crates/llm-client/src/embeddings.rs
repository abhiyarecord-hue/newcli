//! Gemini text embeddings via the `embedContent` REST API.
//!
//! Uses the same base URL and API key as the main Gemini provider.
//!
//! The model is **not** hardcoded. This path previously pinned
//! `text-embedding-004`, which Google deprecated on both Google AI and Vertex
//! AI in January 2026, and pinned its 768-wide output as a compile-time fact.
//! Since Gemini is this tool's default provider, a user with only a Gemini key
//! was steered onto a retired model by default. The current text model is
//! `gemini-embedding-001`; `gemini-embedding-2` adds multimodal input. Both
//! return 3072 dimensions by default and accept a smaller `output_dimensionality`,
//! so the width is a property of the request, not of the code.

use agent_types::{AgentError, Result};
use serde_json::{json, Value};

use crate::secret;

/// Current text embedding model. Overridable, because the vendor's list moves
/// faster than this file does.
pub const DEFAULT_GEMINI_EMBEDDING_MODEL: &str = "gemini-embedding-001";

/// Output width of the retired `text-embedding-004`.
///
/// Retained only so a store written by an older build can still be interpreted.
/// It is not the width of any current model and must not be used to size or to
/// validate a new one: widths are measured from the endpoint.
#[deprecated(
    note = "text-embedding-004 was deprecated by Google in January 2026; measure the width from the endpoint instead"
)]
pub const GEMINI_EMBEDDING_DIMENSION: usize = 768;

pub struct GeminiEmbedder {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl GeminiEmbedder {
    /// Build an embedder using [`DEFAULT_GEMINI_EMBEDDING_MODEL`], unless
    /// `EMBEDDING_MODEL` names another one.
    ///
    /// Reading the same variable the OpenAI-compatible embedder reads keeps one
    /// knob for one decision, instead of a per-vendor variable a user has to
    /// discover.
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let model = std::env::var("EMBEDDING_MODEL")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_GEMINI_EMBEDDING_MODEL.to_string());
        Self::with_model(api_key, base_url, model)
    }

    /// Build an embedder for an explicit model, bypassing the environment.
    pub fn with_model(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: crate::http::client(),
            api_key: api_key.into(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    /// Embedding endpoint URL.
    ///
    /// The API key is intentionally absent: it travels in the sensitive
    /// `x-goog-api-key` header so it cannot leak through proxies, access logs,
    /// or error text.
    fn embed_url(&self) -> String {
        format!("{}/models/{}:embedContent", self.base_url, self.model)
    }

    /// Generate an embedding for a single text string.
    ///
    /// The width is whatever the configured model returns, so the caller must
    /// measure it rather than assume it.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let url = self.embed_url();
        let (key_header, key_value) = secret::api_key_header(&self.api_key)?;

        let body = json!({
            "model": format!("models/{}", self.model),
            "content": {
                "parts": [{"text": text}]
            }
        });

        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .header(key_header, key_value)
            .body(serde_json::to_vec(&body).map_err(|e| AgentError::Llm(e.to_string()))?)
            .send()
            .await
            .map_err(|error| secret::transport_error("embedding request", &url, &error))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = secret::read_bounded_body(resp).await;
            return Err(secret::status_error(
                "embedding request",
                &url,
                status,
                &text,
                &self.api_key,
            ));
        }

        // The success body is parsed unbounded because a valid embedding vector
        // must never be truncated; only error rendering is bounded.
        let response: Value = resp
            .json()
            .await
            .map_err(|error| secret::transport_error("embedding response", &url, &error))?;

        // Extract embedding.values array
        let values = response
            .get("embedding")
            .and_then(|e| e.get("values"))
            .and_then(Value::as_array)
            .ok_or_else(|| AgentError::Llm("no embedding.values in response".into()))?;

        let embedding: Vec<f32> = values
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if embedding.is_empty() {
            return Err(AgentError::Llm("empty embedding returned".into()));
        }

        Ok(embedding)
    }
}

#[async_trait::async_trait]
impl crate::embedder::Embedder for GeminiEmbedder {
    fn provider(&self) -> &str {
        "gemini"
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Not declared. The current models return 3072 by default and accept a
    /// smaller `output_dimensionality`, and the model itself is configurable, so
    /// any compile-time answer here would be a guess. Returning `None` makes the
    /// caller measure the width from a real response, which is what the storage
    /// layer already validates against.
    fn declared_dimension(&self) -> Option<usize> {
        None
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(text).await
    }

    /// `embedContent` takes one input per call, so this is a sequential loop
    /// rather than a native batch.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_one(text).await?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::Embedder;
    use crate::secret::testing::{request_head, spawn_http_capture};
    use std::time::Duration;

    #[test]
    fn embedder_constructs() {
        let e = GeminiEmbedder::with_model(
            "fake-key",
            "https://example.com/v1beta",
            DEFAULT_GEMINI_EMBEDDING_MODEL,
        );
        assert!(!e.api_key.is_empty());
    }

    #[test]
    fn the_retired_model_is_not_the_default_and_no_width_is_claimed() {
        // Google deprecated text-embedding-004 in January 2026, and Gemini is
        // this tool's default provider, so pinning it steered the default path
        // onto a retired model.
        let embedder = GeminiEmbedder::with_model("k", "https://example.com/v1beta", "");
        assert_ne!(DEFAULT_GEMINI_EMBEDDING_MODEL, "text-embedding-004");
        assert_eq!(DEFAULT_GEMINI_EMBEDDING_MODEL, "gemini-embedding-001");

        // The width belongs to the request, not to the code: the current models
        // return 3072 by default and accept a smaller output_dimensionality.
        assert_eq!(embedder.declared_dimension(), None);
    }

    #[tokio::test]
    async fn the_configured_model_reaches_both_the_url_and_the_body() {
        // Both places carried the model name, and only one was parameterised in
        // the first attempt at this change, which would have sent one model in
        // the path and another in the payload.
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"embedding":{"values":[0.5,-0.25]}}"#.to_vec(),
            Duration::from_secs(2),
        )
        .await;

        let embedder = GeminiEmbedder::with_model("k", base_url, "gemini-embedding-2");
        let embedding = embedder.embed("fixture").await.unwrap();
        assert_eq!(embedding, vec![0.5, -0.25]);

        let request = server.await.unwrap().expect("embedding request");
        let head = request_head(&request);
        assert!(
            head.contains("/models/gemini-embedding-2:embedContent"),
            "url: {head}"
        );
        assert!(
            request.contains("models/gemini-embedding-2"),
            "body: {request}"
        );
    }

    #[test]
    fn embedding_url_never_carries_the_credential() {
        let embedder = GeminiEmbedder::new("URL_SECRET", "https://example.com/v1beta");
        let url = embedder.embed_url();
        assert!(!url.contains("URL_SECRET"));
        assert!(!url.contains("key="));
    }

    #[tokio::test]
    async fn embedding_credential_is_sent_only_as_a_sensitive_header() {
        let secret = "EMBEDDING_HEADER_SECRET";
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"embedding":{"values":[0.5,-0.25]}}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = GeminiEmbedder::new(secret, base_url);
        let embedding = embedder.embed("fixture").await.unwrap();
        assert_eq!(embedding, vec![0.5, -0.25]);

        let head = request_head(&server.await.unwrap().expect("embedding request"));
        let request_line = head.lines().next().unwrap_or_default();
        assert!(!request_line.contains(secret));
        assert!(!request_line.contains("key="));
        assert!(head.to_lowercase().contains("x-goog-api-key:"));
        assert!(head.contains(secret));
    }

    #[tokio::test]
    async fn embedding_status_errors_are_sanitized_and_bounded() {
        let secret = "EMBEDDING_ERROR_SECRET";
        let mut body = format!("denied for {secret} ");
        body.push_str(&"detail ".repeat(2048));
        let (base_url, server) = spawn_http_capture(
            "403 Forbidden",
            "application/json",
            body.into_bytes(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = GeminiEmbedder::new(secret, base_url);
        let error = embedder.embed("fixture").await.unwrap_err().to_string();
        let _ = server.await.unwrap();

        assert!(!error.contains(secret));
        assert!(error.contains("[redacted]"));
        assert!(error.contains("403"));
        assert!(!error.contains("key="));
        assert!(error.contains("...[truncated]"));
    }
}
