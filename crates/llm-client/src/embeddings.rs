//! Gemini text embeddings via the `embedContent` REST API.
//!
//! Uses the same base URL and API key as the main Gemini provider.
//! Returns 768-dimensional vectors (text-embedding-004 model).

use agent_types::{AgentError, Result};
use serde_json::{json, Value};

use crate::secret;

const EMBEDDING_MODEL: &str = "text-embedding-004";

/// Fixed output width of `text-embedding-004`.
pub const GEMINI_EMBEDDING_DIMENSION: usize = 768;

pub struct GeminiEmbedder {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl GeminiEmbedder {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Embedding endpoint URL.
    ///
    /// The API key is intentionally absent: it travels in the sensitive
    /// `x-goog-api-key` header so it cannot leak through proxies, access logs,
    /// or error text.
    fn embed_url(&self) -> String {
        format!("{}/models/{}:embedContent", self.base_url, EMBEDDING_MODEL)
    }

    /// Generate an embedding for a single text string.
    /// Returns a 768-dimensional f32 vector.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let url = self.embed_url();
        let (key_header, key_value) = secret::api_key_header(&self.api_key)?;

        let body = json!({
            "model": format!("models/{}", EMBEDDING_MODEL),
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
        EMBEDDING_MODEL
    }

    /// Known without a request: `text-embedding-004` is fixed at 768.
    fn declared_dimension(&self) -> Option<usize> {
        Some(GEMINI_EMBEDDING_DIMENSION)
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
        let e = GeminiEmbedder::new("fake-key", "https://example.com/v1beta");
        assert!(!e.api_key.is_empty());
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
