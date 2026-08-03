//! Embeddings over the OpenAI-compatible `POST {base}/embeddings` shape.
//!
//! One implementation covers most of the field, because this request shape is
//! what OpenAI, Azure AI Foundry, Mistral, DeepSeek, Together, OpenRouter,
//! Ollama, LM Studio, and vLLM all accept. That matters for an open-source tool:
//! a user should be able to point it at whatever they already run, hosted or
//! local, without waiting for a vendor-specific client to be written.
//!
//! Two deliberate choices:
//!
//! - **The credential travels only in the `Authorization` header**, never in the
//!   URL, so it cannot leak through proxy logs or error text.
//! - **An empty key is allowed.** Local runtimes such as Ollama and vLLM require
//!   no credential, and refusing to run without one would exclude exactly the
//!   users who most want a self-hosted option.

use agent_types::{AgentError, Result};
use serde_json::{json, Value};

use crate::embedder::Embedder;
use crate::secret;

pub struct OpenAiCompatEmbedder {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    provider: String,
    /// Set when the operator knows the width, or when the endpoint honours an
    /// explicit `dimensions` request.
    declared_dimension: Option<usize>,
    /// Sent as `dimensions` when the model supports shortening, which OpenAI's
    /// v3 embedding models do.
    requested_dimensions: Option<usize>,
}

impl OpenAiCompatEmbedder {
    /// `base_url` is the API root, for example `https://api.openai.com/v1` or
    /// `http://localhost:11434/v1`. `/embeddings` is appended.
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
            model: model.into(),
            provider: "openai-compatible".to_string(),
            declared_dimension: None,
            requested_dimensions: None,
        }
    }

    /// Record a provider label other than the generic one.
    ///
    /// Worth setting, because this label is stored with every vector and is what
    /// stops a later search from comparing an Ollama vector against an OpenAI
    /// vector of the same width.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// State the width instead of measuring it. Skips the startup probe.
    pub fn with_declared_dimension(mut self, dimension: usize) -> Self {
        self.declared_dimension = Some(dimension);
        self
    }

    /// Ask the endpoint for a specific width via `dimensions`.
    ///
    /// Supported by OpenAI's v3 embedding models. The value is also treated as
    /// declared, since a server that honours the request returns exactly it, and
    /// a server that ignores the field is caught by the length check below.
    pub fn with_requested_dimensions(mut self, dimension: usize) -> Self {
        self.requested_dimensions = Some(dimension);
        self.declared_dimension = Some(dimension);
        self
    }

    fn embed_url(&self) -> String {
        format!("{}/embeddings", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait::async_trait]
impl Embedder for OpenAiCompatEmbedder {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn declared_dimension(&self) -> Option<usize> {
        self.declared_dimension
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = self.embed_url();
        let mut body = json!({ "model": self.model, "input": text });
        if let Some(dimensions) = self.requested_dimensions {
            body["dimensions"] = json!(dimensions);
        }

        let mut request = self
            .client
            .post(&url)
            .header("content-type", "application/json");
        // Local runtimes need no credential; only send the header when there is
        // one to send.
        if !self.api_key.is_empty() {
            request = request.header("authorization", format!("Bearer {}", self.api_key));
        }

        let resp = request
            .body(serde_json::to_vec(&body).map_err(|error| AgentError::Llm(error.to_string()))?)
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

        // Parsed unbounded on success: truncating a vector would corrupt it
        // silently. Only error bodies are bounded.
        let response: Value = resp
            .json()
            .await
            .map_err(|error| secret::transport_error("embedding response", &url, &error))?;

        let values = response
            .get("data")
            .and_then(Value::as_array)
            .and_then(|data| data.first())
            .and_then(|entry| entry.get("embedding"))
            .and_then(Value::as_array)
            .ok_or_else(|| AgentError::Llm("no data[0].embedding in response".into()))?;

        let embedding: Vec<f32> = values
            .iter()
            .filter_map(|value| value.as_f64().map(|float| float as f32))
            .collect();

        if embedding.is_empty() {
            return Err(AgentError::Llm("empty embedding returned".into()));
        }

        // A server that silently ignores `dimensions` would otherwise poison an
        // index built for the requested width. Fail instead.
        if let Some(expected) = self.declared_dimension {
            if embedding.len() != expected {
                return Err(AgentError::Llm(format!(
                    "embedding width {} does not match the configured {expected}; \
                     the endpoint may not support the requested dimensions",
                    embedding.len()
                )));
            }
        }

        Ok(embedding)
    }

    /// Native batching: the OpenAI shape accepts an array `input` and returns one
    /// entry per element, so a whole chunk batch costs a single request.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = self.embed_url();
        let mut body = json!({ "model": self.model, "input": texts });
        if let Some(dimensions) = self.requested_dimensions {
            body["dimensions"] = json!(dimensions);
        }

        let mut request = self
            .client
            .post(&url)
            .header("content-type", "application/json");
        if !self.api_key.is_empty() {
            request = request.header("authorization", format!("Bearer {}", self.api_key));
        }

        let resp = request
            .body(serde_json::to_vec(&body).map_err(|error| AgentError::Llm(error.to_string()))?)
            .send()
            .await
            .map_err(|error| secret::transport_error("embedding batch request", &url, &error))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = secret::read_bounded_body(resp).await;
            return Err(secret::status_error(
                "embedding batch request",
                &url,
                status,
                &text,
                &self.api_key,
            ));
        }

        let response: Value = resp
            .json()
            .await
            .map_err(|error| secret::transport_error("embedding batch response", &url, &error))?;

        let data = response
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| AgentError::Llm("no data array in embedding response".into()))?;

        if data.len() != texts.len() {
            return Err(AgentError::Llm(format!(
                "embedding batch returned {} vectors for {} inputs",
                data.len(),
                texts.len()
            )));
        }

        // `index` is honoured rather than assuming response order matches input
        // order, because the API documents an index per entry and relying on
        // ordering would misalign every vector if a server reordered them.
        let mut out = vec![Vec::new(); texts.len()];
        for entry in data {
            let position = entry
                .get("index")
                .and_then(Value::as_u64)
                .map(|index| index as usize)
                .ok_or_else(|| AgentError::Llm("embedding entry has no index".into()))?;
            if position >= out.len() {
                return Err(AgentError::Llm(format!(
                    "embedding index {position} is outside the {} requested inputs",
                    out.len()
                )));
            }
            let values = entry
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| AgentError::Llm("embedding entry has no embedding array".into()))?;
            let vector: Vec<f32> = values
                .iter()
                .filter_map(|value| value.as_f64().map(|float| float as f32))
                .collect();
            if vector.is_empty() {
                return Err(AgentError::Llm("empty embedding in batch".into()));
            }
            if let Some(expected) = self.declared_dimension {
                if vector.len() != expected {
                    return Err(AgentError::Llm(format!(
                        "embedding width {} does not match the configured {expected}",
                        vector.len()
                    )));
                }
            }
            out[position] = vector;
        }

        if let Some(missing) = out.iter().position(Vec::is_empty) {
            return Err(AgentError::Llm(format!(
                "embedding batch response never filled input {missing}"
            )));
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::testing::{request_head, spawn_http_capture};
    use std::time::Duration;

    fn body_of(request: &str) -> Value {
        let raw = request.split("\r\n\r\n").nth(1).unwrap_or_default();
        serde_json::from_str(raw).unwrap_or(Value::Null)
    }

    #[test]
    fn the_url_appends_embeddings_and_tolerates_a_trailing_slash() {
        let with = OpenAiCompatEmbedder::new("k", "https://example.invalid/v1/", "m");
        let without = OpenAiCompatEmbedder::new("k", "https://example.invalid/v1", "m");
        assert_eq!(with.embed_url(), "https://example.invalid/v1/embeddings");
        assert_eq!(without.embed_url(), with.embed_url());
    }

    #[tokio::test]
    async fn a_single_embedding_is_parsed_and_the_key_stays_in_the_header() {
        let secret = "OPENAI_EMBED_SECRET";
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"data":[{"index":0,"embedding":[0.5,-0.25,0.125]}]}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = OpenAiCompatEmbedder::new(secret, base_url, "text-embedding-3-small");
        let embedding = embedder.embed("fixture").await.unwrap();
        assert_eq!(embedding, vec![0.5, -0.25, 0.125]);

        let request = server.await.unwrap().expect("embedding request");
        let head = request_head(&request);
        assert!(
            !head.lines().next().unwrap_or_default().contains(secret),
            "the credential must never appear in the request line"
        );
        assert!(head.to_lowercase().contains("authorization:"));
        assert!(head.contains(secret));
    }

    #[tokio::test]
    async fn a_local_runtime_without_a_credential_sends_no_authorization_header() {
        // Ollama and vLLM need no key. Requiring one would exclude self-hosted
        // users, which is the opposite of the point.
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"data":[{"index":0,"embedding":[1.0,2.0]}]}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = OpenAiCompatEmbedder::new("", base_url, "nomic-embed-text");
        assert_eq!(embedder.embed("fixture").await.unwrap(), vec![1.0, 2.0]);

        let head = request_head(&server.await.unwrap().expect("embedding request"));
        assert!(!head.to_lowercase().contains("authorization:"));
    }

    #[tokio::test]
    async fn batching_uses_one_request_and_honours_the_returned_index() {
        // Deliberately out of order: relying on response order would misalign
        // every vector against its chunk.
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"data":[{"index":1,"embedding":[2.0]},{"index":0,"embedding":[1.0]}]}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = OpenAiCompatEmbedder::new("k", base_url, "m");
        let out = embedder.embed_batch(&["first", "second"]).await.unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0]]);

        let body = body_of(&server.await.unwrap().expect("batch request"));
        assert_eq!(body["input"], json!(["first", "second"]));
    }

    #[tokio::test]
    async fn a_short_batch_response_is_rejected_rather_than_silently_padded() {
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"data":[{"index":0,"embedding":[1.0]}]}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = OpenAiCompatEmbedder::new("k", base_url, "m");
        let error = embedder
            .embed_batch(&["a", "b"])
            .await
            .expect_err("a short batch must fail")
            .to_string();
        let _ = server.await.unwrap();
        assert!(error.contains("1 vectors for 2 inputs"), "got {error}");
    }

    #[tokio::test]
    async fn a_width_that_contradicts_the_configured_dimension_is_an_error() {
        // A server that ignores `dimensions` would otherwise poison an index
        // built for the requested width.
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let embedder =
            OpenAiCompatEmbedder::new("k", base_url, "m").with_requested_dimensions(1536);
        let error = embedder
            .embed("fixture")
            .await
            .expect_err("width mismatch must fail")
            .to_string();
        let _ = server.await.unwrap();
        assert!(
            error.contains("does not match the configured 1536"),
            "got {error}"
        );
    }

    #[tokio::test]
    async fn requested_dimensions_are_sent_and_treated_as_declared() {
        let (base_url, server) = spawn_http_capture(
            "200 OK",
            "application/json",
            br#"{"data":[{"index":0,"embedding":[0.1,0.2]}]}"#.to_vec(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = OpenAiCompatEmbedder::new("k", base_url, "m").with_requested_dimensions(2);
        assert_eq!(embedder.declared_dimension(), Some(2));
        embedder.embed("fixture").await.unwrap();

        let body = body_of(&server.await.unwrap().expect("embedding request"));
        assert_eq!(body["dimensions"], json!(2));
    }

    #[tokio::test]
    async fn error_bodies_are_sanitized_and_bounded() {
        let secret = "OPENAI_EMBED_ERROR_SECRET";
        let mut body = format!("denied for {secret} ");
        body.push_str(&"detail ".repeat(2048));
        let (base_url, server) = spawn_http_capture(
            "401 Unauthorized",
            "application/json",
            body.into_bytes(),
            Duration::from_secs(5),
        )
        .await;

        let embedder = OpenAiCompatEmbedder::new(secret, base_url, "m");
        let error = embedder.embed("fixture").await.unwrap_err().to_string();
        let _ = server.await.unwrap();

        assert!(!error.contains(secret));
        assert!(error.contains("[redacted]"));
        assert!(error.contains("401"));
        assert!(error.contains("...[truncated]"));
    }

    #[test]
    fn the_provider_label_is_overridable_so_backends_are_not_conflated() {
        let embedder = OpenAiCompatEmbedder::new("k", "https://example.invalid/v1", "m");
        assert_eq!(embedder.provider(), "openai-compatible");
        let labelled = OpenAiCompatEmbedder::new("k", "https://example.invalid/v1", "m")
            .with_provider("ollama");
        assert_eq!(labelled.provider(), "ollama");
    }
}
