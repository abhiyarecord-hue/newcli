//! Gemini text embeddings via the `embedContent` REST API.
//!
//! Uses the same base URL and API key as the main Gemini provider.
//! Returns 768-dimensional vectors (text-embedding-004 model).

use agent_types::{AgentError, Result};
use serde_json::{json, Value};

const EMBEDDING_MODEL: &str = "text-embedding-004";

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

    /// Generate an embedding for a single text string.
    /// Returns a 768-dimensional f32 vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!(
            "{}/models/{}:embedContent?key={}",
            self.base_url, EMBEDDING_MODEL, self.api_key
        );

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
            .body(serde_json::to_vec(&body).map_err(|e| AgentError::Llm(e.to_string()))?)
            .send()
            .await
            .map_err(|e| AgentError::Llm(format!("embedding request: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AgentError::Llm(format!("embedding http {status}: {text}")));
        }

        let response: Value = resp
            .json()
            .await
            .map_err(|e| AgentError::Llm(format!("embedding parse: {e}")))?;

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

    /// Batch embed multiple texts. Returns one vector per input text.
    /// Calls the API once per text (Gemini embedContent doesn't batch natively).
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_constructs() {
        let e = GeminiEmbedder::new("fake-key", "https://example.com/v1beta");
        assert!(!e.api_key.is_empty());
    }
}
