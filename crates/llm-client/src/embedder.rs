//! Provider-agnostic text embedding.
//!
//! Semantic search must not be tied to one vendor. This project is meant to work
//! with whatever a user already has: a hosted closed-source API, a self-hosted
//! open-weights server, or a local runtime with no credential at all. So the
//! indexer depends on the [`Embedder`] trait rather than on any concrete client.
//!
//! Two implementations cover the practical field:
//!
//! - [`crate::GeminiEmbedder`] for Gemini's `embedContent` shape.
//! - [`OpenAiCompatEmbedder`] for the `POST {base}/embeddings` shape, which is
//!   what OpenAI, Azure AI Foundry, Mistral, DeepSeek, Together, OpenRouter,
//!   Ollama, LM Studio, and vLLM all speak.
//!
//! **Dimension is treated as an observed property, not an assumption.** Embedding
//! models disagree: 768, 1024, 1536, and 3072 are all common. Storing vectors of
//! one width into an index built for another silently corrupts retrieval, so a
//! dimension is either declared by the implementation or probed once against the
//! real endpoint before indexing begins.

use agent_types::Result;

/// A text embedding backend.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Stable provider identifier recorded alongside every stored vector, so a
    /// later search can refuse to compare vectors from a different backend.
    fn provider(&self) -> &str;

    /// Exact model or deployment name recorded alongside every stored vector.
    fn model(&self) -> &str;

    /// Vector width, when it is known without contacting the endpoint.
    ///
    /// `None` is the honest answer for an OpenAI-compatible endpoint serving an
    /// arbitrary model: the width is whatever that model returns, and guessing it
    /// would be worse than measuring it.
    fn declared_dimension(&self) -> Option<usize> {
        None
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed several texts. Implementations that support native batching should
    /// override this; the default issues one request per input.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(self.embed(text).await?);
        }
        Ok(out)
    }
}

/// Determine the vector width this embedder will actually produce.
///
/// Uses the declared width when the implementation knows it, and otherwise
/// measures it with a single tiny request. Measuring costs one call at startup
/// and removes an entire class of silent corruption, which is a trade worth
/// making: an index built at the wrong width does not fail loudly, it just
/// returns wrong neighbours.
pub async fn resolve_dimension(embedder: &dyn Embedder) -> Result<usize> {
    if let Some(dimension) = embedder.declared_dimension() {
        return Ok(dimension);
    }
    let probe = embedder.embed("dimension probe").await?;
    Ok(probe.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Fake {
        width: usize,
        declared: Option<usize>,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Embedder for Fake {
        fn provider(&self) -> &str {
            "fake"
        }
        fn model(&self) -> &str {
            "fake-model"
        }
        fn declared_dimension(&self) -> Option<usize> {
            self.declared
        }
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.1; self.width])
        }
    }

    #[tokio::test]
    async fn a_declared_dimension_is_trusted_without_a_request() {
        let fake = Fake {
            width: 1536,
            declared: Some(1536),
            calls: AtomicUsize::new(0),
        };
        assert_eq!(resolve_dimension(&fake).await.unwrap(), 1536);
        assert_eq!(
            fake.calls.load(Ordering::SeqCst),
            0,
            "a declared width must not cost a network call"
        );
    }

    #[tokio::test]
    async fn an_undeclared_dimension_is_measured_once() {
        for width in [768, 1024, 1536, 3072] {
            let fake = Fake {
                width,
                declared: None,
                calls: AtomicUsize::new(0),
            };
            assert_eq!(resolve_dimension(&fake).await.unwrap(), width);
            assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
        }
    }

    #[tokio::test]
    async fn batching_defaults_to_one_request_per_input() {
        let fake = Fake {
            width: 4,
            declared: None,
            calls: AtomicUsize::new(0),
        };
        let out = fake.embed_batch(&["a", "b", "c"]).await.unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|vector| vector.len() == 4));
        assert_eq!(fake.calls.load(Ordering::SeqCst), 3);
    }
}
