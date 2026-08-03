//! `vecstore` (L2): SQLite + sqlite-vec + FTS5, hybrid 3-mode retrieval.
//!
//! - [`schema`]: idempotent migrations, table definitions.
//! - [`store`]: [`VecStore`] — open, upsert, insert, delete, count.
//! - [`hybrid`]: 3-mode retrieval (Vector, Keyword, Graph, Hybrid + RRF fusion).

pub mod hybrid;
pub mod schema;
pub mod store;

pub use hybrid::{search, search_with_profile, SearchHit, SearchMode, SearchReport};
pub use schema::{ensure_vector_dimension, vector_dimension};
pub use store::{
    ChunkInsert, EmbeddingProfile, FileRecord, ReplacementStep, VecStore, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER, EMBEDDING_DIMENSION,
};
