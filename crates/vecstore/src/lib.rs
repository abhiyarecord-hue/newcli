//! `vecstore` (L2): SQLite + sqlite-vec + FTS5, hybrid 3-mode retrieval.
//!
//! - [`schema`]: idempotent migrations, table definitions.
//! - [`store`]: [`VecStore`] — open, upsert, insert, delete, count.
//! - [`hybrid`]: 3-mode retrieval (Vector, Keyword, Graph, Hybrid + RRF fusion).

pub mod hybrid;
pub mod schema;
pub mod store;

pub use hybrid::{search, SearchHit, SearchMode};
pub use store::{ChunkInsert, VecStore};
