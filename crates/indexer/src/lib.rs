//! `indexer` (L2): tree-sitter parsing, AST chunking, scope tree, Merkle sync.
//!
//! - [`entity`]: [`CodeEntity`] / [`EntityKind`] / [`Chunk`] + stable id hash.
//! - [`parser`]: iterative tree-sitter entity extraction (TASK-2.1).
//! - `chunker`: greedy recursive AST chunking (TASK-2.2).
//! - `merkle`: incremental sync (TASK-2.3).

pub mod chunker;
pub mod entity;
pub mod merkle;
pub mod parser;
pub mod scope_tree;

pub use chunker::chunk;
pub use entity::{stable_id, Chunk, CodeEntity, EntityKind};
pub use merkle::MerkleTree;
pub use parser::{parse, Language};
