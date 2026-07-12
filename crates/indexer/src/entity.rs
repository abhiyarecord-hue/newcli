//! Code entity model + stable id hashing. Signatures verbatim from
//! plan.md section 3.

use std::path::PathBuf;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CodeEntity {
    pub id: u64, // stable hash of (path, kind, qualified_name)
    pub kind: EntityKind,
    pub qualified_name: String, // e.g. "MyClass::my_method"
    pub signature: String,
    pub docstring: Option<String>,
    pub path: PathBuf,
    pub byte_range: (usize, usize),
    pub line_range: (u32, u32),
    pub parent_id: Option<u64>, // scope tree edge
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EntityKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Module,
    Block,
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub file: PathBuf,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
    pub token_count: u32,
    pub entity_ids: Vec<u64>,
}

/// Deterministic FNV-1a 64-bit hash — stable across runs and platforms (unlike
/// `DefaultHasher`), so entity ids persist correctly in the index.
pub fn stable_id(path: &std::path::Path, kind: EntityKind, qualified_name: &str) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    };
    mix(path.to_string_lossy().as_bytes());
    mix(b"|");
    mix(&[kind as u8]);
    mix(b"|");
    mix(qualified_name.as_bytes());
    hash
}
