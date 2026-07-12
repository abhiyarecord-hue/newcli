//! Semantic diff: entity-level changes between two versions of a file.
//!
//! Parses both versions with `indexer::parser`, keys entities by
//! `(qualified_name, kind)`, compares body *content* by hash (never byte_range).
//! Whitespace-only changes are normalized away so reformats don't flood diffs.

use std::collections::HashMap;
use std::path::Path;

use agent_types::Result;
use indexer::parser::Language;
use indexer::{parse, CodeEntity, EntityKind};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChangeKind {
    Added,
    Removed,
    Modified,
}

#[derive(Clone, Debug)]
pub struct EntityChange {
    pub kind: ChangeKind,
    pub entity: CodeEntity,
}

/// Compute semantic diff between two versions of the same file.
pub fn semantic_diff(old_src: &str, new_src: &str, path: &Path) -> Result<Vec<EntityChange>> {
    // If language unsupported, return empty.
    if Language::from_path(path).is_none() {
        return Ok(Vec::new());
    }

    let old_entities = parse(path, old_src)?;
    let new_entities = parse(path, new_src)?;

    let old_map = build_map(&old_entities, old_src);
    let new_map = build_map(&new_entities, new_src);

    let mut changes = Vec::new();

    // Find removed / modified.
    for (key, (entity, body_hash)) in &old_map {
        match new_map.get(key) {
            None => changes.push(EntityChange {
                kind: ChangeKind::Removed,
                entity: (*entity).clone(),
            }),
            Some((_, new_hash)) if new_hash != body_hash => {
                changes.push(EntityChange {
                    kind: ChangeKind::Modified,
                    entity: (*new_map[key].0).clone(),
                });
            }
            _ => {} // unchanged
        }
    }

    // Find added.
    for (key, (entity, _)) in &new_map {
        if !old_map.contains_key(key) {
            changes.push(EntityChange {
                kind: ChangeKind::Added,
                entity: (*entity).clone(),
            });
        }
    }

    Ok(changes)
}

type EntityKey = (String, EntityKind);

/// Build a map of `(qualified_name, kind) -> (entity, normalized_body_hash)`.
fn build_map<'a>(
    entities: &'a [CodeEntity],
    source: &'a str,
) -> HashMap<EntityKey, (&'a CodeEntity, [u8; 32])> {
    let mut map = HashMap::new();
    for entity in entities {
        let body = source
            .get(entity.byte_range.0..entity.byte_range.1)
            .unwrap_or("");
        let normalized = normalize_whitespace(body);
        let hash: [u8; 32] = Sha256::digest(normalized.as_bytes()).into();
        let key = (entity.qualified_name.clone(), entity.kind);
        map.insert(key, (entity, hash));
    }
    map
}

/// Normalize whitespace for comparison: collapse runs of whitespace to single
/// space, trim. This prevents reformat noise from polluting the diff.
fn normalize_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_ws {
                result.push(' ');
                in_ws = true;
            }
        } else {
            result.push(c);
            in_ws = false;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn rename_yields_removed_plus_added() {
        let old_src = "fn alpha() {}\nfn beta() {}\n";
        let new_src = "fn alpha() {}\nfn gamma() {}\n";
        let path = PathBuf::from("lib.rs");

        let changes = semantic_diff(old_src, new_src, &path).unwrap();
        let removed: Vec<_> = changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Removed)
            .collect();
        let added: Vec<_> = changes
            .iter()
            .filter(|c| c.kind == ChangeKind::Added)
            .collect();

        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].entity.qualified_name, "beta");
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].entity.qualified_name, "gamma");
    }

    #[test]
    fn whitespace_only_change_is_not_modified() {
        let old_src = "fn hello() {\n    body\n}";
        let new_src = "fn hello() {\n        body\n}"; // extra indent
        let path = PathBuf::from("m.rs");
        let changes = semantic_diff(old_src, new_src, &path).unwrap();
        assert!(changes.is_empty(), "whitespace change should not show: {changes:?}");
    }

    #[test]
    fn body_change_is_modified() {
        let old_src = "fn hello() { let x = 1; }";
        let new_src = "fn hello() { let x = 2; }";
        let path = PathBuf::from("m.rs");
        let changes = semantic_diff(old_src, new_src, &path).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].kind, ChangeKind::Modified);
        assert_eq!(changes[0].entity.qualified_name, "hello");
    }

    #[test]
    fn unsupported_language_returns_empty() {
        let changes = semantic_diff("hello", "world", &PathBuf::from("notes.txt")).unwrap();
        assert!(changes.is_empty());
    }
}
