//! Merkle tree for incremental codebase sync.
//!
//! File-level hash: SHA-256(content). Dir-level hash: SHA-256(sorted(child_name || child_hash)).
//! `diff()` prunes subtrees with equal dir-hashes — editing 1 file in a 50k-file
//! tree touches only O(depth) hashes. Serialize to `.agent/index.merkle` via serde.
//! `.gitignore` patterns are respected via the `ignore` crate. Symlinks are skipped
//! entirely (cycle-safety, TASK-2.3 guard).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use agent_types::{AgentError, Result};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MerkleTree {
    pub root_hash: [u8; 32],
    pub nodes: HashMap<PathBuf, [u8; 32]>,
}

impl MerkleTree {
    /// Build a Merkle tree rooted at `root`, respecting `.gitignore`.
    /// Symlinks are skipped. Errors on individual files are logged and skipped.
    pub fn build(root: &Path) -> Result<Self> {
        let root = std::fs::canonicalize(root)
            .map_err(|e| AgentError::Index(format!("canonicalize root: {e}")))?;

        let mut nodes: HashMap<PathBuf, [u8; 32]> = HashMap::new();

        // Walk using `ignore` crate which respects .gitignore, skips hidden, no symlinks.
        let walker = ignore::WalkBuilder::new(&root)
            .hidden(true) // skip hidden files/dirs (like .git)
            .follow_links(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .sort_by_file_name(|a, b| a.cmp(b))
            .build();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path().to_path_buf();

            // Skip the root dir itself, symlinks, and non-regular files.
            if path == root {
                continue;
            }
            if entry.path_is_symlink() {
                continue;
            }
            let ft = match entry.file_type() {
                Some(ft) => ft,
                None => continue,
            };
            if !ft.is_file() {
                continue;
            }

            // Relative path from root for stable keys.
            let rel = match path.strip_prefix(&root) {
                Ok(r) => r.to_path_buf(),
                Err(_) => continue,
            };

            // SHA-256 of file content.
            match std::fs::read(&path) {
                Ok(content) => {
                    let hash: [u8; 32] = Sha256::digest(&content).into();
                    nodes.insert(rel, hash);
                }
                Err(_) => continue, // skip unreadable files
            }
        }

        // Build dir hashes bottom-up.
        let dir_hashes = compute_dir_hashes(&nodes);
        for (dir, hash) in &dir_hashes {
            nodes.insert(dir.clone(), *hash);
        }

        // Root hash = hash of top-level entries.
        let root_hash = dir_hashes
            .get(&PathBuf::new())
            .copied()
            .unwrap_or_else(|| {
                // Empty tree.
                let h: [u8; 32] = Sha256::digest(b"").into();
                h
            });

        Ok(MerkleTree { root_hash, nodes })
    }

    /// Diff two trees, returning paths of **files** that changed (added, removed, modified).
    /// Prunes descent into directories with equal hashes.
    pub fn diff(&self, other: &MerkleTree) -> Vec<PathBuf> {
        let mut changed: Vec<PathBuf> = Vec::new();

        // Collect all unique paths.
        let mut all_paths: Vec<&PathBuf> = self.nodes.keys().chain(other.nodes.keys()).collect();
        all_paths.sort();
        all_paths.dedup();

        // Identify directories in both trees.
        let self_dirs: std::collections::HashSet<&PathBuf> = self
            .nodes
            .keys()
            .filter(|p| is_dir_key(p, &self.nodes))
            .collect();
        let other_dirs: std::collections::HashSet<&PathBuf> = other
            .nodes
            .keys()
            .filter(|p| is_dir_key(p, &other.nodes))
            .collect();

        // Pruned dirs: dirs present in both with equal hash.
        let mut pruned: std::collections::HashSet<&PathBuf> = std::collections::HashSet::new();
        for d in self_dirs.intersection(&other_dirs) {
            if self.nodes.get(*d) == other.nodes.get(*d) {
                pruned.insert(*d);
            }
        }

        for path in &all_paths {
            // Skip dir-key entries themselves.
            if self_dirs.contains(path) || other_dirs.contains(path) {
                continue;
            }
            // Skip if any ancestor dir is pruned.
            if pruned.iter().any(|d| {
                !d.as_os_str().is_empty() && path.starts_with(d)
            }) {
                continue;
            }
            // File comparison.
            let h1 = self.nodes.get(*path);
            let h2 = other.nodes.get(*path);
            if h1 != h2 {
                changed.push((*path).clone());
            }
        }

        changed
    }

    /// Serialize to a file (`.agent/index.merkle`).
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_vec(self)
            .map_err(|e| AgentError::Index(format!("serialize merkle: {e}")))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &data)?;
        Ok(())
    }

    /// Deserialize from a file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let tree: MerkleTree = serde_json::from_slice(&data)
            .map_err(|e| AgentError::Index(format!("deserialize merkle: {e}")))?;
        Ok(tree)
    }
}

/// Compute directory hashes bottom-up. A dir hash = SHA-256(sorted(child_name || child_hash)).
/// The empty PathBuf ("") represents the virtual root directory.
fn compute_dir_hashes(file_nodes: &HashMap<PathBuf, [u8; 32]>) -> HashMap<PathBuf, [u8; 32]> {
    // Collect all directory paths (including "" for root).
    let mut dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    dirs.insert(PathBuf::new()); // root
    for path in file_nodes.keys() {
        let mut p = path.clone();
        while p.pop() {
            dirs.insert(p.clone());
        }
    }

    // Sort dirs by depth (deepest first) for bottom-up processing.
    let mut dir_list: Vec<PathBuf> = dirs.into_iter().collect();
    dir_list.sort_by(|a, b| {
        let da = a.components().count();
        let db = b.components().count();
        db.cmp(&da) // deepest first
    });

    let mut dir_hashes: HashMap<PathBuf, [u8; 32]> = HashMap::new();

    for dir in &dir_list {
        // Collect immediate children (files and already-computed subdirs).
        let mut children: Vec<(String, [u8; 32])> = Vec::new();

        // File children.
        for (path, hash) in file_nodes {
            if let Some(parent) = path.parent() {
                let parent_path = parent.to_path_buf();
                if parent_path == *dir {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    children.push((name, *hash));
                }
            }
        }

        // Subdir children.
        for (sub_dir, hash) in &dir_hashes {
            if let Some(parent) = sub_dir.parent() {
                let parent_path = parent.to_path_buf();
                if parent_path == *dir {
                    let name = sub_dir
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    children.push((name, *hash));
                }
            }
        }

        // Sort children byte-wise by name (deterministic, TASK-2.3 guard).
        children.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

        // Hash.
        let mut hasher = Sha256::new();
        for (name, hash) in &children {
            hasher.update(name.as_bytes());
            hasher.update(hash);
        }
        let result: [u8; 32] = hasher.finalize().into();
        dir_hashes.insert(dir.clone(), result);
    }

    dir_hashes
}

/// Check if a key is a dir key (exists in nodes but was computed as a dir hash).
fn is_dir_key(path: &PathBuf, nodes: &HashMap<PathBuf, [u8; 32]>) -> bool {
    // A dir key is one that has children in the map.
    if path.as_os_str().is_empty() {
        return true; // root is always a dir
    }
    nodes.keys().any(|k| k != path && k.starts_with(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup_temp_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("b.rs"), "fn helper() {}").unwrap();
        fs::create_dir_all(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub").join("c.rs"), "mod c;").unwrap();
        dir
    }

    #[test]
    fn build_produces_file_hashes() {
        let dir = setup_temp_dir();
        let tree = MerkleTree::build(dir.path()).unwrap();
        // Should contain file entries.
        assert!(tree.nodes.contains_key(&PathBuf::from("a.rs")));
        assert!(tree.nodes.contains_key(&PathBuf::from("b.rs")));
        assert!(tree.nodes.contains_key(&PathBuf::from("sub\\c.rs"))
            || tree.nodes.contains_key(&PathBuf::from("sub/c.rs")));
    }

    #[test]
    fn identical_trees_diff_empty() {
        let dir = setup_temp_dir();
        let t1 = MerkleTree::build(dir.path()).unwrap();
        let t2 = MerkleTree::build(dir.path()).unwrap();
        assert!(t1.diff(&t2).is_empty());
    }

    #[test]
    fn editing_one_file_diffs_exactly_that_path() {
        let dir = setup_temp_dir();
        let t1 = MerkleTree::build(dir.path()).unwrap();
        // Modify a.rs.
        fs::write(dir.path().join("a.rs"), "fn main() { changed }").unwrap();
        let t2 = MerkleTree::build(dir.path()).unwrap();
        let diff = t1.diff(&t2);
        assert_eq!(diff.len(), 1);
        assert!(diff[0].to_string_lossy().contains("a.rs"));
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = setup_temp_dir();
        let tree = MerkleTree::build(dir.path()).unwrap();
        let save_path = dir.path().join(".agent").join("index.merkle");
        tree.save(&save_path).unwrap();
        let loaded = MerkleTree::load(&save_path).unwrap();
        assert_eq!(tree.root_hash, loaded.root_hash);
        assert_eq!(tree.nodes.len(), loaded.nodes.len());
    }

    #[test]
    fn added_file_shows_in_diff() {
        let dir = setup_temp_dir();
        let t1 = MerkleTree::build(dir.path()).unwrap();
        fs::write(dir.path().join("new.rs"), "fn new() {}").unwrap();
        let t2 = MerkleTree::build(dir.path()).unwrap();
        let diff = t1.diff(&t2);
        assert!(diff.iter().any(|p| p.to_string_lossy().contains("new.rs")));
    }

    #[test]
    fn deleted_file_shows_in_diff() {
        let dir = setup_temp_dir();
        let t1 = MerkleTree::build(dir.path()).unwrap();
        fs::remove_file(dir.path().join("b.rs")).unwrap();
        let t2 = MerkleTree::build(dir.path()).unwrap();
        let diff = t1.diff(&t2);
        assert!(diff.iter().any(|p| p.to_string_lossy().contains("b.rs")));
    }
}
