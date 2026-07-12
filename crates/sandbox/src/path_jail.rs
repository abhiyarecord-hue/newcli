//! Path jail: sandboxes all file access to a canonicalized root.
//!
//! This is the security boundary for EVERY file tool — no fast paths, no
//! caching. `canonicalize` fails on paths that don't exist yet (new files!), so
//! we canonicalize the deepest existing ancestor and verify it starts_with(root),
//! then re-append the non-existing tail.
//!
//! Checking for `..` as a substring is wrong (`..foo` is a legal filename);
//! we check `Component::ParentDir` specifically. Symlinks pointing outside the
//! jail are rejected.

use std::path::{Component, Path, PathBuf};

use agent_types::{AgentError, Result};

pub struct PathJail {
    root: PathBuf, // canonicalized
}

impl PathJail {
    /// Create a new jail rooted at `root`. Canonicalizes on construction.
    pub fn new(root: &Path) -> Result<Self> {
        let root = std::fs::canonicalize(root).map_err(|e| {
            AgentError::PathJail(format!("cannot canonicalize root '{}': {e}", root.display()))
        })?;
        Ok(Self { root })
    }

    /// The canonicalized jail root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a user-provided path, verifying it stays inside the jail.
    ///
    /// - Absolute paths that don't start with root → error.
    /// - Relative paths are joined to root.
    /// - The deepest existing ancestor is canonicalized and checked.
    /// - `Component::ParentDir` (`..`) is explicitly rejected.
    pub fn resolve(&self, user_path: &Path) -> Result<PathBuf> {
        // Reject absolute paths not under root.
        if user_path.is_absolute() {
            // Try canonicalizing the deepest existing part.
            let (existing, tail) = split_existing(user_path);
            let canon = std::fs::canonicalize(&existing).map_err(|e| {
                AgentError::PathJail(format!("cannot resolve '{}': {e}", user_path.display()))
            })?;
            if !canon.starts_with(&self.root) {
                return Err(AgentError::PathJail(format!(
                    "absolute path '{}' is outside jail '{}'",
                    user_path.display(),
                    self.root.display()
                )));
            }
            let resolved = canon.join(tail);
            // Final check: ensure no symlink in tail escapes.
            return self.verify_no_escape(&resolved);
        }

        // Check for ParentDir components.
        for component in user_path.components() {
            if matches!(component, Component::ParentDir) {
                return Err(AgentError::PathJail(format!(
                    "path '{}' contains '..' component",
                    user_path.display()
                )));
            }
        }

        // Join with root.
        let joined = self.root.join(user_path);

        // If it fully exists, canonicalize directly (simplest and most reliable).
        if joined.exists() {
            let canon = std::fs::canonicalize(&joined).map_err(|e| {
                AgentError::PathJail(format!("cannot resolve '{}': {e}", joined.display()))
            })?;
            if !canon.starts_with(&self.root) {
                return Err(AgentError::PathJail(format!(
                    "resolved path '{}' escapes jail '{}'",
                    canon.display(),
                    self.root.display()
                )));
            }
            return Ok(canon);
        }

        // Canonicalize deepest existing ancestor.
        let (existing, tail) = split_existing(&joined);
        let canon = std::fs::canonicalize(&existing).map_err(|e| {
            AgentError::PathJail(format!("cannot resolve '{}': {e}", joined.display()))
        })?;

        if !canon.starts_with(&self.root) {
            return Err(AgentError::PathJail(format!(
                "resolved path '{}' escapes jail '{}'",
                canon.display(),
                self.root.display()
            )));
        }

        let resolved = canon.join(tail);
        Ok(resolved)
    }

    /// Ensure the final resolved path still starts_with root.
    fn verify_no_escape(&self, resolved: &Path) -> Result<PathBuf> {
        // If the resolved path fully exists, canonicalize and check.
        if resolved.exists() {
            let canon = std::fs::canonicalize(resolved).map_err(|e| {
                AgentError::PathJail(format!(
                    "cannot canonicalize '{}': {e}",
                    resolved.display()
                ))
            })?;
            if !canon.starts_with(&self.root) {
                return Err(AgentError::PathJail(format!(
                    "path '{}' (resolved to '{}') escapes jail via symlink",
                    resolved.display(),
                    canon.display()
                )));
            }
            return Ok(canon);
        }
        // For non-existing paths (new files): the parent is already verified.
        Ok(resolved.to_path_buf())
    }
}

/// Split a path into (deepest_existing_ancestor, remaining_tail).
fn split_existing(path: &Path) -> (PathBuf, PathBuf) {
    let mut existing = path.to_path_buf();
    let mut tail_parts: Vec<std::ffi::OsString> = Vec::new();

    while !existing.exists() {
        if let Some(name) = existing.file_name() {
            tail_parts.push(name.to_os_string());
            existing.pop();
        } else {
            break;
        }
    }

    tail_parts.reverse();
    let tail: PathBuf = tail_parts.iter().collect();
    (existing, tail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolve_valid_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src").join("main.rs"), "fn main(){}").unwrap();

        let jail = PathJail::new(dir.path()).unwrap();
        let resolved = jail.resolve(Path::new("src")).unwrap();
        assert!(resolved.exists());
        assert!(resolved.starts_with(jail.root()));

        // Also test a file inside a subdir.
        let file_resolved = jail.resolve(&Path::new("src").join("main.rs")).unwrap();
        assert!(file_resolved.exists());
        assert!(file_resolved.starts_with(jail.root()));
    }

    #[test]
    fn resolve_new_file_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let jail = PathJail::new(dir.path()).unwrap();
        // File doesn't exist yet — should still resolve (for writes).
        let resolved = jail.resolve(Path::new("new_file.rs")).unwrap();
        assert!(resolved.starts_with(jail.root()));
    }

    #[test]
    fn parent_dir_component_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let jail = PathJail::new(dir.path()).unwrap();
        let result = jail.resolve(Path::new("../../etc/passwd"));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains(".."));
    }

    #[test]
    fn absolute_path_outside_jail_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let jail = PathJail::new(dir.path()).unwrap();

        // An absolute path that's clearly outside.
        #[cfg(windows)]
        let outside = Path::new("C:\\Windows\\System32\\cmd.exe");
        #[cfg(not(windows))]
        let outside = Path::new("/etc/passwd");

        let result = jail.resolve(outside);
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_rejected() {
        use std::os::unix::fs as unix_fs;

        let dir = tempfile::tempdir().unwrap();
        let jail = PathJail::new(dir.path()).unwrap();

        // Create a symlink inside the jail pointing to /etc.
        unix_fs::symlink("/etc", dir.path().join("escape_link")).unwrap();

        let result = jail.resolve(Path::new("escape_link/passwd"));
        assert!(result.is_err());
    }
}
