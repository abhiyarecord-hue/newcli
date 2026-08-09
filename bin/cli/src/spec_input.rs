//! Bounded collection of the Specify stage description.
//!
//! A long description cannot be typed reliably at a terminal prompt, so this
//! module accepts one from a file as well. Two properties matter more than
//! convenience here:
//!
//! 1. **Nothing is silently truncated.** The old reader stopped at the first
//!    blank line, so a pasted document reached the model as its first paragraph
//!    only, and the remaining lines stayed in the stdin buffer where a later
//!    prompt read them as its answer. Both halves of that failure are
//!    eliminated: blank lines are preserved, and the caller ends input with an
//!    explicit terminator.
//! 2. **Nothing unbounded reaches the provider.** A description is capped, so
//!    an accidental multi-megabyte paste or file fails visibly instead of
//!    inflating the request and triggering context compaction.

use std::fmt;
use std::path::{Path, PathBuf};

/// Largest Specify description accepted from a file, a paste, or a pipe.
///
/// Generous for prose (roughly 15k words) while far below the point where a
/// single stage prompt would dominate the model's context window.
pub const MAX_DESCRIPTION_BYTES: usize = 64 * 1024;

/// Line that ends a typed or pasted description.
pub const DESCRIPTION_TERMINATOR: &str = "/end";

/// Why no usable description was collected. The caller must not run the stage
/// for any of these.
#[derive(Debug)]
pub enum SpecInputError {
    /// The description was absent or only whitespace.
    Empty,
    /// The description exceeded [`MAX_DESCRIPTION_BYTES`].
    TooLarge { bytes: u64, max: usize },
    /// The path exists but is not a regular file, or does not exist.
    NotAFile(PathBuf),
    /// The path could not be read.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The file is not valid UTF-8 text.
    NotUtf8(PathBuf),
}

impl fmt::Display for SpecInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(
                f,
                "no description given. Describe what to build, or pass --from-file <path>"
            ),
            Self::TooLarge { bytes, max } => write!(
                f,
                "description is {bytes} bytes, which exceeds the {max}-byte limit. \
                 Shorten it, or split the work into smaller specifications"
            ),
            Self::NotAFile(path) => write!(
                f,
                "description file '{}' is not a readable file",
                path.display()
            ),
            Self::Io { path, source } => {
                write!(
                    f,
                    "cannot read description file '{}': {source}",
                    path.display()
                )
            }
            Self::NotUtf8(path) => write!(
                f,
                "description file '{}' is not valid UTF-8 text",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SpecInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Read a description from `raw_path`, resolved against `project_root` when
/// relative.
///
/// The path comes from the person running the command, not from the model, so
/// it is not confined to the workspace: keeping requirement documents outside
/// the repository is normal. It is still checked to be a regular file of
/// bounded size holding UTF-8 text, and the file's own formatting — blank
/// lines, headings, lists — is passed through unchanged.
pub fn load_description_file(
    project_root: &Path,
    raw_path: &str,
) -> Result<String, SpecInputError> {
    let path = resolve_description_path(project_root, raw_path);

    let metadata = std::fs::metadata(&path).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            SpecInputError::NotAFile(path.clone())
        } else {
            SpecInputError::Io {
                path: path.clone(),
                source,
            }
        }
    })?;
    if !metadata.is_file() {
        return Err(SpecInputError::NotAFile(path));
    }
    // Checked before reading so an oversized file is never loaded into memory.
    if metadata.len() > MAX_DESCRIPTION_BYTES as u64 {
        return Err(SpecInputError::TooLarge {
            bytes: metadata.len(),
            max: MAX_DESCRIPTION_BYTES,
        });
    }

    let bytes = std::fs::read(&path).map_err(|source| SpecInputError::Io {
        path: path.clone(),
        source,
    })?;
    // Re-checked after reading: the file may have grown between the two calls.
    if bytes.len() > MAX_DESCRIPTION_BYTES {
        return Err(SpecInputError::TooLarge {
            bytes: bytes.len() as u64,
            max: MAX_DESCRIPTION_BYTES,
        });
    }

    let text = String::from_utf8(bytes).map_err(|_| SpecInputError::NotUtf8(path.clone()))?;
    if text.trim().is_empty() {
        return Err(SpecInputError::Empty);
    }
    Ok(text)
}

/// Accept a typed or piped description, applying the same bound and emptiness
/// rules as a file.
pub fn check_typed_description(text: &str) -> Result<String, SpecInputError> {
    if text.len() > MAX_DESCRIPTION_BYTES {
        return Err(SpecInputError::TooLarge {
            bytes: text.len() as u64,
            max: MAX_DESCRIPTION_BYTES,
        });
    }
    if text.trim().is_empty() {
        return Err(SpecInputError::Empty);
    }
    Ok(text.to_string())
}

/// Detect the `@path/to/file.md` form, which lets the interactive session load
/// a description from a file without a command-line flag.
///
/// Only a lone reference counts. Text that merely begins with `@` and then
/// continues onto further lines is a description, not a file reference.
pub fn parse_file_reference(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    let reference = trimmed.strip_prefix('@')?.trim();
    if reference.is_empty() || reference.contains('\n') {
        return None;
    }
    Some(reference)
}

/// Resolve a user-supplied path, tolerating the quotes a shell paste leaves
/// behind around a path containing spaces.
fn resolve_description_path(project_root: &Path, raw_path: &str) -> PathBuf {
    let trimmed = raw_path.trim();
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|rest| rest.strip_suffix('\''))
        })
        .unwrap_or(trimmed);

    let path = Path::new(unquoted);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_root.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_description_preserves_blank_lines_and_structure() {
        // The failure this module exists to prevent: a description with
        // paragraph breaks must arrive whole, not as its first paragraph.
        let dir = tempfile::tempdir().unwrap();
        let body = "# Goal\n\nBuild a todo CLI.\n\n## Notes\n\n- keep it small\n";
        std::fs::write(dir.path().join("req.md"), body).unwrap();

        let loaded = load_description_file(dir.path(), "req.md").unwrap();
        assert_eq!(loaded, body);
        assert_eq!(loaded.lines().count(), 7);
    }

    #[test]
    fn relative_paths_resolve_against_the_project_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("docs")).unwrap();
        std::fs::write(dir.path().join("docs/req.md"), "build a parser").unwrap();

        assert_eq!(
            load_description_file(dir.path(), "docs/req.md").unwrap(),
            "build a parser"
        );
    }

    #[test]
    fn absolute_paths_outside_the_project_root_are_accepted() {
        // The path is typed by the operator, so a requirement document kept
        // outside the repository must still load.
        let outside = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let path = outside.path().join("req.md");
        std::fs::write(&path, "build a parser").unwrap();

        let loaded = load_description_file(project.path(), path.to_str().unwrap()).unwrap();
        assert_eq!(loaded, "build a parser");
    }

    #[test]
    fn quoted_paths_are_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("my req.md"), "build it").unwrap();

        assert_eq!(
            load_description_file(dir.path(), "\"my req.md\"").unwrap(),
            "build it"
        );
    }

    #[test]
    fn missing_path_directory_and_empty_file_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("adir")).unwrap();
        std::fs::write(dir.path().join("blank.md"), "   \n\n\t\n").unwrap();

        assert!(matches!(
            load_description_file(dir.path(), "absent.md"),
            Err(SpecInputError::NotAFile(_))
        ));
        assert!(matches!(
            load_description_file(dir.path(), "adir"),
            Err(SpecInputError::NotAFile(_))
        ));
        assert!(matches!(
            load_description_file(dir.path(), "blank.md"),
            Err(SpecInputError::Empty)
        ));
    }

    #[test]
    fn oversized_file_is_rejected_at_the_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let at_cap = "a".repeat(MAX_DESCRIPTION_BYTES);
        let over_cap = "a".repeat(MAX_DESCRIPTION_BYTES + 1);
        std::fs::write(dir.path().join("at.md"), &at_cap).unwrap();
        std::fs::write(dir.path().join("over.md"), &over_cap).unwrap();

        assert_eq!(load_description_file(dir.path(), "at.md").unwrap(), at_cap);
        assert!(matches!(
            load_description_file(dir.path(), "over.md"),
            Err(SpecInputError::TooLarge { .. })
        ));
    }

    #[test]
    fn non_utf8_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("bad.md"), [0xff, 0xfe, 0x00]).unwrap();

        assert!(matches!(
            load_description_file(dir.path(), "bad.md"),
            Err(SpecInputError::NotUtf8(_))
        ));
    }

    #[test]
    fn typed_description_shares_the_file_bounds() {
        assert_eq!(
            check_typed_description("build a todo").unwrap(),
            "build a todo"
        );
        assert!(matches!(
            check_typed_description("  \n\t\n"),
            Err(SpecInputError::Empty)
        ));
        assert!(matches!(
            check_typed_description(&"a".repeat(MAX_DESCRIPTION_BYTES + 1)),
            Err(SpecInputError::TooLarge { .. })
        ));
        assert!(check_typed_description(&"a".repeat(MAX_DESCRIPTION_BYTES)).is_ok());
    }

    #[test]
    fn file_reference_is_recognized_only_when_it_stands_alone() {
        assert_eq!(parse_file_reference("@req.md\n"), Some("req.md"));
        assert_eq!(
            parse_file_reference("  @ docs/req.md  "),
            Some("docs/req.md")
        );
        assert_eq!(parse_file_reference("@"), None);
        assert_eq!(parse_file_reference("req.md"), None);
        // A description that happens to start with '@' is not a reference.
        assert_eq!(parse_file_reference("@mention the user\nthen do X"), None);
    }
}
