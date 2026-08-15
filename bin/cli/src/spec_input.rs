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

/// What [`expand_file_references`] did, so the caller can report it.
#[derive(Debug, Default)]
pub struct ExpandedDescription {
    pub text: String,
    /// Files that were read in, in the order they appeared.
    pub loaded: Vec<LoadedFile>,
    /// Tokens that looked like a file but resolved to nothing.
    pub unresolved: Vec<String>,
}

#[derive(Debug)]
pub struct LoadedFile {
    pub reference: String,
    pub bytes: usize,
    pub lines: usize,
}

/// Replace every `@<path>` reference in `text` with the contents of that file.
///
/// A reference used to count only when it was the entire description, so
/// `read @req.md and do it` sent the literal `@req.md` to the model, which
/// answered that it cannot open local files. A silent no-op is the worst
/// possible outcome here: the pipeline then continued on a refusal. References
/// are therefore honoured wherever they appear, and anything that looks like a
/// file but does not resolve is reported to the caller instead of being passed
/// through unnoticed.
///
/// A token is only expanded when it names a readable file, so an email address
/// or an `@scope/package` mention is left alone.
pub fn expand_file_references(
    project_root: &Path,
    text: &str,
) -> Result<ExpandedDescription, SpecInputError> {
    let mut out = String::with_capacity(text.len());
    let mut result = ExpandedDescription::default();
    let mut budget = MAX_DESCRIPTION_BYTES;
    let mut rest = text;

    while let Some(at) = rest.find('@') {
        out.push_str(&rest[..at]);
        let after = &rest[at + 1..];

        // A reference starts a word. An `@` with a word character before it is
        // infix punctuation — an email address, a Rust label — not a path, and
        // treating it as one reported `b.com` out of `a@b.com`.
        let preceded_by_word = rest[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | '\\'));
        if preceded_by_word {
            out.push('@');
            rest = after;
            continue;
        }

        let token: String = after
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '@')
            .collect();
        // Trailing sentence punctuation is not part of a filename.
        let token = token.trim_end_matches([',', ';', ':', ')', ']', '"', '\'']);

        if token.is_empty() {
            out.push('@');
            rest = after;
            continue;
        }

        match load_description_file(project_root, token) {
            Ok(contents) => {
                if contents.len() > budget {
                    return Err(SpecInputError::TooLarge {
                        bytes: contents.len() as u64,
                        max: MAX_DESCRIPTION_BYTES,
                    });
                }
                budget -= contents.len();
                result.loaded.push(LoadedFile {
                    reference: token.to_string(),
                    bytes: contents.len(),
                    lines: contents.lines().count(),
                });
                // Delimited so the model can tell the quoted document from the
                // surrounding instruction.
                out.push_str(&format!(
                    "\n\n----- contents of {token} -----\n{}\n----- end of {token} -----\n\n",
                    contents.trim_end()
                ));
            }
            Err(_) => {
                if looks_like_a_path(token) {
                    result.unresolved.push(token.to_string());
                }
                out.push('@');
                out.push_str(token);
            }
        }
        rest = &after[token.len()..];
    }
    out.push_str(rest);

    result.text = out;
    Ok(result)
}

/// Would a reader take this token for a filename? Used only to decide whether
/// an unresolved reference is worth warning about.
///
/// A file extension is required rather than merely a slash, so an `@scope/pkg`
/// package mention does not produce a warning about a missing file. Missing a
/// real reference is the dangerous direction, but a warning on every `@`
/// mention would train the user to ignore warnings, which costs the same thing.
fn looks_like_a_path(token: &str) -> bool {
    match token.rsplit_once('.') {
        Some((stem, extension)) => {
            !stem.is_empty()
                && (1..=8).contains(&extension.len())
                && extension.chars().all(|c| c.is_ascii_alphanumeric())
        }
        None => false,
    }
}

/// Recognise a mistyped `--from-file <path>` entered at the prompt instead of on
/// the command line, which is an easy confusion to make and otherwise reaches
/// the model as prose.
pub fn parse_from_file_line(text: &str) -> Option<&str> {
    let line = text.trim();
    if line.lines().count() != 1 {
        return None;
    }
    let rest = line
        .strip_prefix("--from-file")
        .or_else(|| line.strip_prefix("-from-file"))
        .or_else(|| line.strip_prefix("from-file"))?;
    let path = rest.trim_start_matches([' ', '=', ':']).trim();
    (!path.is_empty()).then_some(path)
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
    fn a_reference_inside_a_sentence_is_read_in_not_passed_through() {
        // The live failure: "read @req.md and do" left the token literal, the
        // provider replied that it cannot open local files, and that reply was
        // published as the specification.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("req.md"), "# Goal\n\nBuild a ball game.\n").unwrap();

        let expanded = expand_file_references(dir.path(), "read @req.md and do it").unwrap();

        assert!(!expanded.text.contains("@req.md"), "{}", expanded.text);
        assert!(expanded.text.contains("Build a ball game."));
        assert!(expanded.text.starts_with("read "));
        assert!(expanded.text.trim_end().ends_with("and do it"));
        assert_eq!(expanded.loaded.len(), 1);
        assert_eq!(expanded.loaded[0].reference, "req.md");
        assert!(expanded.unresolved.is_empty());
    }

    #[test]
    fn a_lone_reference_still_loads_the_whole_description() {
        let dir = tempfile::tempdir().unwrap();
        let body = "# Goal\n\nParagraph one.\n\nParagraph two.\n";
        std::fs::write(dir.path().join("req.md"), body).unwrap();

        let expanded = expand_file_references(dir.path(), "@req.md\n").unwrap();
        assert!(expanded.text.contains("Paragraph two."));
        assert_eq!(expanded.loaded.len(), 1);
    }

    #[test]
    fn several_references_are_each_read_in() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "AAA").unwrap();
        std::fs::write(dir.path().join("b.md"), "BBB").unwrap();

        let expanded = expand_file_references(dir.path(), "combine @a.md with @b.md").unwrap();
        assert!(expanded.text.contains("AAA") && expanded.text.contains("BBB"));
        assert_eq!(expanded.loaded.len(), 2);
    }

    #[test]
    fn a_missing_file_is_reported_rather_than_silently_left_literal() {
        let dir = tempfile::tempdir().unwrap();

        let expanded = expand_file_references(dir.path(), "read @spec.md please").unwrap();
        assert_eq!(expanded.unresolved, vec!["spec.md".to_string()]);
        // Still passed through, so the user sees their own words, but the
        // warning above is what stops this from going unnoticed.
        assert!(expanded.text.contains("@spec.md"));
    }

    #[test]
    fn ordinary_at_signs_are_left_alone_and_not_reported() {
        let dir = tempfile::tempdir().unwrap();

        let expanded = expand_file_references(
            dir.path(),
            "email me at a@b.com, install @scope/pkg, ask @teammate",
        )
        .unwrap();

        assert_eq!(
            expanded.text,
            "email me at a@b.com, install @scope/pkg, ask @teammate"
        );
        assert!(expanded.loaded.is_empty());
        // None of these should produce a missing-file warning: the email has a
        // word character before the '@', and neither package nor person has a
        // file extension.
        assert!(expanded.unresolved.is_empty(), "{:?}", expanded.unresolved);
    }

    #[test]
    fn trailing_punctuation_is_not_part_of_the_filename() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("req.md"), "BODY").unwrap();

        for text in ["see @req.md.", "see @req.md,", "see (@req.md)"] {
            let expanded = expand_file_references(dir.path(), text).unwrap();
            assert!(expanded.text.contains("BODY"), "failed for {text}");
        }
    }

    #[test]
    fn expanded_references_share_the_overall_size_cap() {
        let dir = tempfile::tempdir().unwrap();
        let big = "a".repeat(MAX_DESCRIPTION_BYTES - 10);
        std::fs::write(dir.path().join("one.md"), &big).unwrap();
        std::fs::write(dir.path().join("two.md"), &big).unwrap();

        assert!(matches!(
            expand_file_references(dir.path(), "@one.md @two.md"),
            Err(SpecInputError::TooLarge { .. })
        ));
    }

    #[test]
    fn a_from_file_flag_typed_at_the_prompt_is_understood() {
        // Observed live: the operator typed "from-file req.md" into the prompt,
        // where it was prose and reached the model as such.
        assert_eq!(parse_from_file_line("from-file req.md"), Some("req.md"));
        assert_eq!(parse_from_file_line("--from-file req.md"), Some("req.md"));
        assert_eq!(parse_from_file_line("--from-file=req.md"), Some("req.md"));
        assert_eq!(
            parse_from_file_line("  from-file  docs/req.md  "),
            Some("docs/req.md")
        );
        assert_eq!(parse_from_file_line("from-file"), None);
        assert_eq!(parse_from_file_line("build a from-file feature"), None);
        // Only a lone line counts; a real description is not a flag.
        assert_eq!(parse_from_file_line("from-file req.md\nand more"), None);
    }
}
