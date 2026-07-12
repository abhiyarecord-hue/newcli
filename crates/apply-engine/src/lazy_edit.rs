//! Lazy edit snippet: a snippet containing `... existing code ...` markers.
//! The markers signal to the apply engine which portions of the original are
//! preserved verbatim and where the new code should be inserted/replaced.

use agent_types::{AgentError, Result};
use regex::Regex;

/// A validated lazy-edit snippet.
#[derive(Clone, Debug)]
pub struct LazyEdit {
    pub snippet: String,
}

impl LazyEdit {
    /// Validate the snippet: it must contain at least 1 marker matching the regex
    /// `(?m)^\s*(//|#|--|/\*)\s*\.\.\.\s*existing code\s*\.\.\.`.
    /// Reject empty snippets.
    pub fn new(snippet: &str) -> Result<Self> {
        if snippet.trim().is_empty() {
            return Err(AgentError::Tool {
                name: "lazy_edit".into(),
                reason: "empty snippet".into(),
            });
        }

        let marker_re = Regex::new(
            r"(?m)^\s*(//|#|--|/\*)\s*\.\.\.\s*existing code\s*\.\.\."
        ).unwrap();

        if !marker_re.is_match(snippet) {
            return Err(AgentError::Tool {
                name: "lazy_edit".into(),
                reason: "snippet has no `... existing code ...` marker".into(),
            });
        }

        Ok(Self {
            snippet: snippet.to_string(),
        })
    }

    /// The marker regex pattern (compiled once, reused).
    pub fn marker_regex() -> Regex {
        Regex::new(
            r"(?m)^\s*(//|#|--|/\*)\s*\.\.\.\s*existing code\s*\.\.\."
        ).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_snippet_with_rust_marker() {
        let s = "fn new_code() {}\n// ... existing code ...\nfn end() {}";
        assert!(LazyEdit::new(s).is_ok());
    }

    #[test]
    fn valid_snippet_with_python_marker() {
        let s = "# ... existing code ...\ndef new(): pass";
        assert!(LazyEdit::new(s).is_ok());
    }

    #[test]
    fn empty_snippet_rejected() {
        assert!(LazyEdit::new("").is_err());
        assert!(LazyEdit::new("   \n\t  ").is_err());
    }

    #[test]
    fn no_marker_rejected() {
        assert!(LazyEdit::new("fn hello() {}").is_err());
    }
}
