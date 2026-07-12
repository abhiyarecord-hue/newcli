//! Deterministic schema & code language guard (TASK-11.1).
//!
//! `SchemaLangGuard` as a PreTool Hook. `assert_english_machine_surface` walks
//! JSON recursively:
//! 1. Keys must match `^[A-Za-z_][A-Za-z0-9_-]*$`
//! 2. Devanagari (U+0900–U+097F) rejected in keys and machine-surface field values
//! 3. Prose-valued fields (explanation, description, reasoning, summary) are exempt

use regex::Regex;
use serde_json::Value;

use crate::hooks::{Hook, HookPoint, HookVerdict};

/// Fields whose values are prose (Hinglish welcome).
const PROSE_FIELDS: &[&str] = &[
    "explanation",
    "description",
    "reasoning",
    "summary",
    "thought",
    "plan",
];

/// Fields whose values are machine surfaces (must be English/ASCII).
const MACHINE_FIELDS: &[&str] = &[
    "file_path",
    "path",
    "command",
    "cmd",
    "url",
    "name",
    "code",
    "language",
    "pattern",
];

pub struct SchemaLangGuard {
    key_regex: Regex,
}

impl SchemaLangGuard {
    pub fn new() -> Self {
        Self {
            key_regex: Regex::new(r"^[A-Za-z_][A-Za-z0-9_-]*$").unwrap(),
        }
    }

    /// Validate that all machine surfaces in the JSON value are English/ASCII.
    pub fn assert_english_machine_surface(&self, v: &Value) -> Result<(), String> {
        self.check_value(v, None)
    }

    fn check_value(&self, v: &Value, parent_key: Option<&str>) -> Result<(), String> {
        match v {
            Value::Object(map) => {
                for (key, val) in map {
                    // 1. Key must match pattern.
                    if !self.key_regex.is_match(key) {
                        return Err(format!("invalid key format: '{key}'"));
                    }
                    // 2. Key must not contain Devanagari.
                    if contains_devanagari(key) {
                        return Err(format!("Devanagari in key: '{key}'"));
                    }
                    // 3. Check value based on field name.
                    if is_prose_field(key) {
                        // Prose fields are exempt — skip value check.
                        continue;
                    }
                    self.check_value(val, Some(key))?;
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    self.check_value(item, parent_key)?;
                }
            }
            Value::String(s) => {
                // If parent key is a machine field, reject Devanagari.
                if let Some(key) = parent_key {
                    if is_machine_field(key) && contains_devanagari(s) {
                        return Err(format!(
                            "Devanagari in machine field '{key}': re-emit with English/ASCII"
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Default for SchemaLangGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Hook for SchemaLangGuard {
    fn point(&self) -> HookPoint {
        HookPoint::PreTool
    }

    fn evaluate(&self, _tool_name: &str, payload: &Value) -> HookVerdict {
        match self.assert_english_machine_surface(payload) {
            Ok(()) => HookVerdict::Allow,
            Err(reason) => HookVerdict::Deny { reason },
        }
    }
}

fn is_prose_field(key: &str) -> bool {
    PROSE_FIELDS.contains(&key.to_lowercase().as_str())
}

fn is_machine_field(key: &str) -> bool {
    MACHINE_FIELDS.contains(&key.to_lowercase().as_str())
}

/// Check if a string contains any Devanagari codepoint (U+0900–U+097F).
fn contains_devanagari(s: &str) -> bool {
    s.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn devanagari_in_file_path_denied() {
        let guard = SchemaLangGuard::new();
        let payload = json!({"file_path": "\u{0938}\u{094D}\u{0930}\u{094B}\u{0924}.rs"});
        let verdict = guard.evaluate("write_file", &payload);
        assert!(matches!(verdict, HookVerdict::Deny { .. }));
    }

    #[test]
    fn english_path_with_hinglish_explanation_allowed() {
        let guard = SchemaLangGuard::new();
        let payload = json!({
            "file_path": "src/main.rs",
            "explanation": "yeh function file padhta hai"
        });
        let verdict = guard.evaluate("write_file", &payload);
        assert!(matches!(verdict, HookVerdict::Allow));
    }

    #[test]
    fn devanagari_json_key_denied() {
        let guard = SchemaLangGuard::new();
        let payload = json!({"\u{0928}\u{093E}\u{092E}": "value"});
        let result = guard.assert_english_machine_surface(&payload);
        assert!(result.is_err());
    }

    #[test]
    fn valid_ascii_keys_allowed() {
        let guard = SchemaLangGuard::new();
        let payload = json!({"file_path": "lib.rs", "command": "cargo build"});
        let result = guard.assert_english_machine_surface(&payload);
        assert!(result.is_ok());
    }

    #[test]
    fn nested_devanagari_in_machine_field_denied() {
        let guard = SchemaLangGuard::new();
        let payload = json!({
            "commands": [{"cmd": "\u{0915}\u{094D}\u{0930}\u{092E}"}]
        });
        let result = guard.assert_english_machine_surface(&payload);
        assert!(result.is_err());
    }
}
