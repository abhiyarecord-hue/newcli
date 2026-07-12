//! Hook engine: deterministic policy layer. NO LLM calls, NO network, NO fs writes.
//!
//! `HookEngine::run` evaluates all hooks. First `Deny` wins; `Rewrite`s compose
//! left-to-right. A hook that errors is treated as `Deny` (fail-closed).

use std::sync::Arc;

use agent_types::{AgentError, Result};
use regex::Regex;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookPoint {
    PreTool,
    PostTool,
}

#[derive(Clone, Debug)]
pub enum HookVerdict {
    Allow,
    Deny { reason: String },
    Rewrite(Value),
}

pub trait Hook: Send + Sync {
    fn point(&self) -> HookPoint;
    fn evaluate(&self, tool_name: &str, payload: &Value) -> HookVerdict;
}

pub struct HookEngine {
    hooks: Vec<Arc<dyn Hook>>,
}

impl HookEngine {
    pub fn new(hooks: Vec<Arc<dyn Hook>>) -> Self {
        Self { hooks }
    }

    /// Run all hooks for the given point. First Deny wins, Rewrites compose.
    pub fn run(&self, point: HookPoint, tool_name: &str, payload: &Value) -> Result<HookVerdict> {
        let mut current_payload = payload.clone();

        for hook in &self.hooks {
            if hook.point() != point {
                continue;
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                hook.evaluate(tool_name, &current_payload)
            })) {
                Ok(HookVerdict::Allow) => {}
                Ok(HookVerdict::Deny { reason }) => {
                    return Ok(HookVerdict::Deny { reason });
                }
                Ok(HookVerdict::Rewrite(new_payload)) => {
                    current_payload = new_payload;
                }
                Err(_) => {
                    // Hook panicked → fail-closed.
                    return Ok(HookVerdict::Deny {
                        reason: "hook panicked (fail-closed)".into(),
                    });
                }
            }
        }

        if current_payload != *payload {
            Ok(HookVerdict::Rewrite(current_payload))
        } else {
            Ok(HookVerdict::Allow)
        }
    }
}

// === Built-in Hooks ===

/// Detects secrets in tool payloads (AWS keys, generic API keys, private key headers).
pub struct SecretLeakHook {
    patterns: Vec<(Regex, &'static str)>,
}

impl SecretLeakHook {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                (
                    Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
                    "AWS access key",
                ),
                (
                    Regex::new(r#"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['"][A-Za-z0-9_-]{16,}"#)
                        .unwrap(),
                    "generic secret/token",
                ),
                (
                    Regex::new(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----").unwrap(),
                    "private key",
                ),
            ],
        }
    }
}

impl Default for SecretLeakHook {
    fn default() -> Self {
        Self::new()
    }
}

impl Hook for SecretLeakHook {
    fn point(&self) -> HookPoint {
        HookPoint::PreTool
    }

    fn evaluate(&self, tool_name: &str, payload: &Value) -> HookVerdict {
        // Only scan bash/write_file tools.
        if !matches!(tool_name, "bash" | "write_file") {
            return HookVerdict::Allow;
        }

        // Extract all string values from the payload for scanning.
        let mut texts = Vec::new();
        collect_strings(payload, &mut texts);
        let combined = texts.join("\n");

        for (regex, class) in &self.patterns {
            if regex.is_match(&combined) {
                return HookVerdict::Deny {
                    reason: format!("secret detected: {class}"),
                };
            }
        }
        HookVerdict::Allow
    }
}

/// Recursively collect all string values from a JSON value.
fn collect_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(arr) => {
            for item in arr {
                collect_strings(item, out);
            }
        }
        Value::Object(map) => {
            for val in map.values() {
                collect_strings(val, out);
            }
        }
        _ => {}
    }
}

/// Denies destructive shell commands.
pub struct DestructiveCommandHook {
    patterns: Vec<Regex>,
}

impl DestructiveCommandHook {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                Regex::new(r"rm\s+-rf\s+/").unwrap(),
                Regex::new(r"mkfs").unwrap(),
                Regex::new(r"dd\s+.*of=/dev/").unwrap(),
                Regex::new(r"git\s+push\s+.*--force").unwrap(),
                Regex::new(r"git\s+push\s+-f\b").unwrap(),
            ],
        }
    }
}

impl Default for DestructiveCommandHook {
    fn default() -> Self {
        Self::new()
    }
}

impl Hook for DestructiveCommandHook {
    fn point(&self) -> HookPoint {
        HookPoint::PreTool
    }

    fn evaluate(&self, tool_name: &str, payload: &Value) -> HookVerdict {
        if tool_name != "bash" {
            return HookVerdict::Allow;
        }
        let command = payload
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("");
        for regex in &self.patterns {
            if regex.is_match(command) {
                return HookVerdict::Deny {
                    reason: format!("destructive command blocked: {}", regex.as_str()),
                };
            }
        }
        HookVerdict::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn secret_leak_detects_aws_key() {
        let hook = SecretLeakHook::new();
        let payload = Value::Object(serde_json::Map::from_iter([(
            "command".to_string(),
            Value::String("curl -H x-api-key:AKIAIOSFODNN7EXAMPLE".to_string()),
        )]));
        let verdict = hook.evaluate("bash", &payload);
        assert!(matches!(verdict, HookVerdict::Deny { ref reason } if reason.contains("AWS")));
    }

    #[test]
    fn secret_leak_detects_generic_token() {
        let hook = SecretLeakHook::new();
        let payload = Value::Object(serde_json::Map::from_iter([(
            "command".to_string(),
            Value::String("export api_key=\"abcdefghijklmnopqrstuv\"".to_string()),
        )]));
        let verdict = hook.evaluate("bash", &payload);
        assert!(matches!(verdict, HookVerdict::Deny { .. }));
    }

    #[test]
    fn secret_leak_allows_safe_command() {
        let hook = SecretLeakHook::new();
        let payload = json!({"command": "ls -la"});
        let verdict = hook.evaluate("bash", &payload);
        assert!(matches!(verdict, HookVerdict::Allow));
    }

    #[test]
    fn destructive_blocks_rm_rf_root() {
        let hook = DestructiveCommandHook::new();
        let payload = json!({"command": "rm -rf /"});
        let verdict = hook.evaluate("bash", &payload);
        assert!(matches!(verdict, HookVerdict::Deny { .. }));
    }

    #[test]
    fn destructive_blocks_force_push() {
        let hook = DestructiveCommandHook::new();
        let payload = json!({"command": "git push origin main --force"});
        let verdict = hook.evaluate("bash", &payload);
        assert!(matches!(verdict, HookVerdict::Deny { .. }));
    }

    #[test]
    fn destructive_allows_normal_commands() {
        let hook = DestructiveCommandHook::new();
        let payload = json!({"command": "cargo build"});
        let verdict = hook.evaluate("bash", &payload);
        assert!(matches!(verdict, HookVerdict::Allow));
    }

    #[test]
    fn engine_first_deny_wins() {
        let engine = HookEngine::new(vec![
            Arc::new(SecretLeakHook::new()),
            Arc::new(DestructiveCommandHook::new()),
        ]);
        let payload = Value::Object(serde_json::Map::from_iter([(
            "command".to_string(),
            Value::String("curl AKIAIOSFODNN7EXAMPLE".to_string()),
        )]));
        let verdict = engine.run(HookPoint::PreTool, "bash", &payload).unwrap();
        assert!(matches!(verdict, HookVerdict::Deny { ref reason } if reason.contains("AWS")));
    }

    #[test]
    fn engine_allows_when_all_pass() {
        let engine = HookEngine::new(vec![
            Arc::new(SecretLeakHook::new()),
            Arc::new(DestructiveCommandHook::new()),
        ]);
        let payload = json!({"command": "echo hello"});
        let verdict = engine.run(HookPoint::PreTool, "bash", &payload).unwrap();
        assert!(matches!(verdict, HookVerdict::Allow));
    }

    #[test]
    fn secret_never_echoed_in_deny_reason() {
        let hook = SecretLeakHook::new();
        let payload = Value::Object(serde_json::Map::from_iter([(
            "command".to_string(),
            Value::String("echo AKIAIOSFODNN7EXAMPLE".to_string()),
        )]));
        if let HookVerdict::Deny { reason } = hook.evaluate("bash", &payload) {
            assert!(!reason.contains("AKIAIOSFODNN7EXAMPLE"));
        }
    }
}
