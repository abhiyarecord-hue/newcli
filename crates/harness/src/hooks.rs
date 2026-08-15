//! Hook engine: deterministic policy layer. NO LLM calls, NO network, NO fs writes.
//!
//! `HookEngine::run` evaluates all hooks. First `Deny` wins; `Rewrite`s compose
//! left-to-right. A hook that errors is treated as `Deny` (fail-closed).

use std::sync::Arc;

use agent_types::{Result, ToolEffects};
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

    /// Effect-aware, error-capable hook entry point. Existing hooks remain source
    /// compatible; policy hooks override this method and dispatch supplies the
    /// authoritative effects declared by the tool schema.
    fn evaluate_with_effects(
        &self,
        tool_name: &str,
        payload: &Value,
        _effects: ToolEffects,
    ) -> Result<HookVerdict> {
        Ok(self.evaluate(tool_name, payload))
    }
}

pub struct HookEngine {
    hooks: Vec<Arc<dyn Hook>>,
}

impl HookEngine {
    pub fn new(hooks: Vec<Arc<dyn Hook>>) -> Self {
        Self { hooks }
    }

    /// Compatibility entry point for callers without a schema. Known legacy
    /// names and command-shaped payloads are mapped to conservative effects.
    pub fn run(&self, point: HookPoint, tool_name: &str, payload: &Value) -> Result<HookVerdict> {
        self.run_with_effects(
            point,
            tool_name,
            payload,
            inferred_effects(tool_name, payload),
        )
    }

    /// Run all hooks with authoritative tool effects. First Deny wins and
    /// rewrites compose. Hook errors and unwind panics are converted to a
    /// fail-closed denial before any later hook can run.
    pub fn run_with_effects(
        &self,
        point: HookPoint,
        tool_name: &str,
        payload: &Value,
        effects: ToolEffects,
    ) -> Result<HookVerdict> {
        let mut current_payload = payload.clone();

        for hook in &self.hooks {
            if hook.point() != point {
                continue;
            }
            let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                hook.evaluate_with_effects(tool_name, &current_payload, effects)
            }));
            match evaluated {
                Ok(Ok(HookVerdict::Allow)) => {}
                Ok(Ok(HookVerdict::Deny { reason })) => {
                    return Ok(HookVerdict::Deny { reason });
                }
                Ok(Ok(HookVerdict::Rewrite(new_payload))) => current_payload = new_payload,
                Ok(Err(error)) => {
                    return Ok(HookVerdict::Deny {
                        reason: format!("hook error (fail-closed): {error}"),
                    });
                }
                Err(_) => {
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

fn inferred_effects(tool_name: &str, payload: &Value) -> ToolEffects {
    let normalized_name = tool_name.to_ascii_lowercase();
    match normalized_name.as_str() {
        "bash" | "check_code" => ToolEffects::PROCESS_SPAWN
            .union(ToolEffects::WORKSPACE_WRITE)
            .union(ToolEffects::NETWORK_OUTBOUND)
            .union(ToolEffects::EXTERNAL_WRITE)
            .union(ToolEffects::CREDENTIAL_BEARING),
        "write_file" | "edit_file" => ToolEffects::WORKSPACE_WRITE,
        "web_fetch" | "dispatch_subagent" => ToolEffects::NETWORK_OUTBOUND,
        name if name.starts_with("mcp__") => ToolEffects::NETWORK_OUTBOUND
            .union(ToolEffects::EXTERNAL_WRITE)
            .union(ToolEffects::CREDENTIAL_BEARING),
        _ if has_command_field(payload) => ToolEffects::PROCESS_SPAWN.union(ToolEffects::UNKNOWN),
        _ => ToolEffects::UNKNOWN,
    }
}

fn has_command_field(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.iter().any(|(key, value)| {
            matches!(
                key.to_ascii_lowercase().as_str(),
                "command" | "cmd" | "script"
            ) && value.is_string()
        }),
        _ => false,
    }
}

// === Built-in Hooks ===

/// Detects secrets in every write/outbound payload, independent of tool name.
pub struct SecretLeakHook {
    patterns: Vec<(Regex, &'static str)>,
}

impl SecretLeakHook {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                (Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(), "AWS access key"),
                (
                    Regex::new(r#"(?i)\b(api[_-]?key|access[_-]?token|token|secret)\b\s*(?:[:=]|\s)\s*['\"]?[A-Za-z0-9_./+=-]{16,}"#).unwrap(),
                    "generic secret/token",
                ),
                (Regex::new(r"(?i)\bbearer\s+[A-Za-z0-9_./+=-]{16,}").unwrap(), "bearer token"),
                (
                    Regex::new(r"(?i)-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----").unwrap(),
                    "private key",
                ),
            ],
        }
    }

    fn evaluate_policy(&self, payload: &Value, effects: ToolEffects) -> HookVerdict {
        if !(effects.workspace_write
            || effects.network_outbound
            || effects.external_write
            || effects.credential_bearing
            || effects.unknown)
        {
            return HookVerdict::Allow;
        }

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
        self.evaluate_policy(payload, inferred_effects(tool_name, payload))
    }

    fn evaluate_with_effects(
        &self,
        _tool_name: &str,
        payload: &Value,
        effects: ToolEffects,
    ) -> Result<HookVerdict> {
        Ok(self.evaluate_policy(payload, effects))
    }
}

fn collect_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(arr) => arr.iter().for_each(|item| collect_strings(item, out)),
        Value::Object(map) => map.values().for_each(|value| collect_strings(value, out)),
        _ => {}
    }
}

/// Denies normalized destructive command effects across Unix, cmd.exe, and
/// PowerShell spellings. This is best-effort policy, not shell isolation.
pub struct DestructiveCommandHook {
    /// Patterns specific enough to be safe on any string, including prose.
    unambiguous: Vec<Regex>,
    /// Patterns whose keywords occur in ordinary text, so they are applied only
    /// where the payload is actually a command.
    command_context_only: Vec<Regex>,
}

impl DestructiveCommandHook {
    pub fn new() -> Self {
        Self {
            unambiguous: vec![
                Regex::new(r"\brm(?:\.exe)?\s+(?:-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\b")
                    .unwrap(),
                Regex::new(r"\b(?:mkfs(?:\.[a-z0-9]+)?|format-volume|clear-disk|diskpart)\b")
                    .unwrap(),
                Regex::new(r"\bdd\b.*\bof\s*=\s*/dev/").unwrap(),
                Regex::new(r"\bgit\s+push\b.*(?:--force(?:-with-lease)?|-f)\b").unwrap(),
                Regex::new(r"\bremove-item\b.*(?:-recurse|-force)\b").unwrap(),
                Regex::new(r"\b(?:rd|rmdir)(?:\.exe)?\s+.*(?:/s)\b").unwrap(),
            ],
            // `format` and `del`/`erase` are ordinary English words. Applying
            // them to every string of an unclassified payload would deny prose
            // like "format the code" with a misleading reason, so they need
            // evidence that the payload really is a command.
            command_context_only: vec![
                Regex::new(r"\bformat(?:\.com)?\b").unwrap(),
                Regex::new(r"\b(?:del|erase)(?:\.exe)?\s+(?:/[a-z]\s+)*\S+").unwrap(),
            ],
        }
    }

    fn evaluate_policy(&self, payload: &Value, effects: ToolEffects) -> HookVerdict {
        // Unknown effects are always scanned. Previously an unknown tool was
        // scanned only when its payload happened to carry a command-shaped
        // field, so renaming the field bypassed the scan entirely.
        if !(effects.process_spawn || effects.unknown) {
            return HookVerdict::Allow;
        }
        let command = normalize_command_payload(payload);

        // Keyword patterns need evidence that this payload really is a command.
        // A declared process spawner always qualifies; an unclassified payload
        // qualifies only when it carries a command-shaped field.
        let command_context = effects.process_spawn || has_command_field(payload);
        let applicable = self
            .unambiguous
            .iter()
            .chain(self.command_context_only.iter().filter(|_| command_context));

        for regex in applicable {
            if regex.is_match(&command) {
                return HookVerdict::Deny {
                    reason: format!("destructive command blocked: {}", regex.as_str()),
                };
            }
        }
        HookVerdict::Allow
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
        self.evaluate_policy(payload, inferred_effects(tool_name, payload))
    }

    fn evaluate_with_effects(
        &self,
        _tool_name: &str,
        payload: &Value,
        effects: ToolEffects,
    ) -> Result<HookVerdict> {
        Ok(self.evaluate_policy(payload, effects))
    }
}

fn normalize_command_payload(payload: &Value) -> String {
    let mut strings = Vec::new();
    collect_strings(payload, &mut strings);
    strings
        .join(" ")
        .to_ascii_lowercase()
        .replace(['\'', '\"', '`'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_types::AgentError;
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

    #[test]
    fn normalized_destructive_forms_are_denied_but_allowed_processes_remain() {
        // **Validates: Requirements 2.9, 3.9**
        let engine = HookEngine::new(vec![Arc::new(DestructiveCommandHook::new())]);
        let process_effects = ToolEffects::PROCESS_SPAWN;
        for command in [
            "RM -RF ./generated",
            "cmd.exe /C DEL /Q C:\\fixture\\canary",
            "powershell.exe -Command Remove-Item -Recurse C:\\fixture",
            "git PUSH origin main --FORCE",
        ] {
            let verdict = engine
                .run_with_effects(
                    HookPoint::PreTool,
                    "arbitrary_runner",
                    &json!({"command": command}),
                    process_effects,
                )
                .unwrap();
            assert!(
                matches!(verdict, HookVerdict::Deny { .. }),
                "allowed {command}"
            );
        }
        let allowed = engine
            .run_with_effects(
                HookPoint::PreTool,
                "arbitrary_runner",
                &json!({"command": "cargo test -p harness"}),
                process_effects,
            )
            .unwrap();
        assert!(matches!(allowed, HookVerdict::Allow));
    }

    #[test]
    fn secret_scanning_follows_write_and_outbound_effects_not_tool_names() {
        // **Validates: Requirements 2.9**
        let engine = HookEngine::new(vec![Arc::new(SecretLeakHook::new())]);
        for effects in [ToolEffects::WORKSPACE_WRITE, ToolEffects::NETWORK_OUTBOUND] {
            let verdict = engine
                .run_with_effects(
                    HookPoint::PreTool,
                    "custom_tool",
                    &json!({"nested": {"value": "api_key=abcdefghijklmnop"}}),
                    effects,
                )
                .unwrap();
            assert!(matches!(verdict, HookVerdict::Deny { .. }));
        }
        let read_only = engine
            .run_with_effects(
                HookPoint::PreTool,
                "custom_tool",
                &json!({"query": "api_key=abcdefghijklmnop"}),
                ToolEffects::WORKSPACE_READ,
            )
            .unwrap();
        assert!(matches!(read_only, HookVerdict::Allow));
    }

    struct ErrorHook;

    impl Hook for ErrorHook {
        fn point(&self) -> HookPoint {
            HookPoint::PostTool
        }

        fn evaluate(&self, _tool_name: &str, _payload: &Value) -> HookVerdict {
            HookVerdict::Allow
        }

        fn evaluate_with_effects(
            &self,
            _tool_name: &str,
            _payload: &Value,
            _effects: ToolEffects,
        ) -> Result<HookVerdict> {
            Err(AgentError::Tool {
                name: "error_hook".into(),
                reason: "fixture error".into(),
            })
        }
    }

    #[test]
    fn hook_errors_fail_closed_and_stop_later_hooks() {
        // **Validates: Requirements 2.21, 3.13**
        let engine = HookEngine::new(vec![Arc::new(ErrorHook)]);
        let verdict = engine
            .run_with_effects(
                HookPoint::PostTool,
                "echo",
                &json!({"status": "success"}),
                ToolEffects::NONE,
            )
            .unwrap();
        assert!(matches!(
            verdict,
            HookVerdict::Deny { ref reason } if reason.contains("hook error") && reason.contains("fail-closed")
        ));
    }
}
