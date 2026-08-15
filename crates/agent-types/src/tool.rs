//! Tool abstractions and auditable execution-boundary types.

use std::sync::Arc;

use crate::error::Result;

/// Effects a tool may perform. Missing declarations deliberately deserialize
/// to [`ToolEffects::UNKNOWN`] so old or future schemas fail closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolEffects {
    pub workspace_read: bool,
    pub workspace_write: bool,
    pub process_spawn: bool,
    pub network_outbound: bool,
    pub external_write: bool,
    pub credential_bearing: bool,
    pub unknown: bool,
}

impl ToolEffects {
    pub const UNKNOWN: Self = Self {
        workspace_read: false,
        workspace_write: false,
        process_spawn: false,
        network_outbound: false,
        external_write: false,
        credential_bearing: false,
        unknown: true,
    };
    pub const NONE: Self = Self {
        unknown: false,
        ..Self::UNKNOWN
    };
    pub const WORKSPACE_READ: Self = Self {
        workspace_read: true,
        ..Self::NONE
    };
    pub const WORKSPACE_WRITE: Self = Self {
        workspace_write: true,
        ..Self::NONE
    };
    pub const PROCESS_SPAWN: Self = Self {
        process_spawn: true,
        ..Self::NONE
    };
    pub const NETWORK_OUTBOUND: Self = Self {
        network_outbound: true,
        ..Self::NONE
    };
    pub const EXTERNAL_WRITE: Self = Self {
        external_write: true,
        ..Self::NONE
    };
    pub const CREDENTIAL_BEARING: Self = Self {
        credential_bearing: true,
        ..Self::NONE
    };

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            workspace_read: self.workspace_read || other.workspace_read,
            workspace_write: self.workspace_write || other.workspace_write,
            process_spawn: self.process_spawn || other.process_spawn,
            network_outbound: self.network_outbound || other.network_outbound,
            external_write: self.external_write || other.external_write,
            credential_bearing: self.credential_bearing || other.credential_bearing,
            unknown: self.unknown || other.unknown,
        }
    }

    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.unknown
    }
}

impl Default for ToolEffects {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub effects: ToolEffects,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionStatus {
    Success,
    /// Conservative default: an unreported outcome is treated as a failure.
    #[default]
    Error,
    Denied,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolExecutionTiming {
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: u64,
}

impl ToolExecutionTiming {
    #[must_use]
    pub const fn duration_ms(self) -> u64 {
        self.finished_at_unix_ms
            .saturating_sub(self.started_at_unix_ms)
    }
}

/// Marker that prefixes every disclosure of truncated or unfinished output.
///
/// Consumers detect incompleteness by searching for this exact substring, so
/// every surface that can return partial output must use it verbatim rather than
/// spelling the word differently.
pub const INCOMPLETE_OUTPUT_MARKER: &str = "[incomplete]";

/// Complete post-execution evidence supplied to policy hooks and audit sinks.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolExecutionRecord {
    pub requested_input: serde_json::Value,
    pub effective_input: serde_json::Value,
    #[serde(default)]
    pub status: ToolExecutionStatus,
    pub output: Option<String>,
    pub error: Option<String>,
    pub timing: ToolExecutionTiming,
    #[serde(default)]
    pub effects: ToolEffects,
    /// Turn that requested this execution.
    ///
    /// Attribution matters because a stage may only count work performed by the
    /// current run; a mutation carried over from an earlier turn or an earlier
    /// session must never be credited to this one. `0` means "not attributed".
    #[serde(default)]
    pub turn_id: u64,
    /// Whether this execution actually committed a workspace mutation.
    ///
    /// True only when a workspace-writing tool completed successfully, so a
    /// denied, failed, cancelled, or read-only call can never be mistaken for
    /// real work.
    #[serde(default)]
    pub committed_mutation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalKind {
    ToolExecution,
    WorkspaceMcpStartup,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ApprovalRequest {
    pub id: String,
    pub kind: ApprovalKind,
    pub prompt: String,
    pub tool_name: Option<String>,
    pub requested_input: serde_json::Value,
    #[serde(default)]
    pub effects: ToolEffects,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approved,
    Denied { reason: String },
}

#[async_trait::async_trait]
pub trait ApprovalProvider: Send + Sync {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalDecision>;
}

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn schema(&self) -> ToolSchema;
    async fn invoke(&self, input: serde_json::Value, ctx: &ToolCtx) -> Result<String>;
}

/// Everything a tool may touch. NO global state anywhere in the system.
pub struct ToolCtx {
    pub project_root: std::path::PathBuf,
    pub cancel: tokio_util::sync::CancellationToken,
    /// `None` is a safe default: callers requiring approval must deny rather
    /// than reading process-global input or silently approving.
    pub approval_provider: Option<Arc<dyn ApprovalProvider>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_schema_defaults_to_unknown_effects() {
        // **Validates: Requirements 2.9, 3.13**
        let schema: ToolSchema = serde_json::from_value(serde_json::json!({
            "name": "legacy",
            "description": "old persisted schema",
            "input_schema": {"type": "object"}
        }))
        .unwrap();
        assert_eq!(schema.effects, ToolEffects::UNKNOWN);
    }

    #[test]
    fn execution_record_round_trips_complete_evidence() {
        // **Validates: Requirements 2.21**
        let record = ToolExecutionRecord {
            requested_input: serde_json::json!({"command": "requested"}),
            effective_input: serde_json::json!({"command": "effective"}),
            status: ToolExecutionStatus::Error,
            output: None,
            error: Some("failed".into()),
            timing: ToolExecutionTiming {
                started_at_unix_ms: 10,
                finished_at_unix_ms: 25,
            },
            effects: ToolEffects::PROCESS_SPAWN,
            turn_id: 4,
            committed_mutation: false,
        };
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["status"], "error");
        assert_eq!(json["turn_id"], 4);
        assert_eq!(json["committed_mutation"], false);
        assert_eq!(
            serde_json::from_value::<ToolExecutionRecord>(json).unwrap(),
            record
        );
        assert_eq!(record.timing.duration_ms(), 15);
    }

    #[test]
    fn execution_record_defaults_attribution_for_legacy_payloads() {
        // **Validates: Requirements 2.21, 2.41**
        // A record written before attribution existed must still load, and must
        // default to "not attributed, nothing committed" rather than crediting
        // work to the current run.
        let legacy: ToolExecutionRecord = serde_json::from_value(serde_json::json!({
            "requested_input": {"path": "a.rs"},
            "effective_input": {"path": "a.rs"},
            "status": "success",
            "output": "ok",
            "error": null,
            "timing": {"started_at_unix_ms": 1, "finished_at_unix_ms": 2}
        }))
        .unwrap();
        assert_eq!(legacy.turn_id, 0);
        assert!(!legacy.committed_mutation);
    }

    #[test]
    fn unknown_and_omitted_effects_fail_closed() {
        // **Validates: Requirements 2.9, 2.21, 2.26**
        let partial: ToolEffects = serde_json::from_value(serde_json::json!({
            "workspace_read": true
        }))
        .unwrap();
        assert!(partial.workspace_read);
        assert!(partial.is_unknown());

        let unknown_field = serde_json::from_value::<ToolEffects>(serde_json::json!({
            "unknown": false,
            "future_irreversible_effect": true
        }));
        assert!(unknown_field.is_err());

        let approval: ApprovalRequest = serde_json::from_value(serde_json::json!({
            "id": "approval-1",
            "kind": "workspace_mcp_startup",
            "prompt": "Start workspace MCP server?",
            "tool_name": null,
            "requested_input": {"command": "server"}
        }))
        .unwrap();
        assert_eq!(approval.effects, ToolEffects::UNKNOWN);

        let conservative_record: ToolExecutionRecord = serde_json::from_value(serde_json::json!({
            "requested_input": {},
            "effective_input": {},
            "output": null,
            "error": null,
            "timing": {
                "started_at_unix_ms": 1,
                "finished_at_unix_ms": 1
            }
        }))
        .unwrap();
        assert_eq!(conservative_record.status, ToolExecutionStatus::Error);
        assert_eq!(conservative_record.effects, ToolEffects::UNKNOWN);
        assert_eq!(ToolExecutionStatus::default(), ToolExecutionStatus::Error);
    }

    #[test]
    fn approval_decisions_have_stable_tagged_serde() {
        // **Validates: Requirements 2.26, 2.27**
        let denied = ApprovalDecision::Denied {
            reason: "untrusted workspace".into(),
        };
        let json = serde_json::to_value(&denied).unwrap();
        assert_eq!(json["decision"], "denied");
        assert_eq!(json["reason"], "untrusted workspace");
        assert_eq!(
            serde_json::from_value::<ApprovalDecision>(json).unwrap(),
            denied
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(16))]

        #[test]
        fn property_effect_union_never_drops_a_declared_or_unknown_effect(
            left in proptest::array::uniform7(proptest::bool::ANY),
            right in proptest::array::uniform7(proptest::bool::ANY),
        ) {
            // **Validates: Requirements 2.9**
            let from = |bits: [bool; 7]| ToolEffects {
                workspace_read: bits[0],
                workspace_write: bits[1],
                process_spawn: bits[2],
                network_outbound: bits[3],
                external_write: bits[4],
                credential_bearing: bits[5],
                unknown: bits[6],
            };
            let combined = from(left).union(from(right));
            let actual = [
                combined.workspace_read,
                combined.workspace_write,
                combined.process_spawn,
                combined.network_outbound,
                combined.external_write,
                combined.credential_bearing,
                combined.unknown,
            ];
            for index in 0..7 {
                proptest::prop_assert_eq!(actual[index], left[index] || right[index]);
            }
        }
    }
}
