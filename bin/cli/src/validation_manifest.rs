//! Deterministic, offline validator for the release-evidence manifest.
//!
//! Readiness is derived **only** from recorded evidence. The validator is
//! fail-closed: anything it cannot positively confirm as a passing mandatory
//! gate produces `not_ready` and a nonzero exit. It performs no network access
//! and no platform-specific work, so it returns the same verdict on every
//! supported OS for the same manifest bytes.
//!
//! The rule it enforces is deliberately narrow:
//!
//! 1. Every entry marked `mandatory` must have status `passed`.
//! 2. Each mandatory gate class must be represented by at least one mandatory
//!    entry with status `passed`. A class that is simply absent is a missing
//!    gate, not a pass.
//! 3. A `not_run` status is tolerated **only** for a non-mandatory entry that
//!    explicitly sets `disclosed: true`.
//! 4. An unknown status string, an unknown gate class, or a malformed entry is
//!    a failure rather than something to skip.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Manifest schema version this validator understands.
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Gate classes that must each be present and passing for `ready`.
///
/// This list is the contract shared with `bin/cli/tests/task6_bug_conditions.rs`
/// and must stay in the same order as the manifest documentation.
pub const MANDATORY_GATE_CLASSES: [&str; 5] = [
    "requirement",
    "property",
    "platform",
    "security",
    "failure_path",
];

/// Non-mandatory classes the schema recognizes. Listed explicitly so a typo in
/// a class name is rejected instead of silently ignored.
pub const OPTIONAL_GATE_CLASSES: [&str; 3] = ["external_live", "workspace_command", "migration"];

/// Recorded outcome of one evidence entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Passed,
    Failed,
    /// The check exists but was not executed in this environment.
    NotRun,
    /// No evidence was produced at all.
    Missing,
    /// Evidence exists but its result could not be determined.
    Unknown,
}

impl EvidenceStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::NotRun => "not_run",
            Self::Missing => "missing",
            Self::Unknown => "unknown",
        }
    }
}

/// One recorded gate.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub id: String,
    pub class: String,
    pub mandatory: bool,
    pub status: EvidenceStatus,
    /// Required to be `true` for a non-mandatory `not_run` entry.
    #[serde(default)]
    pub disclosed: bool,
    /// Requirement or property identifiers this gate covers.
    #[serde(default)]
    pub covers: Vec<String>,
    /// Exact command that produced the result, when one exists.
    #[serde(default)]
    pub command: Option<String>,
    /// Observed process exit code, when one exists.
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub detail: Option<String>,
    /// Log artifact or evidence document reference.
    #[serde(default)]
    pub log_reference: Option<String>,
}

/// Provenance of the recorded run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub commit: String,
    pub branch: String,
    pub host_platform: String,
    pub toolchain: String,
    pub recorded_at: String,
}

/// The manifest itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub provenance: Provenance,
    pub evidence: Vec<Evidence>,
}

/// Derived readiness. There is no third state: absence of proof is `NotReady`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    Ready,
    NotReady,
}

impl Readiness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NotReady => "not_ready",
        }
    }
}

/// Validation outcome: the derived readiness plus every blocking reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationOutcome {
    pub readiness: Readiness,
    pub reasons: Vec<String>,
}

impl ValidationOutcome {
    pub fn is_ready(&self) -> bool {
        self.readiness == Readiness::Ready
    }

    /// Human-readable report. Reasons are already sorted deterministically.
    pub fn render(&self) -> String {
        let mut out = format!("readiness: {}\n", self.readiness.as_str());
        if self.reasons.is_empty() {
            out.push_str("blocking reasons: none\n");
            return out;
        }
        let _ = writeln!(out, "blocking reasons: {}", self.reasons.len());
        for reason in &self.reasons {
            let _ = writeln!(out, "  - {reason}");
        }
        out
    }
}

/// Parse failures are themselves a `not_ready` result, never a skipped check.
///
/// A leading UTF-8 BOM is tolerated because Windows editors and
/// `Set-Content -Encoding UTF8` both emit one, and a BOM is not valid leading
/// JSON. Tolerating it affects encoding only, never the completion rule.
pub fn validate_bytes(bytes: &[u8]) -> ValidationOutcome {
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    match serde_json::from_slice::<Manifest>(bytes) {
        Ok(manifest) => validate(&manifest),
        Err(error) => ValidationOutcome {
            readiness: Readiness::NotReady,
            reasons: vec![format!(
                "manifest does not parse against the schema: {error}"
            )],
        },
    }
}

pub fn validate_file(path: &Path) -> ValidationOutcome {
    match std::fs::read(path) {
        Ok(bytes) => validate_bytes(&bytes),
        Err(error) => ValidationOutcome {
            readiness: Readiness::NotReady,
            reasons: vec![format!(
                "cannot read manifest at {}: {error}",
                path.display()
            )],
        },
    }
}

/// Apply the completion rule. Every blocking condition is reported, not just
/// the first, so one run tells the operator everything that is missing.
pub fn validate(manifest: &Manifest) -> ValidationOutcome {
    let mut reasons = Vec::new();

    if manifest.schema_version != SUPPORTED_SCHEMA_VERSION {
        reasons.push(format!(
            "unsupported schema_version {}; this validator understands {SUPPORTED_SCHEMA_VERSION}",
            manifest.schema_version
        ));
    }

    for field in [
        ("commit", &manifest.provenance.commit),
        ("branch", &manifest.provenance.branch),
        ("host_platform", &manifest.provenance.host_platform),
        ("toolchain", &manifest.provenance.toolchain),
        ("recorded_at", &manifest.provenance.recorded_at),
    ] {
        if field.1.trim().is_empty() {
            reasons.push(format!("provenance.{} must not be empty", field.0));
        }
    }

    if manifest.evidence.is_empty() {
        reasons.push("manifest records no evidence at all".to_string());
    }

    let known: BTreeSet<&str> = MANDATORY_GATE_CLASSES
        .iter()
        .chain(OPTIONAL_GATE_CLASSES.iter())
        .copied()
        .collect();

    let mut seen_ids = BTreeSet::new();
    for entry in &manifest.evidence {
        if entry.id.trim().is_empty() {
            reasons.push("an evidence entry has an empty id".to_string());
        } else if !seen_ids.insert(entry.id.as_str()) {
            reasons.push(format!("duplicate evidence id '{}'", entry.id));
        }

        if !known.contains(entry.class.as_str()) {
            reasons.push(format!(
                "evidence '{}' has unknown class '{}'",
                entry.id, entry.class
            ));
        }

        if entry.mandatory && entry.status != EvidenceStatus::Passed {
            reasons.push(format!(
                "mandatory evidence '{}' (class '{}') is '{}', not 'passed'",
                entry.id,
                entry.class,
                entry.status.as_str()
            ));
        }

        if entry.status == EvidenceStatus::NotRun && !entry.mandatory && !entry.disclosed {
            reasons.push(format!(
                "optional evidence '{}' is 'not_run' without disclosure",
                entry.id
            ));
        }
    }

    for gate in MANDATORY_GATE_CLASSES {
        let satisfied = manifest.evidence.iter().any(|entry| {
            entry.class == gate && entry.mandatory && entry.status == EvidenceStatus::Passed
        });
        if !satisfied {
            reasons.push(format!(
                "mandatory gate class '{gate}' has no passing mandatory evidence"
            ));
        }
    }

    reasons.sort();
    reasons.dedup();
    let readiness = if reasons.is_empty() {
        Readiness::Ready
    } else {
        Readiness::NotReady
    };
    ValidationOutcome { readiness, reasons }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn entry(
        id: &str,
        class: &str,
        mandatory: bool,
        status: &str,
        disclosed: bool,
    ) -> serde_json::Value {
        json!({
            "id": id,
            "class": class,
            "mandatory": mandatory,
            "status": status,
            "disclosed": disclosed
        })
    }

    fn manifest_json(evidence: Vec<serde_json::Value>) -> serde_json::Value {
        json!({
            "schema_version": 1,
            "provenance": {
                "commit": "0123456789abcdef",
                "branch": "main",
                "host_platform": "x86_64-pc-windows-msvc",
                "toolchain": "rustc 1.91.1",
                "recorded_at": "2026-07-24"
            },
            "evidence": evidence
        })
    }

    /// All five mandatory gates passing, plus a disclosed optional `not_run`.
    fn fully_passing(bad_gate: Option<usize>, bad_status: &str) -> serde_json::Value {
        let mut evidence: Vec<_> = MANDATORY_GATE_CLASSES
            .iter()
            .enumerate()
            .map(|(index, class)| {
                let status = if Some(index) == bad_gate {
                    bad_status
                } else {
                    "passed"
                };
                entry(&format!("{class}-gate"), class, true, status, false)
            })
            .collect();
        evidence.push(entry(
            "live-provider-check",
            "external_live",
            false,
            "not_run",
            true,
        ));
        manifest_json(evidence)
    }

    fn outcome(value: &serde_json::Value) -> ValidationOutcome {
        validate_bytes(&serde_json::to_vec(value).unwrap())
    }

    #[test]
    fn a_complete_manifest_is_ready() {
        let result = outcome(&fully_passing(None, "passed"));
        assert!(
            result.is_ready(),
            "expected ready, blocked by: {:?}",
            result.reasons
        );
    }

    #[test]
    fn each_invalid_mandatory_status_blocks_readiness() {
        for (gate, class) in MANDATORY_GATE_CLASSES.iter().enumerate() {
            for status in ["failed", "not_run", "missing", "unknown"] {
                let result = outcome(&fully_passing(Some(gate), status));
                assert!(
                    !result.is_ready(),
                    "class {class} status {status} must not be ready"
                );
                assert!(
                    result.reasons.iter().any(|reason| reason.contains(status)),
                    "the blocking reason must name the offending status {status}: {:?}",
                    result.reasons
                );
            }
        }
    }

    #[test]
    fn an_absent_gate_class_is_a_missing_gate_not_a_pass() {
        for omitted in MANDATORY_GATE_CLASSES {
            let evidence: Vec<_> = MANDATORY_GATE_CLASSES
                .iter()
                .filter(|class| **class != omitted)
                .map(|class| entry(&format!("{class}-gate"), class, true, "passed", false))
                .collect();
            let result = outcome(&manifest_json(evidence));
            assert!(!result.is_ready(), "omitting {omitted} must not be ready");
            assert!(result
                .reasons
                .iter()
                .any(|reason| reason.contains(omitted) && reason.contains("no passing mandatory")));
        }
    }

    #[test]
    fn optional_not_run_must_be_disclosed() {
        let mut evidence: Vec<_> = MANDATORY_GATE_CLASSES
            .iter()
            .map(|class| entry(&format!("{class}-gate"), class, true, "passed", false))
            .collect();
        evidence.push(entry("live", "external_live", false, "not_run", false));
        let result = outcome(&manifest_json(evidence));
        assert!(!result.is_ready());
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("without disclosure")));
    }

    #[test]
    fn unknown_class_and_unknown_status_are_rejected() {
        let mut evidence: Vec<_> = MANDATORY_GATE_CLASSES
            .iter()
            .map(|class| entry(&format!("{class}-gate"), class, true, "passed", false))
            .collect();
        evidence.push(entry("typo", "requirememt", false, "passed", false));
        let result = outcome(&manifest_json(evidence));
        assert!(!result.is_ready());
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("unknown class")));

        // An unrecognized status string must fail the parse, not be skipped.
        let mut evidence: Vec<_> = MANDATORY_GATE_CLASSES
            .iter()
            .map(|class| entry(&format!("{class}-gate"), class, true, "passed", false))
            .collect();
        evidence.push(entry(
            "weird",
            "external_live",
            false,
            "probably_fine",
            true,
        ));
        let result = outcome(&manifest_json(evidence));
        assert!(!result.is_ready());
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("does not parse")));
    }

    #[test]
    fn unknown_fields_and_wrong_schema_version_are_rejected() {
        let mut value = fully_passing(None, "passed");
        value["evidence"][0]["surprise"] = json!(true);
        assert!(!outcome(&value).is_ready());

        let mut value = fully_passing(None, "passed");
        value["schema_version"] = json!(2);
        let result = outcome(&value);
        assert!(!result.is_ready());
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("unsupported schema_version")));
    }

    #[test]
    fn duplicate_ids_and_empty_provenance_are_rejected() {
        let mut evidence: Vec<_> = MANDATORY_GATE_CLASSES
            .iter()
            .map(|class| entry(&format!("{class}-gate"), class, true, "passed", false))
            .collect();
        evidence.push(entry(
            "requirement-gate",
            "requirement",
            true,
            "passed",
            false,
        ));
        let result = outcome(&manifest_json(evidence));
        assert!(!result.is_ready());
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("duplicate evidence id")));

        let mut value = fully_passing(None, "passed");
        value["provenance"]["commit"] = json!("   ");
        let result = outcome(&value);
        assert!(!result.is_ready());
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.contains("provenance.commit")));
    }

    #[test]
    fn a_leading_utf8_bom_does_not_change_the_verdict() {
        let ready = serde_json::to_vec(&fully_passing(None, "passed")).unwrap();
        let mut with_bom = vec![0xEF, 0xBB, 0xBF];
        with_bom.extend_from_slice(&ready);
        assert!(validate_bytes(&with_bom).is_ready());
        assert_eq!(validate_bytes(&with_bom), validate_bytes(&ready));

        // Tolerating the BOM must not tolerate a failing gate.
        let blocked = serde_json::to_vec(&fully_passing(Some(2), "not_run")).unwrap();
        let mut with_bom = vec![0xEF, 0xBB, 0xBF];
        with_bom.extend_from_slice(&blocked);
        assert!(!validate_bytes(&with_bom).is_ready());
    }

    #[test]
    fn malformed_bytes_are_not_ready() {
        assert!(!validate_bytes(b"").is_ready());
        assert!(!validate_bytes(b"{").is_ready());
        assert!(!validate_bytes(b"[]").is_ready());
    }

    #[test]
    fn validation_is_deterministic_for_identical_bytes() {
        let bytes = serde_json::to_vec(&fully_passing(Some(2), "not_run")).unwrap();
        let first = validate_bytes(&bytes);
        let second = validate_bytes(&bytes);
        assert_eq!(first, second);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Any invalid status on any mandatory gate must block readiness, and the
        /// reported reason must name that gate. This is the generated-manifest
        /// property required by task 30.1.
        #[test]
        fn property_generated_invalid_mandatory_gate_never_yields_ready(
            gate in 0usize..MANDATORY_GATE_CLASSES.len(),
            status in prop::sample::select(vec!["failed", "not_run", "missing", "unknown"]),
        ) {
            let result = outcome(&fully_passing(Some(gate), status));
            prop_assert!(!result.is_ready());
            let class = MANDATORY_GATE_CLASSES[gate];
            prop_assert!(
                result.reasons.iter().any(|reason| reason.contains(class)),
                "reason must name class {class}: {:?}",
                result.reasons
            );
        }

        /// Dropping an arbitrary subset of mandatory gates can never yield ready.
        #[test]
        fn property_any_missing_subset_of_gates_never_yields_ready(
            keep in proptest::collection::vec(any::<bool>(), MANDATORY_GATE_CLASSES.len())
        ) {
            let evidence: Vec<_> = MANDATORY_GATE_CLASSES
                .iter()
                .zip(keep.iter())
                .filter(|(_, keep)| **keep)
                .map(|(class, _)| entry(&format!("{class}-gate"), class, true, "passed", false))
                .collect();
            let complete = evidence.len() == MANDATORY_GATE_CLASSES.len();
            let result = outcome(&manifest_json(evidence));
            prop_assert_eq!(result.is_ready(), complete);
        }
    }
}
