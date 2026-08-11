//! `spec-pipeline` (L3): RustySpec 7-stage pipeline.
//!
//! Stages: Specify → Clarify → Plan → Tasks → Tests → Implement → Analyze.
//! Each stage builds a prompt from prior artifacts and produces a new one.

pub mod artifacts;
pub mod stages;

pub use artifacts::{Artifact, ArtifactKind, ArtifactSpec};
pub use stages::{
    Pipeline, PromptReport, Stage, CONTAINS_PREFIX, CONTAINS_SEPARATOR,
    DEFAULT_CONTEXT_BUDGET_BYTES, EXISTS_PREFIX, TRUNCATION_MARKER,
};
