//! `spec-pipeline` (L3): RustySpec 7-stage pipeline.
//!
//! Stages: Specify → Clarify → Plan → Tasks → Tests → Implement → Analyze.
//! Each stage builds a prompt from prior artifacts and produces a new one.

pub mod artifacts;
pub mod stages;

pub use artifacts::Artifact;
pub use stages::{Pipeline, Stage};
