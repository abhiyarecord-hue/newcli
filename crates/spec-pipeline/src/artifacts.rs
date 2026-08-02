//! Artifact model for the spec pipeline.

use std::path::PathBuf;

use crate::stages::Stage;

/// Filesystem shape of a stage artifact root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactKind {
    File,
    Directory,
}

/// Fixed paths and shape for one stage's artifact.
///
/// `root` is the compatibility path used to determine stage completion.
/// `primary` is the concrete file read and written by the pipeline. Directory
/// artifacts may contain additional files beside their primary file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactSpec {
    pub root: &'static str,
    pub primary: &'static str,
    pub kind: ArtifactKind,
}

#[derive(Clone, Debug)]
pub struct Artifact {
    pub stage: Stage,
    pub path: PathBuf,
    pub content: String,
}
