//! Artifact model for the spec pipeline.

use std::path::PathBuf;

use crate::stages::Stage;

#[derive(Clone, Debug)]
pub struct Artifact {
    pub stage: Stage,
    pub path: PathBuf,
    pub content: String,
}
