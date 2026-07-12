//! RustySpec 7-stage pipeline stages and orchestration.

use std::path::{Path, PathBuf};

use agent_types::{AgentError, Result};
use sandbox::PathJail;

use crate::artifacts::Artifact;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Specify,
    Clarify,
    Plan,
    Tasks,
    Tests,
    Implement,
    Analyze,
}

impl Stage {
    pub fn artifact(&self) -> &'static str {
        match self {
            Stage::Specify => "spec.md",
            Stage::Clarify => "clarifications.md",
            Stage::Plan => "plan.md",
            Stage::Tasks => "tasks.md",
            Stage::Tests => "tests/",
            Stage::Implement => "code/",
            Stage::Analyze => "analysis.md",
        }
    }

    pub fn prerequisites(&self) -> &'static [Stage] {
        match self {
            Stage::Specify => &[],
            Stage::Clarify => &[Stage::Specify],
            Stage::Plan => &[Stage::Specify],
            Stage::Tasks => &[Stage::Plan],
            Stage::Tests => &[Stage::Tasks],
            Stage::Implement => &[Stage::Tasks],
            Stage::Analyze => &[Stage::Implement],
        }
    }

    pub fn all() -> &'static [Stage] {
        &[
            Stage::Specify,
            Stage::Clarify,
            Stage::Plan,
            Stage::Tasks,
            Stage::Tests,
            Stage::Implement,
            Stage::Analyze,
        ]
    }
}

pub struct Pipeline {
    session_dir: PathBuf,
    jail: PathJail,
}

impl Pipeline {
    /// Create a new pipeline for a session. All artifacts live under
    /// `.agent/specs/<session>/`.
    pub fn new(project_root: &Path, session_id: &str) -> Result<Self> {
        let jail = PathJail::new(project_root)?;
        let session_dir = project_root
            .join(".agent")
            .join("specs")
            .join(session_id);
        Ok(Self { session_dir, jail })
    }

    /// Check that all prerequisites for a stage are satisfied (artifacts exist on disk).
    pub fn check_prerequisites(&self, stage: Stage) -> Result<()> {
        for prereq in stage.prerequisites() {
            let artifact_path = self.session_dir.join(prereq.artifact());
            if !artifact_path.exists() {
                return Err(AgentError::Tool {
                    name: "spec_pipeline".into(),
                    reason: format!(
                        "prerequisite '{}' missing for stage {:?} (expected at {})",
                        prereq.artifact(),
                        stage,
                        artifact_path.display()
                    ),
                });
            }
        }
        Ok(())
    }

    /// Build the prompt for a stage by embedding prior artifacts.
    pub fn build_prompt(&self, stage: Stage, user_context: &str) -> Result<String> {
        self.check_prerequisites(stage)?;

        let mut prompt = format!("## Stage: {:?}\n\n", stage);
        prompt.push_str(user_context);
        prompt.push('\n');

        // Embed prior artifacts.
        for prereq in stage.prerequisites() {
            let path = self.session_dir.join(prereq.artifact());
            if path.is_file() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    prompt.push_str(&format!(
                        "\n---\n### Prior Artifact: {}\n{}\n",
                        prereq.artifact(),
                        content
                    ));
                }
            }
        }

        // Add stage-specific instructions.
        prompt.push_str(&stage_instructions(stage));
        Ok(prompt)
    }

    /// Write a stage artifact atomically. Validates required headers for spec.md.
    pub async fn write_artifact(&self, stage: Stage, content: &str) -> Result<PathBuf> {
        // Validate structure for Specify stage.
        if stage == Stage::Specify {
            validate_spec_headers(content)?;
        }

        let artifact_path = self.session_dir.join(stage.artifact());
        if let Some(parent) = artifact_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Atomic write via temp file.
        let tmp = artifact_path.with_extension("tmp");
        tokio::fs::write(&tmp, content).await?;
        tokio::fs::rename(&tmp, &artifact_path).await?;

        Ok(artifact_path)
    }

    /// Load an existing artifact.
    pub fn load_artifact(&self, stage: Stage) -> Result<Artifact> {
        let path = self.session_dir.join(stage.artifact());
        if !path.exists() {
            return Err(AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!("artifact '{}' not found", stage.artifact()),
            });
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!("read artifact: {e}"),
            })?;
        Ok(Artifact {
            stage,
            path,
            content,
        })
    }

    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    pub fn jail(&self) -> &PathJail {
        &self.jail
    }
}

fn stage_instructions(stage: Stage) -> String {
    match stage {
        Stage::Specify => "\n\nProduce a specification with the following headers:\n\
            ## User Stories\n## Functional Requirements\n## Non-Functional Requirements\n"
            .to_string(),
        Stage::Clarify => "\n\nList any ambiguities or questions about the spec.\n".to_string(),
        Stage::Plan => "\n\nProduce a technical plan with architecture decisions.\n".to_string(),
        Stage::Tasks => "\n\nBreak the plan into ordered implementation tasks.\n".to_string(),
        Stage::Tests => "\n\nWrite test cases covering the spec requirements.\n".to_string(),
        Stage::Implement => "\n\nImplement the code per the task list.\n".to_string(),
        Stage::Analyze => "\n\nAnalyze the implementation for correctness and gaps.\n".to_string(),
    }
}

/// Validate that spec.md contains required headers.
fn validate_spec_headers(content: &str) -> Result<()> {
    let required = ["## User Stories", "## Functional Requirements"];
    for header in required {
        if !content.contains(header) {
            return Err(AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!("spec.md missing required header: '{header}'"),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_ordering_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "test-session").unwrap();
        // Plan requires Specify to exist.
        let result = pipeline.check_prerequisites(Stage::Plan);
        assert!(result.is_err());
    }

    #[test]
    fn specify_has_no_prerequisites() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "s1").unwrap();
        assert!(pipeline.check_prerequisites(Stage::Specify).is_ok());
    }

    #[tokio::test]
    async fn write_artifact_validates_spec_headers() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "s1").unwrap();

        // Missing headers → error.
        let bad = "# Spec\nSome content without headers.";
        let result = pipeline.write_artifact(Stage::Specify, bad).await;
        assert!(result.is_err());

        // With headers → success.
        let good = "## User Stories\n- As a user...\n## Functional Requirements\n- The system...";
        let result = pipeline.write_artifact(Stage::Specify, good).await;
        assert!(result.is_ok());
        assert!(result.unwrap().exists());
    }

    #[test]
    fn all_stages_have_artifacts() {
        for stage in Stage::all() {
            assert!(!stage.artifact().is_empty());
        }
    }
}
