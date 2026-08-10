//! RustySpec 7-stage pipeline stages and orchestration.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use agent_types::{AgentError, Result};
use runtime_core::{atomic_replace, AtomicWriteOptions};
use sandbox::PathJail;
use tokio_util::sync::CancellationToken;

use crate::artifacts::{Artifact, ArtifactKind, ArtifactSpec};

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
    /// Return the complete artifact model for this stage.
    pub fn artifact_spec(&self) -> ArtifactSpec {
        match self {
            Stage::Specify => ArtifactSpec {
                root: "spec.md",
                primary: "spec.md",
                kind: ArtifactKind::File,
            },
            Stage::Clarify => ArtifactSpec {
                root: "clarifications.md",
                primary: "clarifications.md",
                kind: ArtifactKind::File,
            },
            Stage::Plan => ArtifactSpec {
                root: "plan.md",
                primary: "plan.md",
                kind: ArtifactKind::File,
            },
            Stage::Tasks => ArtifactSpec {
                root: "tasks.md",
                primary: "tasks.md",
                kind: ArtifactKind::File,
            },
            Stage::Tests => ArtifactSpec {
                root: "tests",
                primary: "tests/test-plan.md",
                kind: ArtifactKind::Directory,
            },
            Stage::Implement => ArtifactSpec {
                root: "code",
                primary: "code/IMPLEMENTATION.md",
                kind: ArtifactKind::Directory,
            },
            Stage::Analyze => ArtifactSpec {
                root: "analysis.md",
                primary: "analysis.md",
                kind: ArtifactKind::File,
            },
        }
    }

    /// Compatibility accessor for the stage artifact root.
    ///
    /// Directory roots retain their historical trailing slash so existing
    /// callers and user-facing diagnostics remain unchanged.
    pub fn artifact(&self) -> &'static str {
        match self {
            Stage::Tests => "tests/",
            Stage::Implement => "code/",
            _ => self.artifact_spec().root,
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
    /// PathJail-resolved root for this session. All artifact paths are derived
    /// from this stored path rather than rebuilding a path from caller input.
    session_dir: PathBuf,
    jail: PathJail,
}

impl Pipeline {
    /// Create a new pipeline for a session. All artifacts live under
    /// `.agent/specs/<session>/`.
    pub fn new(project_root: &Path, session_id: &str) -> Result<Self> {
        // Validate before touching the workspace so an unsafe identifier can
        // never influence artifact lookup or creation.
        validate_session_id(session_id)?;

        let jail = PathJail::new(project_root)?;
        let session_dir = jail.resolve(&Path::new(".agent").join("specs").join(session_id))?;
        Ok(Self { session_dir, jail })
    }

    /// Resolve a fixed path below the stored jailed session root.
    fn resolve_session_path(&self, relative: &str) -> Result<PathBuf> {
        let path = self.jail.resolve(&self.session_dir.join(relative))?;
        if !path.starts_with(&self.session_dir) {
            return Err(AgentError::PathJail(format!(
                "specification artifact '{}' escapes session root '{}'",
                path.display(),
                self.session_dir.display()
            )));
        }
        Ok(path)
    }

    /// Return a lexical fixed path after proving that its current resolution
    /// is jailed. Migration uses the lexical path so a symlink cannot be
    /// mistaken for the legacy regular file that owns the `tests` name.
    fn checked_lexical_session_path(&self, relative: &str) -> Result<PathBuf> {
        let path = self.session_dir.join(relative);
        self.resolve_session_path(relative)?;
        Ok(path)
    }

    /// Resolve a stage's compatibility artifact root below the stored jailed
    /// session root.
    ///
    /// Callers select a [`Stage`]; they cannot provide a later raw artifact
    /// path. Re-resolving before each operation detects a session directory
    /// that was replaced with an escaping symlink after pipeline creation.
    pub fn artifact_path(&self, stage: Stage) -> Result<PathBuf> {
        self.resolve_session_path(stage.artifact_spec().root)
    }

    /// Resolve the concrete file read and written for a stage.
    pub fn primary_artifact_path(&self, stage: Stage) -> Result<PathBuf> {
        self.resolve_session_path(stage.artifact_spec().primary)
    }

    /// Check that all prerequisites for a stage are satisfied with the
    /// filesystem shape declared by each prerequisite's artifact kind.
    pub fn check_prerequisites(&self, stage: Stage) -> Result<()> {
        for prereq in stage.prerequisites() {
            let spec = prereq.artifact_spec();
            let artifact_path = self.artifact_path(*prereq)?;
            let exists_with_expected_kind = match spec.kind {
                ArtifactKind::File => artifact_path.is_file(),
                ArtifactKind::Directory => artifact_path.is_dir(),
            };
            if !exists_with_expected_kind {
                return Err(AgentError::Tool {
                    name: "spec_pipeline".into(),
                    reason: format!(
                        "prerequisite '{}' missing or has the wrong artifact kind for stage {:?} (expected {:?} at {})",
                        prereq.artifact(),
                        stage,
                        spec.kind,
                        artifact_path.display()
                    ),
                });
            }
        }
        Ok(())
    }

    /// Build the prompt for a stage by embedding prior primary artifacts.
    pub fn build_prompt(&self, stage: Stage, user_context: &str) -> Result<String> {
        self.check_prerequisites(stage)?;

        let mut prompt = format!("## Stage: {:?}\n\n", stage);
        prompt.push_str(user_context);
        prompt.push('\n');

        // Embed prior primary artifacts. Directory roots remain completion
        // contracts, while their primary file is the only generated prose
        // suitable for prompt inclusion.
        for prereq in stage.prerequisites() {
            let spec = prereq.artifact_spec();
            let path = self.primary_artifact_path(*prereq)?;
            if path.is_file() {
                if let Ok(content) = fs::read_to_string(&path) {
                    prompt.push_str(&format!(
                        "\n---\n### Prior Artifact: {}\n{}\n",
                        spec.primary, content
                    ));
                }
            }
        }

        prompt.push_str(&stage_instructions(stage));
        Ok(prompt)
    }

    /// Write a stage's primary artifact atomically. Directory artifacts retain
    /// their root and any sibling artifacts. Validates required headers for
    /// `spec.md`.
    pub async fn write_artifact(&self, stage: Stage, content: &str) -> Result<PathBuf> {
        // Applies to every stage. An artifact with no substance is not a result,
        // and publishing one lets the next stage run against nothing while the
        // run reports success.
        if content.trim().is_empty() {
            return Err(AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!(
                    "stage {stage:?} produced no content, so no artifact was published. \
                     Re-run the stage."
                ),
            });
        }
        if stage == Stage::Specify {
            validate_spec_headers(content)?;
        }

        let spec = stage.artifact_spec();
        if spec.kind == ArtifactKind::Directory {
            self.prepare_directory_artifact(stage).await?;
        }

        let artifact_path = self.primary_artifact_path(stage)?;
        if let Some(parent) = artifact_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let cancel = CancellationToken::new();
        atomic_replace(
            &artifact_path,
            content.as_bytes(),
            AtomicWriteOptions::default(),
            &cancel,
        )
        .await?;

        Ok(artifact_path)
    }

    /// Load an existing stage's primary artifact.
    pub fn load_artifact(&self, stage: Stage) -> Result<Artifact> {
        let spec = stage.artifact_spec();
        let path = self.primary_artifact_path(stage)?;
        if !path.is_file() {
            return Err(AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!("artifact '{}' not found", spec.primary),
            });
        }
        let content = fs::read_to_string(&path).map_err(|error| AgentError::Tool {
            name: "spec_pipeline".into(),
            reason: format!("read artifact: {error}"),
        })?;
        Ok(Artifact {
            stage,
            path,
            content,
        })
    }

    async fn prepare_directory_artifact(&self, stage: Stage) -> Result<()> {
        let spec = stage.artifact_spec();
        debug_assert_eq!(spec.kind, ArtifactKind::Directory);

        let root = self.checked_lexical_session_path(spec.root)?;
        let parent = root
            .parent()
            .ok_or_else(|| artifact_error("artifact root has no parent"))?;
        tokio::fs::create_dir_all(parent).await?;

        match fs::symlink_metadata(&root) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(artifact_error(format!(
                    "directory artifact root '{}' is a symlink; refusing ambiguous artifact migration",
                    root.display()
                )));
            }
            Ok(metadata) if metadata.is_dir() => return Ok(()),
            Ok(metadata) if metadata.is_file() && stage == Stage::Tests => {
                self.migrate_legacy_tests_file(&root)?;
            }
            Ok(_) => {
                return Err(artifact_error(format!(
                    "directory artifact root '{}' has an unsupported filesystem kind",
                    root.display()
                )));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(artifact_error(format!(
                    "inspect directory artifact root '{}': {error}",
                    root.display()
                )));
            }
        }

        match tokio::fs::create_dir(&root).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(&root).map_err(|inspect_error| {
                    artifact_error(format!(
                        "artifact root '{}' appeared during directory creation but could not be inspected: {inspect_error}",
                        root.display()
                    ))
                })?;
                if metadata.is_dir() && !metadata.file_type().is_symlink() {
                    Ok(())
                } else {
                    Err(artifact_error(format!(
                        "artifact root '{}' changed to an ambiguous filesystem kind during directory creation",
                        root.display()
                    )))
                }
            }
            Err(error) => Err(artifact_error(format!(
                "create directory artifact root '{}': {error}",
                root.display()
            ))),
        }
    }

    /// Move the legacy regular `tests` file to one unambiguous backup name.
    ///
    /// An exclusive migration lock serializes cooperating writers. A stale
    /// lock, a pre-existing backup, a changed source kind, or rename failure
    /// stops migration without deleting the legacy file. Directory creation is
    /// attempted only after the rename succeeds.
    fn migrate_legacy_tests_file(&self, legacy_root: &Path) -> Result<()> {
        let lock_path = self.checked_lexical_session_path("tests.migration.lock")?;
        let _lock = MigrationLock::acquire(&lock_path)?;

        let metadata = fs::symlink_metadata(legacy_root).map_err(|error| {
            artifact_error(format!(
                "legacy Tests artifact '{}' changed before migration: {error}",
                legacy_root.display()
            ))
        })?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(artifact_error(format!(
                "legacy Tests artifact '{}' is no longer an unambiguous regular file",
                legacy_root.display()
            )));
        }

        let backup = self.checked_lexical_session_path("tests.backup")?;
        match fs::symlink_metadata(&backup) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err(artifact_error(format!(
                    "legacy Tests backup '{}' already exists; refusing ambiguous migration",
                    backup.display()
                )));
            }
            Err(error) => {
                return Err(artifact_error(format!(
                    "inspect legacy Tests backup '{}': {error}",
                    backup.display()
                )));
            }
        }

        fs::rename(legacy_root, &backup).map_err(|error| {
            artifact_error(format!(
                "back up legacy Tests artifact '{}' to '{}': {error}",
                legacy_root.display(),
                backup.display()
            ))
        })?;
        Ok(())
    }

    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    pub fn jail(&self) -> &PathJail {
        &self.jail
    }
}

struct MigrationLock {
    path: PathBuf,
    file: Option<File>,
}

impl MigrationLock {
    fn acquire(path: &Path) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        let file = options.open(path).map_err(|error| {
            artifact_error(format!(
                "acquire Tests artifact migration lock '{}': {error}",
                path.display()
            ))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
        })
    }
}

impl Drop for MigrationLock {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

fn artifact_error(reason: impl Into<String>) -> AgentError {
    AgentError::Tool {
        name: "spec_pipeline".into(),
        reason: reason.into(),
    }
}

/// Validate the selected public session identifier grammar:
/// `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`.
fn validate_session_id(session_id: &str) -> Result<()> {
    let bytes = session_id.as_bytes();
    let contains_separator = bytes.iter().any(|byte| matches!(byte, b'/' | b'\\'));
    let has_dot_segment = session_id == "."
        || session_id == ".."
        || session_id
            .split(['/', '\\'])
            .any(|component| matches!(component, "." | ".."));
    let has_drive_prefix = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    let has_unc_prefix =
        bytes.len() >= 2 && matches!(bytes[0], b'/' | b'\\') && matches!(bytes[1], b'/' | b'\\');

    let valid_grammar = (1..=64).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes[1..]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));

    if !valid_grammar
        || has_dot_segment
        || contains_separator
        || has_drive_prefix
        || has_unc_prefix
        || Path::new(session_id).is_absolute()
    {
        return Err(AgentError::Tool {
            name: "spec_pipeline".into(),
            reason: format!(
                "invalid session identifier {session_id:?}: expected ASCII [A-Za-z0-9][A-Za-z0-9._-]{{0,63}} without dot segments, separators, absolute, drive, or UNC syntax"
            ),
        });
    }

    Ok(())
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

/// Validate that spec.md is a specification and not a message about one.
///
/// Substring presence alone is not enough, and this was found the hard way: a
/// provider that could not see a referenced file answered "I don't have access
/// to your local files… I will produce a specification organized under these
/// headers:" and then listed the header names. Every required substring was
/// present, so a refusal was published as `spec.md` and the pipeline carried on
/// to the next stage. Nothing downstream could tell.
///
/// Each required header must therefore start a line and be followed by at least
/// one line of its own content before the next header or the end of the file.
/// A merely *listed* header has nothing under it and is rejected. This is a
/// structural rule with no wording or language assumptions.
fn validate_spec_headers(content: &str) -> Result<()> {
    let required = ["## User Stories", "## Functional Requirements"];
    for header in required {
        if !has_populated_section(content, header) {
            return Err(AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!(
                    "spec.md has no content under required header '{header}'. \
                     The stage produced a message rather than a specification, \
                     so it was not published. Re-run the stage with the missing \
                     information supplied."
                ),
            });
        }
    }
    Ok(())
}

/// Does `header` start a line and have at least one non-blank, non-header line
/// beneath it?
fn has_populated_section(content: &str, header: &str) -> bool {
    let mut lines = content.lines().map(str::trim_end);
    while let Some(line) = lines.next() {
        if line.trim() != header {
            continue;
        }
        for body in lines.by_ref() {
            let body = body.trim();
            if body.is_empty() {
                continue;
            }
            // A new heading ends this section without having filled it.
            return !body.starts_with('#');
        }
        return false;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact text a provider returned when it could not see a referenced
    /// file. Kept verbatim because it is what defeated the previous check.
    const REFUSAL_LISTING_THE_HEADERS: &str = "\
I don't have access to your local files. Please paste the contents of req.md here.

Once you paste req.md, I will produce a specification organized under these headers:
## User Stories
## Functional Requirements
## Non-Functional Requirements

If you prefer, confirm and I can create a first-draft spec from minimal input.
";

    #[tokio::test]
    async fn an_empty_artifact_is_refused_for_every_stage() {
        // The header rule only guards Specify. Emptiness is checked everywhere,
        // because an empty artifact still let the next stage run and still
        // reported success.
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "default").unwrap();

        for stage in Stage::all() {
            for blank in ["", "   ", "\n\n", "\t \r\n"] {
                let error = pipeline
                    .write_artifact(*stage, blank)
                    .await
                    .expect_err("an empty artifact must not be published");
                assert!(
                    format!("{error}").contains("produced no content"),
                    "{stage:?}: {error}"
                );
            }
            assert!(
                !pipeline.primary_artifact_path(*stage).unwrap().exists(),
                "{stage:?} must not leave a file behind"
            );
        }
    }

    #[test]
    fn a_message_that_merely_lists_the_headers_is_not_a_specification() {
        // This was published as spec.md on a live run, and the pipeline then ran
        // the next stage against it.
        let error = validate_spec_headers(REFUSAL_LISTING_THE_HEADERS)
            .expect_err("a refusal that lists the headers must be rejected");
        let message = format!("{error}");
        assert!(message.contains("## User Stories"), "{message}");
        assert!(message.contains("no content under"), "{message}");
    }

    #[test]
    fn a_real_specification_is_accepted() {
        let content = "\
## User Stories
- As a user, I want to record an expense.

## Functional Requirements
1. The tool shall accept an amount and a category.

## Non-Functional Requirements
- Single file storage.
";
        validate_spec_headers(content).unwrap();
    }

    #[test]
    fn an_empty_or_header_only_section_is_rejected_either_way() {
        // Present but empty, with the next header immediately after.
        let empty_first = "## User Stories\n## Functional Requirements\n- something\n";
        assert!(validate_spec_headers(empty_first).is_err());

        // Present, populated, but the second required header is bare at the end.
        let empty_last = "## User Stories\n- a story\n\n## Functional Requirements\n\n";
        assert!(validate_spec_headers(empty_last).is_err());

        // Blank lines between the header and its content are fine.
        let spaced = "## User Stories\n\n\n- a story\n## Functional Requirements\n\n- a rule\n";
        validate_spec_headers(spaced).unwrap();
    }

    #[test]
    fn a_header_must_start_its_own_line() {
        // Quoted inline, so the document only talks about the header.
        let inline = "I will write ## User Stories and ## Functional Requirements next.\n";
        assert!(validate_spec_headers(inline).is_err());
    }

    #[test]
    fn session_id_grammar_accepts_only_selected_ascii_shape() {
        for session_id in ["a", "Z9", "safe-session", "release_2026.07", "a..b"] {
            validate_session_id(session_id).unwrap();
        }
        validate_session_id(&"a".repeat(64)).unwrap();

        for session_id in [
            "",
            ".",
            "..",
            "../outside",
            "safe/../outside",
            r"safe\..\outside",
            "/outside",
            r"C:\outside",
            "C:outside",
            r"\\server\share",
            "-leading",
            "_leading",
            "contains space",
            "session:alias",
            "सत्र",
        ] {
            assert!(
                validate_session_id(session_id).is_err(),
                "unsafe or out-of-grammar identifier was accepted: {session_id:?}"
            );
        }
        assert!(validate_session_id(&"a".repeat(65)).is_err());
    }

    #[test]
    fn unsafe_session_id_is_rejected_before_workspace_access() {
        let missing_root = Path::new("definitely-missing-session-root");
        let error = match Pipeline::new(missing_root, "../outside") {
            Ok(_) => panic!("unsafe session identifier was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("invalid session identifier"));
    }

    #[test]
    fn artifact_paths_are_derived_from_stored_jailed_session_root() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "safe.session-1").unwrap();
        // The jail strips Windows' `\\?\` verbatim prefix, so compare against
        // the jail root rather than a raw `canonicalize` result.
        let expected_root = pipeline
            .jail()
            .root()
            .join(".agent")
            .join("specs")
            .join("safe.session-1");
        assert_eq!(pipeline.session_dir(), expected_root);

        for stage in Stage::all() {
            for path in [
                pipeline.artifact_path(*stage).unwrap(),
                pipeline.primary_artifact_path(*stage).unwrap(),
            ] {
                assert!(path.starts_with(pipeline.session_dir()));
                assert!(path.starts_with(pipeline.jail().root()));
            }
        }
    }

    // **Validates: Requirements 2.19, 3.12**
    #[test]
    fn artifact_specs_retain_compatibility_roots_and_define_primary_files() {
        assert_eq!(
            Stage::Tests.artifact_spec(),
            ArtifactSpec {
                root: "tests",
                primary: "tests/test-plan.md",
                kind: ArtifactKind::Directory,
            }
        );
        assert_eq!(Stage::Tests.artifact(), "tests/");
        assert_eq!(
            Stage::Implement.artifact_spec(),
            ArtifactSpec {
                root: "code",
                primary: "code/IMPLEMENTATION.md",
                kind: ArtifactKind::Directory,
            }
        );
        assert_eq!(Stage::Implement.artifact(), "code/");

        for stage in [
            Stage::Specify,
            Stage::Clarify,
            Stage::Plan,
            Stage::Tasks,
            Stage::Analyze,
        ] {
            let spec = stage.artifact_spec();
            assert_eq!(spec.kind, ArtifactKind::File);
            assert_eq!(spec.root, spec.primary);
            assert_eq!(stage.artifact(), spec.root);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn artifact_write_rejects_session_symlink_escape() {
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(workspace.path(), "safe-session").unwrap();
        fs::create_dir_all(pipeline.session_dir().parent().unwrap()).unwrap();
        symlink(outside.path(), pipeline.session_dir()).unwrap();

        let content = "## User Stories\n- safe\n## Functional Requirements\n- confined\n";
        let error = pipeline
            .write_artifact(Stage::Specify, content)
            .await
            .unwrap_err();
        assert!(matches!(error, AgentError::PathJail(_)));
        assert!(!outside.path().join("spec.md").exists());
    }

    #[test]
    fn stage_ordering_and_artifact_kinds_are_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "test-session").unwrap();
        assert!(pipeline.check_prerequisites(Stage::Plan).is_err());

        fs::create_dir_all(pipeline.session_dir()).unwrap();
        fs::create_dir(pipeline.session_dir().join("tasks.md")).unwrap();
        assert!(pipeline.check_prerequisites(Stage::Implement).is_err());
        fs::remove_dir(pipeline.session_dir().join("tasks.md")).unwrap();
        fs::write(pipeline.session_dir().join("tasks.md"), "# Tasks").unwrap();
        pipeline.check_prerequisites(Stage::Implement).unwrap();

        fs::write(pipeline.session_dir().join("code"), "not a directory").unwrap();
        assert!(pipeline.check_prerequisites(Stage::Analyze).is_err());
        fs::remove_file(pipeline.session_dir().join("code")).unwrap();
        fs::create_dir(pipeline.session_dir().join("code")).unwrap();
        pipeline.check_prerequisites(Stage::Analyze).unwrap();
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

        let bad = "# Spec\nSome content without headers.";
        let result = pipeline.write_artifact(Stage::Specify, bad).await;
        assert!(result.is_err());

        let good = "## User Stories\n- As a user...\n## Functional Requirements\n- The system...";
        let result = pipeline.write_artifact(Stage::Specify, good).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_file());
    }

    // **Validates: Requirements 2.19**
    #[tokio::test]
    async fn tests_stage_writes_primary_and_preserves_sibling_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "multi-artifact").unwrap();
        let tests_root = pipeline.session_dir().join("tests");
        fs::create_dir_all(&tests_root).unwrap();
        fs::write(tests_root.join("generated-case.rs"), "preserve me").unwrap();

        let written = pipeline
            .write_artifact(Stage::Tests, "# Test plan")
            .await
            .unwrap();
        assert_eq!(written, tests_root.join("test-plan.md"));
        assert_eq!(fs::read_to_string(&written).unwrap(), "# Test plan");
        assert_eq!(
            fs::read_to_string(tests_root.join("generated-case.rs")).unwrap(),
            "preserve me"
        );

        let loaded = pipeline.load_artifact(Stage::Tests).unwrap();
        assert_eq!(loaded.path, written);
        assert_eq!(loaded.content, "# Test plan");
    }

    // **Validates: Requirements 2.19**
    #[tokio::test]
    async fn legacy_tests_file_is_renamed_before_directory_creation() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "legacy-tests").unwrap();
        fs::create_dir_all(pipeline.session_dir()).unwrap();
        let legacy = pipeline.session_dir().join("tests");
        fs::write(&legacy, "legacy plan").unwrap();

        let written = pipeline
            .write_artifact(Stage::Tests, "new plan")
            .await
            .unwrap();
        let backup = pipeline.session_dir().join("tests.backup");
        assert!(legacy.is_dir());
        assert_eq!(written, legacy.join("test-plan.md"));
        assert_eq!(fs::read_to_string(backup).unwrap(), "legacy plan");
        assert_eq!(fs::read_to_string(written).unwrap(), "new plan");
        assert!(!pipeline.session_dir().join("tests.migration.lock").exists());
    }

    // **Validates: Requirements 2.19**
    #[tokio::test]
    async fn ambiguous_legacy_backup_stops_without_touching_either_file() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "ambiguous-tests").unwrap();
        fs::create_dir_all(pipeline.session_dir()).unwrap();
        let legacy = pipeline.session_dir().join("tests");
        let backup = pipeline.session_dir().join("tests.backup");
        fs::write(&legacy, "legacy plan").unwrap();
        fs::write(&backup, "older backup").unwrap();

        let error = pipeline
            .write_artifact(Stage::Tests, "new plan")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("refusing ambiguous migration"));
        assert!(legacy.is_file());
        assert_eq!(fs::read_to_string(&legacy).unwrap(), "legacy plan");
        assert_eq!(fs::read_to_string(&backup).unwrap(), "older backup");
        assert!(!pipeline.session_dir().join("tests/test-plan.md").exists());
        assert!(!pipeline.session_dir().join("tests.migration.lock").exists());
    }

    // **Validates: Requirements 3.12**
    #[tokio::test]
    async fn implement_primary_coexists_with_compatibility_root() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "implementation-contract").unwrap();
        let written = pipeline
            .write_artifact(Stage::Implement, "# Implementation")
            .await
            .unwrap();

        assert_eq!(
            pipeline.artifact_path(Stage::Implement).unwrap(),
            pipeline.session_dir().join("code")
        );
        assert_eq!(
            written,
            pipeline.session_dir().join("code/IMPLEMENTATION.md")
        );
        assert_eq!(fs::read_to_string(written).unwrap(), "# Implementation");
    }

    #[test]
    fn all_stages_have_artifacts() {
        for stage in Stage::all() {
            let spec = stage.artifact_spec();
            assert!(!stage.artifact().is_empty());
            assert!(!spec.root.is_empty());
            assert!(!spec.primary.is_empty());
        }
    }
}
