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

    /// Stages whose artifact must exist, with the right shape, before this
    /// stage may run. This is the gate, and it is intentionally narrow: adding a
    /// stage here forbids skipping it.
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

    /// Stages whose artifact is supplied as additional context when it exists.
    ///
    /// Kept separate from [`Self::prerequisites`] because requiring an artifact
    /// gates a stage while context only informs it. Without this split the
    /// pipeline lost most of its own output: every stage received exactly one
    /// document, its immediate prerequisite's.
    ///
    /// Two consequences were observed on a real run. `clarifications.md` was
    /// read by nothing at all, because Plan depends on Specify rather than
    /// Clarify — so the stage whose entire purpose is resolving ambiguity before
    /// planning never reached the planner. `tests/test-plan.md` was likewise
    /// read by nothing, so Implement never saw the tests it was meant to
    /// satisfy; on one project that was the largest document produced, 61 KB,
    /// discarded. Implement also never saw `spec.md` or `plan.md`, so any
    /// requirement not restated in the task list was simply gone.
    pub fn context_inputs(&self) -> &'static [Stage] {
        match self {
            // Nothing precedes it.
            Stage::Specify => &[],
            // Already receives the specification as its prerequisite.
            Stage::Clarify => &[],
            // The whole point of Clarify.
            Stage::Plan => &[Stage::Clarify],
            // Ordered tasks must trace back to requirements, not only to the
            // architecture that was chosen for them.
            Stage::Tasks => &[Stage::Specify],
            // Tests are written against requirements; the plan says which
            // seams exist to test through.
            Stage::Tests => &[Stage::Specify, Stage::Plan],
            // The code must satisfy the requirements and the tests, not just
            // the task titles.
            Stage::Implement => &[Stage::Specify, Stage::Plan, Stage::Tests],
            // A review needs the requirements to judge against, and the task
            // list to find what was skipped. The code itself is read with tools.
            Stage::Analyze => &[Stage::Specify, Stage::Tasks],
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

/// Combined byte budget for a stage's optional context artifacts.
///
/// Required artifacts are never counted against this and never truncated: they
/// are the stage's contract. Context is additional, so it is the part that gets
/// trimmed when a plan or a test plan grows large — and any trim is reported
/// rather than done quietly.
pub const DEFAULT_CONTEXT_BUDGET_BYTES: usize = 96 * 1024;

/// Verification form asserting a path exists.
///
/// Defined here, next to the instruction that asks for it, so the wording the
/// Tasks stage is told to produce and the wording the Implement stage parses
/// cannot drift apart.
pub const EXISTS_PREFIX: &str = "verify-exists:";

/// Verification form asserting a file contains a substring.
pub const CONTAINS_PREFIX: &str = "verify-contains:";

/// Separator between path and expected text in [`CONTAINS_PREFIX`].
pub const CONTAINS_SEPARATOR: &str = "::";

/// Marker left in a prompt where context was cut. Deliberately explicit: a
/// model that receives half a document must be able to tell.
pub const TRUNCATION_MARKER: &str = "\n\n[... TRUNCATED: this artifact was cut to fit the context \
budget. Do not assume the omitted part is empty; ask for it or work only from what is present. ...]";

/// What [`Pipeline::build_prompt_with_report`] put into a prompt.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PromptReport {
    /// Artifact primary names included whole, with their byte counts.
    pub included: Vec<(&'static str, usize)>,
    /// Artifacts that were cut: name, bytes included, bytes omitted.
    pub truncated: Vec<(&'static str, usize, usize)>,
    /// Context artifacts that were expected but absent, so the caller can say
    /// which stage has not been run yet.
    pub missing: Vec<&'static str>,
}

impl PromptReport {
    pub fn was_truncated(&self) -> bool {
        !self.truncated.is_empty()
    }
}

pub struct Pipeline {
    /// PathJail-resolved root for this session. All artifact paths are derived
    /// from this stored path rather than rebuilding a path from caller input.
    session_dir: PathBuf,
    jail: PathJail,
    context_budget: usize,
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
        Ok(Self {
            session_dir,
            jail,
            context_budget: DEFAULT_CONTEXT_BUDGET_BYTES,
        })
    }

    /// Override the combined context budget. Exists so a deployment with a
    /// larger or smaller model window can decide this without a code change,
    /// and so tests can exercise the trimming path with small documents.
    pub fn with_context_budget(mut self, bytes: usize) -> Self {
        self.context_budget = bytes;
        self
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
        Ok(self.build_prompt_with_report(stage, user_context)?.0)
    }

    /// Build the prompt and report what was included, cut, or absent.
    ///
    /// Required artifacts are embedded whole. Context artifacts then share
    /// [`Self::with_context_budget`] by fair share: the smallest is offered an
    /// equal slice first and whatever it does not use rolls forward, which
    /// maximises the number of documents that arrive complete instead of
    /// cutting all of them.
    pub fn build_prompt_with_report(
        &self,
        stage: Stage,
        user_context: &str,
    ) -> Result<(String, PromptReport)> {
        self.check_prerequisites(stage)?;

        let mut prompt = format!("## Stage: {:?}\n\n", stage);
        prompt.push_str(user_context);
        prompt.push('\n');
        let mut report = PromptReport::default();

        // Required artifacts first, whole. Directory roots remain completion
        // contracts, while their primary file is the only generated prose
        // suitable for prompt inclusion.
        for prereq in stage.prerequisites() {
            let spec = prereq.artifact_spec();
            let path = self.primary_artifact_path(*prereq)?;
            if path.is_file() {
                if let Ok(content) = fs::read_to_string(&path) {
                    report.included.push((spec.primary, content.len()));
                    prompt.push_str(&format!(
                        "\n---\n### Prior Artifact: {}\n{}\n",
                        spec.primary, content
                    ));
                }
            }
        }

        // Context artifacts, smallest first so a fair share is not wasted on a
        // document that did not need all of it.
        let required = stage.prerequisites();
        let mut available: Vec<(&'static str, String)> = Vec::new();
        for extra in stage.context_inputs() {
            if required.contains(extra) {
                continue; // already embedded whole above
            }
            let spec = extra.artifact_spec();
            let path = self.primary_artifact_path(*extra)?;
            match fs::read_to_string(&path) {
                Ok(content) if !content.trim().is_empty() => {
                    available.push((spec.primary, content))
                }
                _ => report.missing.push(spec.primary),
            }
        }
        available.sort_by_key(|(_, content)| content.len());

        let mut remaining_budget = self.context_budget;
        let mut remaining_count = available.len();
        for (name, content) in available {
            let share = remaining_budget.checked_div(remaining_count).unwrap_or(0);
            remaining_count = remaining_count.saturating_sub(1);

            if content.len() <= share {
                remaining_budget -= content.len();
                report.included.push((name, content.len()));
                prompt.push_str(&format!("\n---\n### Context Artifact: {name}\n{content}\n"));
                continue;
            }

            // Cut on a character boundary, and say so in the prompt itself.
            let mut cut = share.min(content.len());
            while cut > 0 && !content.is_char_boundary(cut) {
                cut -= 1;
            }
            remaining_budget = remaining_budget.saturating_sub(cut);
            report.truncated.push((name, cut, content.len() - cut));
            prompt.push_str(&format!(
                "\n---\n### Context Artifact: {name} (TRUNCATED)\n{}{}\n",
                &content[..cut],
                TRUNCATION_MARKER
            ));
        }

        prompt.push_str(&stage_instructions(stage));
        Ok((prompt, report))
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
        // Each task carries a check the tool can run itself. Without one, a
        // finished task is only the agent's word: on a real project 112 files
        // were written while the checklist recorded nothing, and on another the
        // list stayed empty while the run reported success. A check turns "done"
        // from a claim into something falsifiable.
        Stage::Tasks => format!(
            "\n\nBreak the plan into ordered implementation tasks.\n\
            \n\
            FORMAT — follow it exactly:\n\
            - Write every task as a markdown checkbox: `- [ ] <what to do>`\n\
            - Order tasks so each one can be done when reached.\n\
            - Give each task at least one indented verification line stating how a \
            program can confirm the task is finished, using ONLY these forms:\n\
            \n\
            \x20   `{EXISTS_PREFIX} <path>`  — that path must exist when the task is done\n\
            \x20   `{CONTAINS_PREFIX} <path> {CONTAINS_SEPARATOR} <text>`  — that file must contain that text\n\
            \n\
            Example:\n\
            \x20 - [ ] Add the score model\n\
            \x20   {EXISTS_PREFIX} src/model/score.ts\n\
            \x20   {CONTAINS_PREFIX} src/model/score.ts {CONTAINS_SEPARATOR} export interface Score\n\
            \n\
            Choose paths and text that are specific enough that the check fails if the \
            task was not really done, and that do not depend on installing anything. \
            Do not invent other verification forms; anything else cannot be checked and \
            leaves the task unverifiable.\n"
        ),
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
        // Absent and present-but-empty are different faults with different
        // fixes, and reporting both as "no content under" sent at least one
        // debugging session down the wrong path.
        let present = content.lines().any(|line| line.trim_end().trim() == header);
        if !present {
            return Err(AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!(
                    "spec.md does not contain the required header '{header}' on a line of its \
                     own, so it was not published. The stage produced something other than a \
                     specification in the requested format."
                ),
            });
        }
        if !has_populated_section(content, header) {
            return Err(AgentError::Tool {
                name: "spec_pipeline".into(),
                reason: format!(
                    "spec.md has the header '{header}' but nothing under it, so it was not \
                     published. The stage listed the headers rather than filling them in."
                ),
            });
        }
    }
    Ok(())
}

/// Does `header` start a line and have real content beneath it?
///
/// A section ends at the next heading of the **same or shallower** depth. A
/// deeper heading is part of the section, not the end of it: grouping user
/// stories under `### Movement` and `### Combat` is ordinary Markdown and, if
/// anything, better structure.
///
/// The first version of this rule treated any `#` line as the end, which
/// rejected a perfectly good 8882-byte specification because the model happened
/// to organise it with sub-headings. Worse, it made the check depend on which
/// model was in use: one that writes bullets directly under the header passed,
/// one that groups them did not. A validation rule that varies by model is not a
/// rule.
fn has_populated_section(content: &str, header: &str) -> bool {
    let depth = heading_depth(header).unwrap_or(2);
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
            match heading_depth(body) {
                // Same or shallower heading: this section closed empty.
                Some(found) if found <= depth => return false,
                // Deeper heading: still inside the section, keep looking for the
                // content it introduces.
                Some(_) => continue,
                // Anything else is content.
                None => return true,
            }
        }
        return false;
    }
    false
}

/// Number of leading `#` characters, when the line is an ATX heading.
fn heading_depth(line: &str) -> Option<usize> {
    let hashes = line.chars().take_while(|c| *c == '#').count();
    // A heading needs at least one `#` followed by a space, so `#hashtag` in
    // prose is not mistaken for a heading.
    (hashes > 0 && line[hashes..].starts_with(' ')).then_some(hashes)
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

    /// Write every stage's primary artifact with recognisable content.
    async fn seed_all(pipeline: &Pipeline, size: usize) {
        for stage in Stage::all() {
            let name = stage.artifact_spec().primary;
            let body = if *stage == Stage::Specify {
                format!(
                    "## User Stories\n- from {name}\n## Functional Requirements\n- from {name}\n{}",
                    "x".repeat(size)
                )
            } else {
                format!("MARKER::{name}\n{}", "x".repeat(size))
            };
            pipeline.write_artifact(*stage, &body).await.unwrap();
        }
    }

    #[test]
    fn every_stage_output_is_read_by_a_later_stage() {
        // The defect this encodes: Clarify's output was consumed by nothing,
        // because Plan depends on Specify, and Tests' output was consumed by
        // nothing either. Two of seven stages wrote documents that no stage
        // ever read, which is indistinguishable from not running them.
        for producer in Stage::all() {
            if *producer == Stage::Analyze {
                continue; // the last stage has no consumer by definition
            }
            let consumed = Stage::all().iter().any(|consumer| {
                consumer.prerequisites().contains(producer)
                    || consumer.context_inputs().contains(producer)
            });
            assert!(
                consumed,
                "{producer:?} produces an artifact that no later stage reads"
            );
        }
    }

    #[test]
    fn context_inputs_never_point_forward_or_at_themselves() {
        // A stage may only be informed by stages that run before it, otherwise
        // the prompt would depend on an artifact that cannot exist yet.
        for stage in Stage::all() {
            for input in stage.context_inputs() {
                assert!(
                    input < stage,
                    "{stage:?} lists {input:?} as context, which does not run earlier"
                );
            }
            for prereq in stage.prerequisites() {
                assert!(prereq < stage, "{stage:?} requires {prereq:?}");
            }
        }
    }

    #[tokio::test]
    async fn implement_receives_the_requirements_plan_and_test_plan() {
        // Observed on a real project: Implement saw only tasks.md, so a 61 KB
        // test plan and the whole specification never reached the code writer.
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "default").unwrap();
        seed_all(&pipeline, 0).await;

        let (prompt, report) = pipeline
            .build_prompt_with_report(Stage::Implement, "build it")
            .unwrap();

        assert!(prompt.contains("### Prior Artifact: tasks.md"), "{prompt}");
        for expected in ["spec.md", "plan.md", "tests/test-plan.md"] {
            assert!(
                prompt.contains(&format!("### Context Artifact: {expected}")),
                "Implement prompt is missing {expected}"
            );
        }
        assert!(!report.was_truncated(), "{report:?}");
        assert!(report.missing.is_empty(), "{report:?}");
    }

    #[tokio::test]
    async fn plan_receives_the_clarifications() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "default").unwrap();
        seed_all(&pipeline, 0).await;

        let prompt = pipeline.build_prompt(Stage::Plan, "plan it").unwrap();
        assert!(prompt.contains("### Prior Artifact: spec.md"), "{prompt}");
        assert!(
            prompt.contains("MARKER::clarifications.md"),
            "Plan must see the answers Clarify produced"
        );
    }

    #[tokio::test]
    async fn a_missing_context_artifact_is_reported_and_does_not_fail_the_stage() {
        // Context is optional by design: skipping Clarify must not block Plan.
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "default").unwrap();
        pipeline
            .write_artifact(
                Stage::Specify,
                "## User Stories\n- a\n## Functional Requirements\n- b\n",
            )
            .await
            .unwrap();

        let (prompt, report) = pipeline
            .build_prompt_with_report(Stage::Plan, "plan it")
            .unwrap();
        assert!(prompt.contains("### Prior Artifact: spec.md"));
        assert_eq!(report.missing, vec!["clarifications.md"]);
        assert!(!report.was_truncated());
    }

    #[tokio::test]
    async fn context_is_trimmed_visibly_and_required_artifacts_are_never_cut() {
        // A required artifact is the stage's contract and must arrive whole even
        // when it is larger than the entire context budget.
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "default")
            .unwrap()
            .with_context_budget(600);
        seed_all(&pipeline, 4096).await;

        let (prompt, report) = pipeline
            .build_prompt_with_report(Stage::Implement, "build it")
            .unwrap();

        // tasks.md is required: present in full, marker and all.
        assert!(prompt.contains("MARKER::tasks.md"));
        let tasks_len =
            std::fs::read_to_string(pipeline.primary_artifact_path(Stage::Tasks).unwrap())
                .unwrap()
                .len();
        assert!(
            report
                .included
                .iter()
                .any(|(n, len)| *n == "tasks.md" && *len == tasks_len),
            "{report:?}"
        );

        // Context was cut, and the prompt says so where it was cut.
        assert!(report.was_truncated(), "{report:?}");
        assert!(prompt.contains(TRUNCATION_MARKER));
        assert!(prompt.contains("(TRUNCATED)"));

        // The budget was respected across all context artifacts together.
        let context_bytes: usize = report
            .truncated
            .iter()
            .map(|(_, kept, _)| *kept)
            .chain(
                report
                    .included
                    .iter()
                    .filter(|(n, _)| *n != "tasks.md")
                    .map(|(_, len)| *len),
            )
            .sum();
        assert!(context_bytes <= 600, "context used {context_bytes} bytes");
    }

    #[tokio::test]
    async fn fair_share_prefers_delivering_whole_documents() {
        // One small and one huge context artifact: the small one must arrive
        // complete rather than both being cut in half.
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "default")
            .unwrap()
            .with_context_budget(1000);
        pipeline
            .write_artifact(
                Stage::Specify,
                "## User Stories\n- small\n## Functional Requirements\n- small\n",
            )
            .await
            .unwrap();
        pipeline
            .write_artifact(
                Stage::Plan,
                &format!("MARKER::plan\n{}", "y".repeat(20_000)),
            )
            .await
            .unwrap();
        pipeline
            .write_artifact(Stage::Tasks, "MARKER::tasks")
            .await
            .unwrap();

        let (prompt, report) = pipeline
            .build_prompt_with_report(Stage::Tests, "write tests")
            .unwrap();

        assert!(
            report.included.iter().any(|(n, _)| *n == "spec.md"),
            "the small specification must arrive whole: {report:?}"
        );
        assert!(
            report.truncated.iter().any(|(n, _, _)| *n == "plan.md"),
            "{report:?}"
        );
        assert!(prompt.contains("- small"));
    }

    #[tokio::test]
    async fn truncation_never_splits_a_multibyte_character() {
        let dir = tempfile::tempdir().unwrap();
        let pipeline = Pipeline::new(dir.path(), "default")
            .unwrap()
            .with_context_budget(41);
        pipeline
            .write_artifact(
                Stage::Specify,
                &format!(
                    "## User Stories\n- {}\n## Functional Requirements\n- x\n",
                    "क".repeat(200)
                ),
            )
            .await
            .unwrap();
        pipeline
            .write_artifact(Stage::Plan, "MARKER::plan")
            .await
            .unwrap();

        // Building must not panic, and the prompt must remain valid UTF-8 text.
        let (prompt, report) = pipeline
            .build_prompt_with_report(Stage::Tasks, "tasks")
            .unwrap();
        assert!(report.was_truncated(), "{report:?}");
        assert!(prompt.contains(TRUNCATION_MARKER));
    }

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
        // Asserted on meaning rather than phrasing: the header must be named and
        // the artifact must be reported as unpublished.
        let message = format!("{error}");
        assert!(message.contains("## User Stories"), "{message}");
        assert!(message.contains("not published"), "{message}");
        assert!(message.contains("nothing under it"), "{message}");
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
    fn a_section_organised_with_sub_headings_is_populated() {
        // The exact shape a live model produced, which the first version of this
        // rule rejected: grouping stories under deeper headings.
        let content = "\
## User Stories
### Player Movement & Traversal
- As a player, I want to walk and run left/right so I can navigate levels.
### Combat
- As a player, I want multiple weapon types so I can adapt.

## Functional Requirements
#### Deeply nested is still content
1. The tool shall do the thing.
";
        validate_spec_headers(content).unwrap();
    }

    #[test]
    fn a_hash_in_prose_is_not_treated_as_a_heading() {
        let content = "\
## User Stories
#hashtag style text is prose, not a heading
## Functional Requirements
- a rule
";
        validate_spec_headers(content).unwrap();
    }

    #[test]
    fn a_same_or_shallower_heading_still_closes_an_empty_section() {
        // Same depth.
        assert!(validate_spec_headers(
            "## User Stories\n## Functional Requirements\n- something\n"
        )
        .is_err());
        // Shallower.
        assert!(validate_spec_headers(
            "## User Stories\n# Appendix\n- something\n## Functional Requirements\n- rule\n"
        )
        .is_err());
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
