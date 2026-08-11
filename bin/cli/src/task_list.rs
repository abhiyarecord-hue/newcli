//! Parsing and durable ticking of the Tasks stage checklist.
//!
//! The Implement stage used to be a single turn handed the whole task list. On a
//! real project that list held 338 checkbox items; the run produced the project
//! scaffolding, stopped, reported success, and left all 338 boxes unticked.
//! Nothing recorded how much was done, so nothing could resume, and nothing
//! noticed the remaining 300-odd tasks. Planning looked excellent and the
//! product was a coloured background.
//!
//! Progress therefore has to live in the document itself. Ticking a box in
//! `tasks.md` is durable, is visible to the person running the tool, survives a
//! crash, and lets a later run pick up where the previous one stopped.
//!
//! Rewriting is deliberately surgical: only the two characters inside a
//! checkbox change. Indentation, numbering, nesting, trailing whitespace and
//! every non-task line survive byte for byte, because this file is also a
//! document a human reads and edits.
//!
//! A task may also carry verifications — checks this tool runs itself before
//! ticking. Without them a tick is only the agent's word, which was not good
//! enough in practice: one run wrote 112 files and reported nothing, another
//! reported success with an empty checklist.

use std::path::Path;

/// A check the tool can run itself to decide whether a task is really finished.
///
/// Only forms that need no installation, no network and no command execution are
/// supported, so verification is free, deterministic, and safe to run after
/// every batch. A task without one can still be ticked, but only as the agent's
/// claim, and it is counted separately so the difference stays visible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verification {
    /// The path must exist.
    Exists { path: String },
    /// The file must exist and contain this text.
    Contains { path: String, text: String },
}

impl Verification {
    /// Run the check against `root`. `Ok(())` means satisfied.
    pub fn check(&self, root: &Path) -> std::result::Result<(), String> {
        match self {
            Self::Exists { path } => {
                let full = root.join(path);
                if full.exists() {
                    Ok(())
                } else {
                    Err(format!("{path} does not exist"))
                }
            }
            Self::Contains { path, text } => {
                let full = root.join(path);
                match std::fs::read_to_string(&full) {
                    Ok(body) if body.contains(text.as_str()) => Ok(()),
                    Ok(_) => Err(format!("{path} does not contain {text:?}")),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        Err(format!("{path} does not exist"))
                    }
                    Err(error) => Err(format!("{path} could not be read: {error}")),
                }
            }
        }
    }
}

/// A checkbox line in the task document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    /// Zero-based index of the line this task occupies.
    pub line: usize,
    pub done: bool,
    /// The task text, with the list marker and checkbox removed.
    pub text: String,
    /// Checks attached to this task, in the order they were written.
    pub verifications: Vec<Verification>,
}

impl Task {
    /// Run every attached check, returning the failures.
    pub fn failing_checks(&self, root: &Path) -> Vec<String> {
        self.verifications
            .iter()
            .filter_map(|check| check.check(root).err())
            .collect()
    }
}

/// One source line and the exact terminator that followed it.
///
/// Terminators are carried rather than normalised. `str::lines` discards them,
/// and rejoining with `\n` silently rewrote every CRLF document to LF — on
/// Windows that turns a two-character tick into a diff touching every line of a
/// file the user may have committed.
#[derive(Clone, Debug)]
struct Line {
    text: String,
    terminator: &'static str,
}

/// A parsed task document that can be ticked and rendered back.
#[derive(Clone, Debug)]
pub struct TaskList {
    lines: Vec<Line>,
    tasks: Vec<Task>,
}

impl TaskList {
    /// Parse every markdown checkbox item, at any indentation level.
    pub fn parse(content: &str) -> Self {
        let mut lines: Vec<Line> = Vec::new();
        let mut rest = content;
        while !rest.is_empty() {
            match rest.find('\n') {
                Some(at) => {
                    let raw = &rest[..at];
                    let (text, terminator) = match raw.strip_suffix('\r') {
                        Some(stripped) => (stripped, "\r\n"),
                        None => (raw, "\n"),
                    };
                    lines.push(Line {
                        text: text.to_string(),
                        terminator,
                    });
                    rest = &rest[at + 1..];
                }
                None => {
                    lines.push(Line {
                        text: rest.to_string(),
                        terminator: "",
                    });
                    break;
                }
            }
        }

        let mut tasks: Vec<Task> = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            if let Some((done, text)) = parse_checkbox(&line.text) {
                tasks.push(Task {
                    line: index,
                    done,
                    text,
                    verifications: Vec::new(),
                });
                continue;
            }
            // A verification belongs to the task above it. Attaching it to the
            // most recent task rather than by indentation depth keeps this
            // tolerant of however the model chose to indent.
            if let Some(check) = parse_verification(&line.text) {
                if let Some(task) = tasks.last_mut() {
                    task.verifications.push(check);
                }
            }
        }

        Self { lines, tasks }
    }

    /// Every parsed task, in document order. Used by the tests that pin down
    /// exactly which lines count as tasks; the runtime works from
    /// [`Self::pending`].
    #[cfg(test)]
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    pub fn total(&self) -> usize {
        self.tasks.len()
    }

    pub fn done_count(&self) -> usize {
        self.tasks.iter().filter(|task| task.done).count()
    }

    pub fn pending_count(&self) -> usize {
        self.total() - self.done_count()
    }

    /// Pending tasks in document order, which is the order the Tasks stage was
    /// asked to produce them in.
    pub fn pending(&self) -> Vec<&Task> {
        self.tasks.iter().filter(|task| !task.done).collect()
    }

    /// Tick the given lines. Returns how many actually changed, so a caller can
    /// tell real progress from a repeated claim about already-finished work.
    pub fn mark_done(&mut self, lines: &[usize]) -> usize {
        let mut changed = 0;
        for &line in lines {
            let Some(task) = self.tasks.iter_mut().find(|task| task.line == line) else {
                continue;
            };
            if task.done {
                continue;
            }
            let Some(source) = self.lines.get_mut(line) else {
                continue;
            };
            // Replace only the checkbox, so the rest of the line is untouched.
            let Some(open) = source.text.find('[') else {
                continue;
            };
            let close = open + 2;
            if source.text.len() <= close || !source.text[close..].starts_with(']') {
                continue;
            }
            source.text.replace_range(open + 1..close, "x");
            task.done = true;
            changed += 1;
        }
        changed
    }

    /// Render the document back, preserving each original line terminator.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for line in &self.lines {
            out.push_str(&line.text);
            out.push_str(line.terminator);
        }
        out
    }
}

/// Recognise a verification line in either supported form.
///
/// A leading list marker is tolerated because a model asked for an indented line
/// under a bullet will often write another bullet.
fn parse_verification(line: &str) -> Option<Verification> {
    let mut body = line.trim();
    for marker in ["- ", "* ", "+ ", "-", "*", "+"] {
        if let Some(rest) = body.strip_prefix(marker) {
            body = rest.trim_start();
            break;
        }
    }
    // Backticks are common when a model quotes the form it was shown.
    let body = body.trim_matches('`').trim();

    if let Some(rest) = strip_prefix_ignore_case(body, spec_pipeline::EXISTS_PREFIX) {
        let path = rest.trim().trim_matches('`').trim();
        return (!path.is_empty()).then(|| Verification::Exists {
            path: path.to_string(),
        });
    }
    if let Some(rest) = strip_prefix_ignore_case(body, spec_pipeline::CONTAINS_PREFIX) {
        let (path, text) = rest.split_once(spec_pipeline::CONTAINS_SEPARATOR)?;
        let path = path.trim().trim_matches('`').trim();
        let text = text.trim();
        if path.is_empty() || text.is_empty() {
            return None;
        }
        return Some(Verification::Contains {
            path: path.to_string(),
            text: text.to_string(),
        });
    }
    None
}

fn strip_prefix_ignore_case<'a>(body: &'a str, prefix: &str) -> Option<&'a str> {
    body.get(..prefix.len())
        .filter(|head| head.eq_ignore_ascii_case(prefix))
        .map(|_| &body[prefix.len()..])
}

/// Recognise `- [ ] text`, `* [x] text`, `+ [X] text` at any indentation.
fn parse_checkbox(line: &str) -> Option<(bool, String)> {
    let trimmed = line.trim_start();
    let rest = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
        .or_else(|| trimmed.strip_prefix('-'))
        .or_else(|| trimmed.strip_prefix('*'))
        .or_else(|| trimmed.strip_prefix('+'))?
        .trim_start();

    let inner = rest.strip_prefix('[')?;
    let mut chars = inner.chars();
    let state = chars.next()?;
    let closed = chars.next()? == ']';
    if !closed {
        return None;
    }
    let done = match state {
        ' ' => false,
        'x' | 'X' => true,
        _ => return None,
    };
    let text = inner[2..].trim().to_string();
    if text.is_empty() {
        return None;
    }
    Some((done, text))
}

/// Marker the Implement stage asks the model to end its reply with.
pub const COMPLETION_MARKER: &str = "COMPLETED:";

/// Read the batch numbers the model claims to have finished.
///
/// The last marker wins, because a model that restates its plan before acting
/// would otherwise have its intention read as its result. Numbers outside the
/// batch are discarded rather than clamped: a number that does not identify a
/// task in this batch identifies nothing.
///
/// This is a *claim*. The caller is expected to require that the batch also
/// committed a file mutation before acting on it.
pub fn parse_completed_report(response: &str, batch_size: usize) -> Vec<usize> {
    let Some(position) = response.rfind(COMPLETION_MARKER) else {
        return Vec::new();
    };
    let tail = &response[position + COMPLETION_MARKER.len()..];
    // Stop at a blank line so a later paragraph cannot contribute digits.
    let region = tail.split("\n\n").next().unwrap_or(tail);

    if region.to_ascii_lowercase().contains("all") {
        return (1..=batch_size).collect();
    }

    let mut claimed: Vec<usize> = Vec::new();
    let mut digits = String::new();
    for character in region.chars() {
        if character.is_ascii_digit() {
            digits.push(character);
            continue;
        }
        if !digits.is_empty() {
            if let Ok(value) = digits.parse::<usize>() {
                if (1..=batch_size).contains(&value) && !claimed.contains(&value) {
                    claimed.push(value);
                }
            }
            digits.clear();
        }
    }
    if let Ok(value) = digits.parse::<usize>() {
        if (1..=batch_size).contains(&value) && !claimed.contains(&value) {
            claimed.push(value);
        }
    }
    claimed.sort_unstable();
    claimed
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCUMENT: &str = "\
# Tasks

## 1. Foundation
- [ ] Create the Vite project
- [x] Add TypeScript config
  - [ ] Enable strict mode
* [ ] Configure ESLint

Some prose that is not a task.
- Not a checkbox at all
- [?] Unknown state is not a task
- [ ]
";

    #[test]
    fn every_checkbox_is_found_at_any_depth_and_marker_style() {
        let list = TaskList::parse(DOCUMENT);
        let texts: Vec<&str> = list.tasks().iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "Create the Vite project",
                "Add TypeScript config",
                "Enable strict mode",
                "Configure ESLint",
            ]
        );
        assert_eq!(list.total(), 4);
        assert_eq!(list.done_count(), 1);
        assert_eq!(list.pending_count(), 3);
    }

    #[test]
    fn non_tasks_are_not_mistaken_for_tasks() {
        let list = TaskList::parse(DOCUMENT);
        // Prose, a plain bullet, an unknown state, and an empty box are all out.
        assert!(list.tasks().iter().all(|t| !t.text.contains("prose")));
        assert!(list
            .tasks()
            .iter()
            .all(|t| t.text != "Not a checkbox at all"));
        assert!(list
            .tasks()
            .iter()
            .all(|t| !t.text.contains("Unknown state")));
    }

    #[test]
    fn ticking_changes_only_the_checkbox_and_nothing_else() {
        // The document is also read and edited by a person, so formatting must
        // survive exactly.
        let source = "  -   [ ]   Keep   odd   spacing   \n- [ ] second\n";
        let mut list = TaskList::parse(source);
        let first = list.pending()[0].line;
        assert_eq!(list.mark_done(&[first]), 1);

        let rendered = list.render();
        assert_eq!(
            rendered,
            "  -   [x]   Keep   odd   spacing   \n- [ ] second\n"
        );
    }

    #[test]
    fn a_document_without_a_trailing_newline_keeps_that_shape() {
        let mut list = TaskList::parse("- [ ] one");
        list.mark_done(&[0]);
        assert_eq!(list.render(), "- [x] one");
    }

    #[test]
    fn ticking_an_already_done_task_reports_no_progress() {
        let mut list = TaskList::parse("- [x] done\n- [ ] todo\n");
        assert_eq!(list.mark_done(&[0]), 0, "already ticked is not progress");
        assert_eq!(list.mark_done(&[1]), 1);
        assert_eq!(list.mark_done(&[1]), 0, "ticking twice is not progress");
        assert_eq!(list.pending_count(), 0);
    }

    #[test]
    fn an_unknown_line_is_ignored_rather_than_corrupting_the_document() {
        let mut list = TaskList::parse("- [ ] one\n");
        assert_eq!(list.mark_done(&[99]), 0);
        assert_eq!(list.render(), "- [ ] one\n");
    }

    #[test]
    fn round_trip_without_ticking_is_byte_identical() {
        for source in [
            DOCUMENT,
            "no tasks here\n",
            "",
            "- [ ] a\r\n- [ ] b\r\n",
            "- [ ] no trailing newline",
            "\n\n- [ ] spaced\n\n",
            "mixed\r\n- [ ] one\n- [ ] two\r\n",
        ] {
            let list = TaskList::parse(source);
            assert_eq!(list.render(), source, "round trip changed {source:?}");
        }
    }

    #[test]
    fn ticking_a_crlf_document_changes_two_characters_and_keeps_crlf() {
        // Windows editors write CRLF. Normalising it would turn a tick into a
        // diff across the whole file.
        let source = "- [ ] one\r\n- [ ] two\r\n";
        let mut list = TaskList::parse(source);
        assert_eq!(list.mark_done(&[1]), 1);
        assert_eq!(list.render(), "- [ ] one\r\n- [x] two\r\n");
    }

    #[test]
    fn verifications_attach_to_the_task_above_them_in_both_forms() {
        let source = "\
- [ ] Add the score model
  verify-exists: src/model/score.ts
  verify-contains: src/model/score.ts :: export interface Score
- [ ] Unchecked task with no verification
";
        let list = TaskList::parse(source);
        assert_eq!(list.total(), 2);
        assert_eq!(
            list.tasks()[0].verifications,
            vec![
                Verification::Exists {
                    path: "src/model/score.ts".into()
                },
                Verification::Contains {
                    path: "src/model/score.ts".into(),
                    text: "export interface Score".into()
                },
            ]
        );
        assert!(list.tasks()[1].verifications.is_empty());
    }

    #[test]
    fn a_verification_written_as_a_bullet_or_in_backticks_is_still_read() {
        // Asked for an indented line under a bullet, a model often writes
        // another bullet, or quotes the form it was shown.
        let source = "\
- [ ] One
  - verify-exists: a.txt
- [ ] Two
  * `verify-contains: b.txt :: HELLO`
- [ ] Three
  VERIFY-EXISTS: c.txt
";
        let list = TaskList::parse(source);
        assert_eq!(
            list.tasks()[0].verifications,
            vec![Verification::Exists {
                path: "a.txt".into()
            }]
        );
        assert_eq!(
            list.tasks()[1].verifications,
            vec![Verification::Contains {
                path: "b.txt".into(),
                text: "HELLO".into()
            }]
        );
        assert_eq!(
            list.tasks()[2].verifications,
            vec![Verification::Exists {
                path: "c.txt".into()
            }]
        );
    }

    #[test]
    fn a_malformed_verification_is_ignored_rather_than_half_understood() {
        let source = "\
- [ ] One
  verify-exists:
  verify-contains: only-a-path.txt
  verify-contains:  :: only text
  verify-something-else: nope
";
        let list = TaskList::parse(source);
        assert!(
            list.tasks()[0].verifications.is_empty(),
            "{:?}",
            list.tasks()[0].verifications
        );
    }

    #[test]
    fn a_verification_before_any_task_is_discarded() {
        let list = TaskList::parse("verify-exists: stray.txt\n- [ ] One\n");
        assert!(list.tasks()[0].verifications.is_empty());
    }

    #[test]
    fn verifications_survive_a_round_trip_and_a_tick() {
        let source = "- [ ] One\n  verify-exists: a.txt\n";
        let mut list = TaskList::parse(source);
        assert_eq!(list.render(), source);
        list.mark_done(&[0]);
        assert_eq!(list.render(), "- [x] One\n  verify-exists: a.txt\n");
    }

    #[test]
    fn checks_pass_and_fail_against_a_real_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.txt"), "hello world").unwrap();

        let present = Verification::Exists {
            path: "src/a.txt".into(),
        };
        let absent = Verification::Exists {
            path: "src/missing.txt".into(),
        };
        assert!(present.check(dir.path()).is_ok());
        assert!(absent
            .check(dir.path())
            .unwrap_err()
            .contains("does not exist"));

        let holds = Verification::Contains {
            path: "src/a.txt".into(),
            text: "hello".into(),
        };
        let fails = Verification::Contains {
            path: "src/a.txt".into(),
            text: "goodbye".into(),
        };
        let no_file = Verification::Contains {
            path: "src/missing.txt".into(),
            text: "hello".into(),
        };
        assert!(holds.check(dir.path()).is_ok());
        assert!(fails
            .check(dir.path())
            .unwrap_err()
            .contains("does not contain"));
        assert!(no_file
            .check(dir.path())
            .unwrap_err()
            .contains("does not exist"));
    }

    #[test]
    fn failing_checks_reports_every_failure_not_just_the_first() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("there.txt"), "body").unwrap();
        let list = TaskList::parse(
            "- [ ] Two checks\n  verify-exists: gone.txt\n  verify-contains: there.txt :: absent\n",
        );

        let failures = list.tasks()[0].failing_checks(dir.path());
        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(failures[0].contains("gone.txt"));
        assert!(failures[1].contains("absent"));
    }

    #[test]
    fn a_task_whose_checks_all_pass_reports_no_failures() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "AAA").unwrap();
        let list = TaskList::parse(
            "- [ ] Fine\n  verify-exists: a.txt\n  verify-contains: a.txt :: AAA\n",
        );
        assert!(list.tasks()[0].failing_checks(dir.path()).is_empty());
    }

    #[test]
    fn a_task_without_checks_has_nothing_to_fail() {
        // Such a task can still be ticked, but only on the agent's word, which
        // the caller counts separately.
        let dir = tempfile::tempdir().unwrap();
        let list = TaskList::parse("- [ ] No checks here\n");
        assert!(list.tasks()[0].verifications.is_empty());
        assert!(list.tasks()[0].failing_checks(dir.path()).is_empty());
    }

    #[test]
    fn a_completion_report_is_read_from_the_last_marker() {
        // A model that restates its plan first must not have the plan counted.
        let response = "I will do COMPLETED: 1, 2, 3 eventually.\n\nWork done.\n\nCOMPLETED: 2, 4";
        assert_eq!(parse_completed_report(response, 5), vec![2, 4]);
    }

    #[test]
    fn completion_report_forms_are_accepted() {
        assert_eq!(parse_completed_report("COMPLETED: 1,2,3", 3), vec![1, 2, 3]);
        assert_eq!(parse_completed_report("COMPLETED: 1 and 3", 3), vec![1, 3]);
        assert_eq!(parse_completed_report("COMPLETED: all", 3), vec![1, 2, 3]);
        assert_eq!(
            parse_completed_report("COMPLETED: none", 3),
            Vec::<usize>::new()
        );
        assert_eq!(
            parse_completed_report("no marker at all", 3),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn numbers_outside_the_batch_are_discarded_not_clamped() {
        // Claiming task 9 of a 3-task batch identifies nothing, so it must not
        // silently tick task 3.
        assert_eq!(
            parse_completed_report("COMPLETED: 0, 4, 9", 3),
            Vec::<usize>::new()
        );
        assert_eq!(parse_completed_report("COMPLETED: 2, 7", 3), vec![2]);
    }

    #[test]
    fn digits_after_a_blank_line_do_not_leak_into_the_report() {
        let response = "COMPLETED: 1\n\nNext I plan to handle 2 and 3.";
        assert_eq!(parse_completed_report(response, 3), vec![1]);
    }

    #[test]
    fn duplicate_claims_are_counted_once() {
        assert_eq!(parse_completed_report("COMPLETED: 2, 2, 2", 3), vec![2]);
    }
}
