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

/// A checkbox line in the task document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    /// Zero-based index of the line this task occupies.
    pub line: usize,
    pub done: bool,
    /// The task text, with the list marker and checkbox removed.
    pub text: String,
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

        let mut tasks = Vec::new();
        for (index, line) in lines.iter().enumerate() {
            if let Some((done, text)) = parse_checkbox(&line.text) {
                tasks.push(Task {
                    line: index,
                    done,
                    text,
                });
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
