//! Ephemeral in-memory state: pending tasks + ring buffer of tool outputs.
//! Wiped after task completion.

use std::collections::VecDeque;

const MAX_TOOL_OUTPUTS: usize = 500;

#[derive(Clone, Debug)]
pub struct PendingTask {
    pub description: String,
    pub done: bool,
}

pub struct EphemeralState {
    pending_tasks: VecDeque<PendingTask>,
    tool_outputs: VecDeque<String>,
}

impl EphemeralState {
    pub fn new() -> Self {
        Self {
            pending_tasks: VecDeque::new(),
            tool_outputs: VecDeque::with_capacity(MAX_TOOL_OUTPUTS),
        }
    }

    pub fn add_task(&mut self, description: impl Into<String>) {
        self.pending_tasks.push_back(PendingTask {
            description: description.into(),
            done: false,
        });
    }

    pub fn complete_task(&mut self, idx: usize) {
        if let Some(task) = self.pending_tasks.get_mut(idx) {
            task.done = true;
        }
    }

    pub fn pending_tasks(&self) -> &VecDeque<PendingTask> {
        &self.pending_tasks
    }

    pub fn push_tool_output(&mut self, output: impl Into<String>) {
        if self.tool_outputs.len() >= MAX_TOOL_OUTPUTS {
            self.tool_outputs.pop_front();
        }
        self.tool_outputs.push_back(output.into());
    }

    pub fn recent_outputs(&self, n: usize) -> Vec<&str> {
        self.tool_outputs
            .iter()
            .rev()
            .take(n)
            .map(|s| s.as_str())
            .collect()
    }

    /// Wipe all ephemeral state (call after task completion).
    pub fn wipe(&mut self) {
        self.pending_tasks.clear();
        self.tool_outputs.clear();
    }
}

impl Default for EphemeralState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_caps_at_500() {
        let mut state = EphemeralState::new();
        for i in 0..600 {
            state.push_tool_output(format!("output {i}"));
        }
        assert_eq!(state.tool_outputs.len(), MAX_TOOL_OUTPUTS);
        // Oldest should be output 100 (600-500).
        assert_eq!(state.tool_outputs.front().unwrap(), "output 100");
    }

    #[test]
    fn wipe_clears_everything() {
        let mut state = EphemeralState::new();
        state.add_task("do something");
        state.push_tool_output("result");
        state.wipe();
        assert!(state.pending_tasks.is_empty());
        assert!(state.tool_outputs.is_empty());
    }
}
