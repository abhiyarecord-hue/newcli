//! Compaction configuration + the [`Compactor`] trait. Signatures verbatim
//! from plan.md section 3.

use agent_types::Message;

pub struct CompactionConfig {
    pub max_context_tokens: u32,
    pub keep_recent_messages: usize, // default = 4
    pub summary_target_tokens: u32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 8192,
            keep_recent_messages: 4,
            summary_target_tokens: 512,
        }
    }
}

pub trait Compactor: Send + Sync {
    /// Returns (new_system_suffix, retained_messages). Pure function — no I/O.
    fn compact(&self, cfg: &CompactionConfig, history: &[Message]) -> (String, Vec<Message>);
}
