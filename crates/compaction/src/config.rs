//! Compaction configuration + the [`Compactor`] trait. Signatures verbatim
//! from plan.md section 3.

use agent_types::Message;

use crate::summary::ConversationSummary;

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
    /// Returns the structured summary of the dropped prefix and the retained
    /// messages. Pure function — no I/O.
    ///
    /// The summary is conversation data with explicit provenance, not a system
    /// prompt fragment, so a caller cannot accidentally grant compacted text
    /// policy priority.
    fn compact(
        &self,
        cfg: &CompactionConfig,
        history: &[Message],
    ) -> (ConversationSummary, Vec<Message>);
}
