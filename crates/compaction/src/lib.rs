//! `compaction` (L1): session compaction / context-window management.
//!
//! - [`token_count`]: deterministic token heuristic (no tokenizer dep).
//! - [`config`]: [`CompactionConfig`] + [`Compactor`] trait.
//! - [`compactor`]: [`ThresholdCompactor`] — keep-recent + structured summary.

pub mod compactor;
pub mod config;
pub mod summary;
pub mod token_count;

pub use compactor::{is_summary_block, ThresholdCompactor};
pub use config::{CompactionConfig, Compactor};
pub use summary::{
    ConversationSummary, MAX_ENTRIES_PER_CATEGORY, UNTRUSTED_SUMMARY_CLOSE, UNTRUSTED_SUMMARY_OPEN,
};
pub use token_count::{estimate_history, estimate_message, estimate_tokens};
