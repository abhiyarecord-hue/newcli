//! `state-store` (L3): persistent (SOUL/HEARTBEAT/MEMORY .md) + ephemeral state + chat history.

pub mod ephemeral;
pub mod history;
pub mod persistent;

pub use ephemeral::EphemeralState;
pub use history::ChatHistory;
pub use persistent::PersistentState;
