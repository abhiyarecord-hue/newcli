//! `state-store` (L3): persistent (SOUL/HEARTBEAT/MEMORY .md) + ephemeral state.

pub mod ephemeral;
pub mod persistent;

pub use ephemeral::EphemeralState;
pub use persistent::PersistentState;
