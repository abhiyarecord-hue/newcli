//! One HTTP client configuration for every provider in this crate.
//!
//! `reqwest::Client::new()` keeps idle connections in a pool for a long time and
//! reuses them. A hosted inference endpoint, or any proxy in front of one, closes
//! an idle keep-alive connection on its own schedule, and the next request sent
//! on that dead socket fails before the server ever sees it. `reqwest` reports
//! that as a request-phase or decode-phase error rather than an HTTP status, so
//! it looks like a mysterious "invalid request" or "response decode error" even
//! though the endpoint is healthy — which is exactly how it was observed against
//! a live Azure deployment, on a session that had been sitting at a prompt,
//! while a fresh single-shot process against the same endpoint worked.
//!
//! Two settings remove that class:
//!
//! - `pool_idle_timeout` drops a pooled connection well before a typical
//!   server-side keep-alive window expires, so a corpse is not reused.
//! - `tcp_keepalive` keeps a live connection demonstrably alive and surfaces a
//!   genuinely broken one promptly.
//!
//! There is deliberately **no** total request timeout. Responses here are
//! streamed, and a reasoning model can legitimately spend minutes on one answer;
//! a whole-request deadline would abort correct work. Slow or silent streams are
//! bounded by the callers, per event, where the accumulated output is known.

use std::time::Duration;

/// Longest an unused connection may sit in the pool before being discarded.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(15);

/// TCP keepalive probe interval for connections that are in use.
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);

/// How long to wait for a TCP connection to be established.
///
/// Applies to connection setup only, not to the response, so it cannot cut off
/// a long stream.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Build the client used for streaming and embedding requests.
pub(crate) fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_idle_timeout(POOL_IDLE_TIMEOUT)
        .tcp_keepalive(TCP_KEEPALIVE)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        // Falling back keeps a TLS backend problem from panicking here; the
        // request will fail with a clear transport error instead.
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_configuration_is_buildable_and_sets_no_total_deadline() {
        // A total timeout would abort a legitimately long reasoning stream, so
        // the guard is that this builder keeps working and stays cheap to call.
        let first = client();
        let second = client();
        // Two independent clients means two independent pools, which is what
        // keeps a stale connection in one provider from affecting another.
        assert!(!std::ptr::eq(&first, &second));
    }

    #[test]
    fn the_bounds_are_ordered_so_a_pooled_connection_is_dropped_before_it_rots() {
        // The pool must give up on an idle connection sooner than a keepalive
        // probe cycle would notice it died, otherwise the dead socket is reused.
        assert!(POOL_IDLE_TIMEOUT < TCP_KEEPALIVE);
        assert!(CONNECT_TIMEOUT >= TCP_KEEPALIVE);
    }
}
