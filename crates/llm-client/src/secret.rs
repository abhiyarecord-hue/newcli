//! Secret-safe request construction and error rendering for provider adapters.
//!
//! Credentials are supplied through sensitive headers, never URL query strings,
//! so proxies, access logs, and error text cannot capture them. Error rendering
//! is restricted to a sanitized endpoint, the HTTP status, and a bounded
//! redacted body.

use agent_types::{AgentError, Result};
use reqwest::header::{HeaderName, HeaderValue};
use reqwest::{Response, Url};

/// Header Google accepts for API-key authentication.
pub(crate) const API_KEY_HEADER: &str = "x-goog-api-key";

/// Maximum number of body scalars rendered in an error message.
const MAX_ERROR_BODY_SCALARS: usize = 2048;

/// Maximum number of body bytes read while rendering an error.
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

const REDACTED: &str = "[redacted]";

/// Build the sensitive API-key header.
///
/// `HeaderValue::set_sensitive` keeps the credential out of `reqwest`'s
/// `Debug`/tracing output. The key itself is never echoed in the error.
pub(crate) fn api_key_header(api_key: &str) -> Result<(HeaderName, HeaderValue)> {
    let name = HeaderName::from_static(API_KEY_HEADER);
    let mut value = HeaderValue::from_str(api_key).map_err(|_| {
        AgentError::Llm("api key contains characters invalid in an HTTP header".into())
    })?;
    value.set_sensitive(true);
    Ok((name, value))
}

/// Render only the scheme, host, port, and path of an endpoint.
///
/// Query strings, fragments, and userinfo are dropped because they are the
/// places credentials historically leaked from.
pub(crate) fn sanitized_endpoint(url: &str) -> String {
    match Url::parse(url) {
        Ok(parsed) => {
            let scheme = parsed.scheme();
            let mut rendered = match parsed.host_str() {
                Some(host) => format!("{scheme}://{host}"),
                None => format!("{scheme}://"),
            };
            if let Some(port) = parsed.port() {
                rendered.push_str(&format!(":{port}"));
            }
            rendered.push_str(parsed.path());
            rendered
        }
        // An unparseable endpoint may be arbitrary text, so expose nothing.
        Err(_) => "<invalid endpoint>".to_string(),
    }
}

/// Replace every occurrence of the credential and bound the scalar count.
pub(crate) fn bounded_redacted_body(body: &str, api_key: &str) -> String {
    let redacted = redact(body, api_key);
    let mut bounded: String = redacted.chars().take(MAX_ERROR_BODY_SCALARS).collect();
    if redacted.chars().count() > MAX_ERROR_BODY_SCALARS {
        bounded.push_str("...[truncated]");
    }
    bounded
}

fn redact(body: &str, api_key: &str) -> String {
    if api_key.is_empty() {
        return body.to_string();
    }
    body.replace(api_key, REDACTED)
}

/// Read at most [`MAX_ERROR_BODY_BYTES`] of an error response body.
pub(crate) async fn read_bounded_body(mut response: Response) -> String {
    let mut collected: Vec<u8> = Vec::new();
    while collected.len() < MAX_ERROR_BODY_BYTES {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let remaining = MAX_ERROR_BODY_BYTES - collected.len();
                let take = remaining.min(chunk.len());
                collected.extend_from_slice(&chunk[..take]);
                if take < chunk.len() {
                    break;
                }
            }
            Ok(None) => break,
            // A failed error-body read must not replace the status information.
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&collected).into_owned()
}

/// Describe a transport failure without rendering the underlying URL.
///
/// `reqwest::Error`'s own `Display` includes the request URL, so only a
/// category is derived here and the sanitized endpoint is supplied separately.
pub(crate) fn transport_error_kind(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connection failure"
    } else if error.is_redirect() {
        "too many redirects"
    } else if error.is_request() {
        "invalid request"
    } else if error.is_body() {
        "response body error"
    } else if error.is_decode() {
        "response decode error"
    } else {
        "transport error"
    }
}

/// Build a secret-free transport error message.
pub(crate) fn transport_error(context: &str, endpoint: &str, error: &reqwest::Error) -> AgentError {
    AgentError::Llm(format!(
        "{context} to {}: {}",
        sanitized_endpoint(endpoint),
        transport_error_kind(error)
    ))
}

/// Build a secret-free non-success status error message.
pub(crate) fn status_error(
    context: &str,
    endpoint: &str,
    status: reqwest::StatusCode,
    body: &str,
    api_key: &str,
) -> AgentError {
    AgentError::Llm(format!(
        "{context} to {} failed with http {status}: {}",
        sanitized_endpoint(endpoint),
        bounded_redacted_body(body, api_key)
    ))
}

/// Deterministic loopback HTTP capture used by provider adapter tests.
///
/// No live credentials or network access are required: the fixture binds an
/// ephemeral loopback port, records exactly one request, and replies with a
/// canned response.
#[cfg(test)]
pub(crate) mod testing {
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    /// Parse the declared `content-length` of a captured request head.
    fn content_length(head: &str) -> usize {
        head.lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.trim()
                    .eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0)
    }

    /// Return the request head, excluding any body.
    pub(crate) fn request_head(request: &str) -> String {
        request
            .split_once("\r\n\r\n")
            .map(|(head, _)| head.to_string())
            .unwrap_or_else(|| request.to_string())
    }

    /// Bind a loopback server that captures one request and returns `body`.
    pub(crate) async fn spawn_http_capture(
        status: &'static str,
        content_type: &'static str,
        body: Vec<u8>,
        timeout: Duration,
    ) -> (String, JoinHandle<Option<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base_url = format!("http://{}/v1beta", listener.local_addr().expect("addr"));

        let handle = tokio::spawn(async move {
            let accepted = tokio::time::timeout(timeout, listener.accept()).await;
            let (mut stream, _) = match accepted {
                Ok(Ok(pair)) => pair,
                _ => return None,
            };

            // Read the head, then drain exactly `content-length` body bytes so
            // the client's write completes and the connection is not reset.
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            loop {
                match tokio::time::timeout(timeout, stream.read(&mut buffer)).await {
                    Ok(Ok(0)) => break,
                    Ok(Ok(read)) => {
                        request.extend_from_slice(&buffer[..read]);
                        let text = String::from_utf8_lossy(&request);
                        if let Some((head, body)) = text.split_once("\r\n\r\n") {
                            let expected = content_length(head);
                            if body.len() >= expected {
                                break;
                            }
                        }
                    }
                    _ => break,
                }
            }

            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.write_all(&body).await;
            let _ = stream.flush().await;
            let _ = stream.shutdown().await;

            Some(String::from_utf8_lossy(&request).into_owned())
        });

        (base_url, handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_header_is_marked_sensitive_and_omits_the_key_on_error() {
        let (name, value) = api_key_header("secret-key").unwrap();
        assert_eq!(name.as_str(), API_KEY_HEADER);
        assert!(value.is_sensitive());
        assert_eq!(value.to_str().unwrap(), "secret-key");
        // Sensitive values must not be rendered by Debug output.
        assert!(!format!("{value:?}").contains("secret-key"));

        let rejected = api_key_header("bad\nkey").unwrap_err();
        assert!(!rejected.to_string().contains("bad"));
    }

    #[test]
    fn sanitized_endpoint_drops_query_userinfo_and_fragment() {
        assert_eq!(
            sanitized_endpoint("https://example.com/v1/models/m:generate?alt=sse&key=SECRET"),
            "https://example.com/v1/models/m:generate"
        );
        assert_eq!(
            sanitized_endpoint("https://user:SECRET@example.com:8443/v1/x#SECRET"),
            "https://example.com:8443/v1/x"
        );
        assert_eq!(sanitized_endpoint("not a url SECRET"), "<invalid endpoint>");
    }

    #[test]
    fn bodies_are_redacted_and_bounded() {
        let body = "error for key SECRET-KEY and SECRET-KEY again";
        let rendered = bounded_redacted_body(body, "SECRET-KEY");
        assert!(!rendered.contains("SECRET-KEY"));
        assert_eq!(rendered.matches(REDACTED).count(), 2);

        let long = "😀".repeat(MAX_ERROR_BODY_SCALARS + 10);
        let bounded = bounded_redacted_body(&long, "unused");
        assert_eq!(
            bounded.chars().filter(|c| *c == '😀').count(),
            MAX_ERROR_BODY_SCALARS
        );
        assert!(bounded.ends_with("...[truncated]"));

        // An empty key must not turn every position into a redaction marker.
        assert_eq!(bounded_redacted_body("plain", ""), "plain");
    }

    #[test]
    fn status_error_renders_only_sanitized_endpoint_status_and_bounded_body() {
        let error = status_error(
            "embedding request",
            "https://example.com/v1/models/m:embedContent?key=SECRET",
            reqwest::StatusCode::FORBIDDEN,
            "denied for SECRET",
            "SECRET",
        );
        let message = error.to_string();
        assert!(!message.contains("SECRET"));
        assert!(message.contains("https://example.com/v1/models/m:embedContent"));
        assert!(message.contains("403"));
        assert!(!message.contains("key="));
    }
}
