//! Bounded, incremental raw-byte Server-Sent Events framing.
//!
//! Bytes are retained without decoding until a complete `\n\n` or
//! `\r\n\r\n` boundary is present. Only then is the frame decoded as strict
//! UTF-8 and parsed into SSE fields. Call [`SseParser::finish`] at EOF so an
//! incomplete final frame cannot be mistaken for successful completion.

use agent_types::{AgentError, Result};

/// Public production ceiling for one raw SSE frame (8 MiB).
pub const DEFAULT_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// One complete SSE frame: an optional `event:` name and joined `data:` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawSseFrame {
    pub event: Option<String>,
    pub data: String,
}

pub struct SseParser {
    buf: Vec<u8>,
    max_frame_bytes: usize,
    failed: bool,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    /// Create a parser with the bounded production default of 8 MiB per frame.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            failed: false,
        }
    }

    /// Create a parser with a smaller injected frame bound.
    ///
    /// Zero and values above the production ceiling are rejected, so callers
    /// cannot accidentally configure an unbounded parser.
    pub fn with_max_frame_bytes(max_frame_bytes: usize) -> Result<Self> {
        if max_frame_bytes == 0 || max_frame_bytes > DEFAULT_MAX_FRAME_BYTES {
            return Err(AgentError::Llm(format!(
                "sse frame limit must be between 1 and {DEFAULT_MAX_FRAME_BYTES} bytes"
            )));
        }
        Ok(Self {
            buf: Vec::new(),
            max_frame_bytes,
            failed: false,
        })
    }

    /// Feed raw transport bytes and return all newly completed frames.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<RawSseFrame>> {
        if self.failed {
            return Err(AgentError::Llm(
                "sse parser is in a failed state".to_string(),
            ));
        }
        self.buf.extend_from_slice(bytes);

        let mut frames = Vec::new();
        while let Some((position, separator_len)) = find_frame_boundary(&self.buf) {
            if position > self.max_frame_bytes {
                return self.fail(format!("sse frame exceeded {} bytes", self.max_frame_bytes));
            }
            let consumed: Vec<u8> = self.buf.drain(..position + separator_len).collect();
            if let Some(frame) = parse_block(&consumed[..position])? {
                frames.push(frame);
            }
        }

        if buffered_payload_exceeds_limit(&self.buf, self.max_frame_bytes) {
            return self.fail(format!(
                "sse frame exceeded {} bytes without a frame boundary",
                self.max_frame_bytes
            ));
        }
        Ok(frames)
    }

    /// Validate EOF.
    ///
    /// Trailing whitespace (common from proxies/routers) is tolerated, but any
    /// other retained bytes are an incomplete final frame and are rejected so a
    /// truncated stream can never be reported as successful completion.
    pub fn finish(&mut self) -> Result<()> {
        if self.failed {
            return Err(AgentError::Llm(
                "sse parser is in a failed state".to_string(),
            ));
        }
        if self.buf.iter().all(|byte| byte.is_ascii_whitespace()) {
            self.buf.clear();
            return Ok(());
        }
        let retained = self.buf.len();
        self.fail(format!(
            "sse stream ended with an incomplete frame of {retained} retained bytes"
        ))
    }

    fn fail<T>(&mut self, message: String) -> Result<T> {
        self.buf.clear();
        self.failed = true;
        Err(AgentError::Llm(message))
    }
}

fn find_frame_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let lf = buf
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|p| (p, 2));
    let crlf = buf
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|p| (p, 4));
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(boundary), None) | (None, Some(boundary)) => Some(boundary),
        (None, None) => None,
    }
}

fn buffered_payload_exceeds_limit(buf: &[u8], limit: usize) -> bool {
    if buf.len() <= limit {
        return false;
    }
    let suffix = &buf[limit..];
    !b"\n\n".starts_with(suffix) && !b"\r\n\r\n".starts_with(suffix)
}

fn parse_block(block: &[u8]) -> Result<Option<RawSseFrame>> {
    let text = std::str::from_utf8(block)
        .map_err(|error| AgentError::Llm(format!("sse frame is not valid UTF-8: {error}")))?;
    let mut event = None;
    let mut data_lines = Vec::new();

    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => event = Some(value.to_string()),
            "data" => data_lines.push(value.to_string()),
            _ => {}
        }
    }

    if event.is_none() && data_lines.is_empty() {
        return Ok(None);
    }
    Ok(Some(RawSseFrame {
        event,
        data: data_lines.join("\n"),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lf_and_crlf_frames_without_changing_order() {
        let mut parser = SseParser::new();
        let frames = parser
            .feed(b"event: delta\ndata: one\n\ndata: two\r\n\r\n")
            .unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].event.as_deref(), Some("delta"));
        assert_eq!(frames[0].data, "one");
        assert_eq!(frames[1].data, "two");
        parser.finish().unwrap();
    }

    #[test]
    fn every_unicode_and_boundary_byte_split_is_invariant() {
        // **Validates: Requirements 2.37**
        for sample in [
            "data: {\"text\":\"ह😀\"}\n\n",
            "event: delta\r\ndata: {\"text\":\"ह😀\"}\r\n\r\n",
        ] {
            let expected: serde_json::Value = serde_json::from_str("{\"text\":\"ह😀\"}").unwrap();
            for split in 1..sample.len() {
                let mut parser = SseParser::new();
                assert!(parser.feed(&sample.as_bytes()[..split]).unwrap().is_empty());
                let frames = parser.feed(&sample.as_bytes()[split..]).unwrap();
                assert_eq!(frames.len(), 1, "split {split} for {sample:?}");
                let actual: serde_json::Value = serde_json::from_str(&frames[0].data).unwrap();
                assert_eq!(actual, expected, "split {split} for {sample:?}");
                parser.finish().unwrap();
            }
        }
    }

    #[test]
    fn invalid_utf8_is_rejected_only_after_the_frame_is_complete() {
        // **Validates: Requirements 2.37**
        let mut parser = SseParser::with_max_frame_bytes(32).unwrap();
        assert!(parser.feed(b"data: \xF0\x9F").unwrap().is_empty());
        let error = parser.feed(b"\n\n").unwrap_err();
        assert!(matches!(error, AgentError::Llm(message) if message.contains("UTF-8")));
    }

    #[test]
    fn exact_bound_is_accepted_and_one_byte_over_is_rejected() {
        // **Validates: Requirements 2.37**
        let mut exact = SseParser::with_max_frame_bytes(8).unwrap();
        assert_eq!(exact.feed(b"data: ab\n\n").unwrap()[0].data, "ab");

        let mut oversized = SseParser::with_max_frame_bytes(8).unwrap();
        let error = oversized.feed(b"data: abc\n\n").unwrap_err();
        assert!(matches!(error, AgentError::Llm(message) if message.contains("exceeded 8")));
    }

    #[test]
    fn delimiter_prefix_does_not_make_an_exact_size_frame_oversized() {
        let mut parser = SseParser::with_max_frame_bytes(8).unwrap();
        assert!(parser.feed(b"data: ab\r").unwrap().is_empty());
        assert!(parser.feed(b"\n\r").unwrap().is_empty());
        assert_eq!(parser.feed(b"\n").unwrap()[0].data, "ab");
    }

    #[test]
    fn eof_rejects_incomplete_frame_and_limit_cannot_be_unbounded() {
        // **Validates: Requirements 2.37**
        let mut parser = SseParser::with_max_frame_bytes(16).unwrap();
        parser.feed(b"data: partial").unwrap();
        let error = parser.finish().unwrap_err();
        assert!(matches!(error, AgentError::Llm(message) if message.contains("incomplete")));
        assert!(SseParser::with_max_frame_bytes(0).is_err());
        assert!(SseParser::with_max_frame_bytes(DEFAULT_MAX_FRAME_BYTES + 1).is_err());
    }

    #[test]
    fn multiple_data_lines_and_comments_follow_sse_field_rules() {
        let mut parser = SseParser::new();
        let frames = parser
            .feed(b": keep-alive\nid: 7\nevent: ping\ndata: line1\ndata: line2\n\n")
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("ping"));
        assert_eq!(frames[0].data, "line1\nline2");
    }
}
