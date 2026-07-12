//! Pure, incremental Server-Sent-Events parser. NO network code here.
//!
//! SSE frames arrive fragmented across TCP reads, so [`SseParser::feed`] never
//! assumes it receives whole frames: any unterminated tail is retained in the
//! internal buffer until the next call. `\r\n` and lone `\r` are normalized to
//! `\n` before splitting (handled even when a `\r\n` straddles two `feed`
//! calls). The buffer is capped at 1 MiB — a server that streams forever
//! without a frame terminator is rejected rather than exhausting memory.

use agent_types::{AgentError, Result};

/// Hard cap on the retained buffer (malicious-server DoS guard).
const MAX_BUF: usize = 1024 * 1024;

/// One raw SSE frame: an optional `event:` name and the joined `data:` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawSseFrame {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Default)]
pub struct SseParser {
    /// Normalized (LF-only) bytes not yet consumed into a frame.
    buf: Vec<u8>,
    /// A trailing `\r` seen at the end of the previous chunk, held back so a
    /// `\r\n` split across two `feed` calls collapses to a single `\n`.
    pending_cr: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            pending_cr: false,
        }
    }

    /// Feed the next chunk of bytes. Returns every frame now complete; the
    /// unterminated remainder stays buffered. Errors only on the 1 MiB cap.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<RawSseFrame>> {
        self.push_normalized(bytes);

        if self.buf.len() > MAX_BUF {
            // Drop the buffer so a caller that ignores the error and keeps
            // feeding does not keep re-triggering on stale bytes.
            self.buf.clear();
            self.pending_cr = false;
            return Err(AgentError::Llm(
                "sse buffer exceeded 1 MiB without a frame boundary".to_string(),
            ));
        }

        let mut frames = Vec::new();
        while let Some(pos) = find_double_newline(&self.buf) {
            // Bytes before the blank-line separator are the frame block.
            let block: Vec<u8> = self.buf.drain(..pos + 2).collect();
            if let Some(frame) = parse_block(&block[..pos]) {
                frames.push(frame);
            }
        }
        Ok(frames)
    }

    /// Append `bytes`, converting `\r\n` and lone `\r` to `\n`. A `\r` at the
    /// very end is held in `pending_cr` until the next byte is known.
    fn push_normalized(&mut self, bytes: &[u8]) {
        for &b in bytes {
            match (self.pending_cr, b) {
                (true, b'\n') => {
                    self.buf.push(b'\n');
                    self.pending_cr = false;
                }
                (true, b'\r') => {
                    // Previous lone CR becomes LF; this CR stays pending.
                    self.buf.push(b'\n');
                }
                (true, other) => {
                    self.buf.push(b'\n');
                    self.buf.push(other);
                    self.pending_cr = false;
                }
                (false, b'\r') => {
                    self.pending_cr = true;
                }
                (false, other) => {
                    self.buf.push(other);
                }
            }
        }
    }
}

/// Index of the first `\n\n` (blank line) in `buf`, if any.
fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

/// Parse a single frame block (the bytes before the `\n\n` separator).
///
/// Field lines follow the SSE grammar: `field: value` (one optional leading
/// space stripped from the value), lines starting with `:` are comments, and a
/// line with no colon is a field with an empty value. Only `event` and `data`
/// are meaningful here; multiple `data:` lines join with `\n`.
fn parse_block(block: &[u8]) -> Option<RawSseFrame> {
    let text = String::from_utf8_lossy(block);
    let mut event: Option<String> = None;
    let mut data_lines: Vec<String> = Vec::new();

    for line in text.split('\n') {
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => event = Some(value.to_string()),
            "data" => data_lines.push(value.to_string()),
            _ => {} // id / retry / unknown fields ignored
        }
    }

    if event.is_none() && data_lines.is_empty() {
        return None;
    }
    Some(RawSseFrame {
        event,
        data: data_lines.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_frame_one_feed() {
        let mut p = SseParser::new();
        let frames = p.feed(b"event: delta\ndata: {\"x\":1}\n\n").unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("delta"));
        assert_eq!(frames[0].data, "{\"x\":1}");
    }

    #[test]
    fn frame_split_across_two_feeds() {
        let mut p = SseParser::new();
        let first = p.feed(b"event: delta\ndata: {\"x").unwrap();
        assert!(first.is_empty(), "no complete frame yet");
        let second = p.feed(b"\":1}\n\n").unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].data, "{\"x\":1}");
    }

    #[test]
    fn multi_line_data_joined_with_newline() {
        let mut p = SseParser::new();
        let frames = p.feed(b"data: line1\ndata: line2\n\n").unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "line1\nline2");
    }

    #[test]
    fn crlf_is_normalized_including_across_feeds() {
        let mut p = SseParser::new();
        // Split the CRLF terminator across two feeds: "...}\r" then "\n\r\n".
        assert!(p.feed(b"data: hi\r").unwrap().is_empty());
        let frames = p.feed(b"\n\r\n").unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "hi");
    }

    #[test]
    fn comment_and_unknown_fields_ignored() {
        let mut p = SseParser::new();
        let frames = p
            .feed(b": keep-alive\nid: 7\nevent: ping\ndata: ok\n\n")
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("ping"));
        assert_eq!(frames[0].data, "ok");
    }

    #[test]
    fn buffer_cap_returns_llm_error() {
        let mut p = SseParser::new();
        let big = vec![b'a'; MAX_BUF + 1];
        let err = p.feed(&big).unwrap_err();
        assert!(matches!(err, AgentError::Llm(_)));
    }

    #[test]
    fn two_frames_in_one_feed() {
        let mut p = SseParser::new();
        let frames = p.feed(b"data: a\n\ndata: b\n\n").unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].data, "a");
        assert_eq!(frames[1].data, "b");
    }
}
