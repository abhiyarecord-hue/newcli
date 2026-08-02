//! Bounded, streaming child-output capture.
//!
//! Readers continue draining after the configured limit so a child cannot block
//! on a full pipe, but bytes beyond the limit are discarded immediately.

use tokio::io::{AsyncRead, AsyncReadExt};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedOutput {
    pub text: String,
    pub truncated: bool,
}

pub async fn capture_bounded<R>(mut reader: R, limit: usize) -> std::io::Result<CapturedOutput>
where
    R: AsyncRead + Unpin,
{
    let mut retained = Vec::with_capacity(limit.min(8 * 1024));
    let mut chunk = [0_u8; 8 * 1024];
    let mut truncated = false;

    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            break;
        }
        let remaining = limit.saturating_sub(retained.len());
        let keep = remaining.min(count);
        retained.extend_from_slice(&chunk[..keep]);
        truncated |= keep < count;
    }

    let mut text = String::from_utf8_lossy(&retained).into_owned();
    if text.len() > limit {
        let boundary = floor_char_boundary(&text, limit);
        text.truncate(boundary);
        truncated = true;
    }
    Ok(CapturedOutput { text, truncated })
}

fn floor_char_boundary(value: &str, mut index: usize) -> usize {
    index = index.min(value.len());
    while index > 0 && !value.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn captures_prefix_and_reports_truncation_while_draining() {
        let (mut writer, reader) = tokio::io::duplex(8);
        let write = tokio::spawn(async move {
            writer.write_all(b"stdout-flood").await.unwrap();
        });
        let captured = capture_bounded(reader, 6).await.unwrap();
        write.await.unwrap();
        assert_eq!(captured.text, "stdout");
        assert!(captured.truncated);
    }

    #[tokio::test]
    async fn exact_limit_is_not_truncated() {
        let (mut writer, reader) = tokio::io::duplex(8);
        writer.write_all(b"1234").await.unwrap();
        drop(writer);
        assert_eq!(
            capture_bounded(reader, 4).await.unwrap(),
            CapturedOutput {
                text: "1234".into(),
                truncated: false
            }
        );
    }

    proptest! {
        // **Validates: Requirements 2.8**
        #[test]
        fn property_retained_bytes_never_exceed_the_bound(
            bytes in proptest::collection::vec(any::<u8>(), 0..4096),
            limit in 0usize..256,
        ) {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let captured = runtime.block_on(async {
                let (mut writer, reader) = tokio::io::duplex(64);
                let write = tokio::spawn(async move { writer.write_all(&bytes).await.unwrap() });
                let captured = capture_bounded(reader, limit).await.unwrap();
                write.await.unwrap();
                captured
            });
            prop_assert!(captured.text.len() <= limit);
        }
    }
}
