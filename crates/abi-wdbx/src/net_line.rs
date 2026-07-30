//! Bounded newline framing shared by local WDBX TCP transports.

use std::io::{Read, Write};

/// Largest cluster/compute line frame.
pub const MAX_LINE_SIZE: usize = 4 * 1024;

/// A bounded line-frame failure.
#[derive(Debug)]
pub enum LineError {
    /// Underlying stream failure.
    Io(std::io::Error),
    /// The caller's buffer filled before a newline arrived.
    TooLong,
}

impl std::fmt::Display for LineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::TooLong => formatter.write_str("line frame too long"),
        }
    }
}

impl std::error::Error for LineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::TooLong => None,
        }
    }
}

impl From<std::io::Error> for LineError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Read one line, stripping LF and an optional preceding CR.
///
/// EOF before a newline returns the partial frame, leaving semantic rejection
/// to the protocol parser. Filling `buffer` without a newline is a hard bound.
pub fn read_line<'buffer, R: Read>(
    reader: &mut R,
    buffer: &'buffer mut [u8],
) -> Result<&'buffer [u8], LineError> {
    let mut total = 0;
    while total < buffer.len() {
        let read = reader.read(&mut buffer[total..])?;
        if read == 0 {
            break;
        }
        total += read;
        if let Some(end) = buffer[..total].iter().position(|byte| *byte == b'\n') {
            let end = end - usize::from(end > 0 && buffer[end - 1] == b'\r');
            return Ok(&buffer[..end]);
        }
    }
    if total == buffer.len() {
        return Err(LineError::TooLong);
    }
    let end = total - usize::from(total > 0 && buffer[total - 1] == b'\r');
    Ok(&buffer[..end])
}

/// Write and flush one already-framed message.
pub fn write_line<W: Write>(writer: &mut W, bytes: &[u8]) -> std::io::Result<()> {
    writer.write_all(bytes)?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_crlf_and_returns_partial_eof_frames() {
        let mut buffer = [0; 32];
        assert_eq!(
            read_line(&mut &b"hello\r\nignored"[..], &mut buffer).expect("line"),
            b"hello"
        );
        assert_eq!(
            read_line(&mut &b"partial"[..], &mut buffer).expect("partial"),
            b"partial"
        );
    }

    #[test]
    fn rejects_a_frame_that_fills_the_buffer_without_newline() {
        let mut buffer = [0; 4];
        assert!(matches!(
            read_line(&mut &b"abcd"[..], &mut buffer),
            Err(LineError::TooLong)
        ));
    }
}
