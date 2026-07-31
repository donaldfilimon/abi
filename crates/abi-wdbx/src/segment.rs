//! Segment parse/read for the WDBX on-disk format.

use sha2::{Digest as _, Sha256};
use std::path::Path;

use crate::hash::{FormatError, Result};
use crate::record::Record;

/// Magic first line of a segment file.
pub const SEGMENT_HEADER: &str = "# ABI-WDBX v1";

/// Magic first line of the compatibility-mirror epoch sidecar.
pub const MIRROR_EPOCH_HEADER: &str = "# ABI-WDBX-MIRROR-EPOCH v1";

/// Prefix of the optional checksum trailer line.
pub const CHECKSUM_PREFIX: &str = "# checksum:";

/// The outcome of reading one segment file.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// The epoch this segment represents.
    pub epoch: u64,
    /// Records, in file order.
    pub records: Vec<Record>,
    /// The checksum from a `# checksum:` trailer, if present.
    pub checksum: Option<String>,
    /// Whether the final line was truncated and therefore dropped.
    ///
    /// Surfaced rather than swallowed so `wdbx_stats` and recovery can report that
    /// a write was interrupted.
    pub truncated_tail: bool,
}

/// Parse a segment body.
///
/// `path` is used only for error messages.
pub fn parse_segment(epoch: u64, path: &Path, content: &str) -> Result<Segment> {
    let mut lines = content.lines();

    let header = lines.next().unwrap_or_default();
    if header != SEGMENT_HEADER {
        return Err(FormatError::InvalidHeader {
            path: path.to_path_buf(),
            found: header.to_string(),
        });
    }

    verify_checksum(path, content)?;

    // A trailing newline means the last record line is complete. Without one, the
    // final line may be a torn write.
    let ends_cleanly = content.ends_with('\n');
    let body: Vec<&str> = lines.collect();

    let mut records = Vec::new();
    let mut checksum = None;
    let mut truncated_tail = false;

    for (index, line) in body.iter().enumerate() {
        let is_last = index + 1 == body.len();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix(CHECKSUM_PREFIX) {
            checksum = Some(rest.trim().to_string());
            continue;
        }
        match Record::parse(line) {
            Ok(record) => records.push(record),
            Err(err) => {
                // Only the final line of a file with no trailing newline may be
                // torn. A parse failure anywhere else is real corruption and must
                // not be silently dropped.
                if is_last && !ends_cleanly {
                    truncated_tail = true;
                } else {
                    return Err(err);
                }
            }
        }
    }

    Ok(Segment {
        epoch,
        records,
        checksum,
        truncated_tail,
    })
}

fn verify_checksum(path: &Path, content: &str) -> Result<()> {
    let marker = format!("\n{CHECKSUM_PREFIX}");
    let Some(marker_start) = content.rfind(&marker) else {
        return Ok(());
    };

    let expected_start = marker_start + marker.len();
    let expected = content[expected_start..].trim();
    let body_start = SEGMENT_HEADER.len() + 1;
    let body_end = marker_start + 1;
    let actual = hex_digest(Sha256::digest(&content.as_bytes()[body_start..body_end]));

    if expected != actual {
        return Err(FormatError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

pub(crate) fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    use std::fmt::Write as _;
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Read and parse a segment file.
pub fn read_segment(epoch: u64, path: &Path) -> Result<Segment> {
    let content = std::fs::read_to_string(path).map_err(|e| FormatError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    parse_segment(epoch, path, &content)
}
