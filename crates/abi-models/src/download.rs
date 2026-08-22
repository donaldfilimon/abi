//! Resumable download plumbing.
//!
//! This module owns the byte-range / partial-file state machine and nothing
//! else. Bytes arrive through the [`ChunkTransport`] trait, so the whole
//! machine — first attempt, interrupted attempt, resumed attempt, corrupted
//! resume — is exercised without a network and without an HTTP dependency.
//!
//! ## The state machine
//!
//! ```text
//! destination exists ──► verify ──► AlreadyPresent | error
//!         │ no
//!         ▼
//! partial file length = resume offset
//!         │
//!         ├─ offset > declared size ──► ResumeBeyondEnd (refuse)
//!         ├─ offset = declared size ──► skip to verify
//!         └─ offset < declared size ──► fetch(url, offset, remaining), append, repeat
//!                                        │
//!                                        ▼
//!                            verify partial: size, then SHA-256, then fsync
//!                                        │
//!                     mismatch ──► delete partial, error
//!                        ok     ──► atomically link partial to destination
//! ```
//!
//! Two properties are load-bearing:
//!
//! - Bytes are appended to a `.part` file and only renamed into place once the
//!   whole file verifies, so a destination path never holds a half-written or
//!   unverified artifact.
//! - A partial file that fails verification is **deleted**. Leaving it would
//!   make every later resume start from poisoned bytes and fail identically
//!   forever; deleting it means the next attempt is a clean retry.
//!
//! ## Limitations
//!
//! Named rather than left implicit:
//!
//! - **Concurrent downloads of the same destination are not supported.** There
//!   is no cross-process lock on the `.part` file, so two runs would interleave
//!   appends. The final hash check turns that into a loud failure rather than
//!   silent corruption, but it is not a supported mode.
//!
//! Custom transports are responsible for their own range semantics. The shipped
//! [`HttpTransport`] requires an exact `206 Content-Range`, refuses redirects,
//! and bounds each response before returning bytes to this state machine.

use crate::error::ModelError;
use crate::manifest::Artifact;
use crate::verify::hash_file;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Bytes requested per transport call.
const REQUEST_SIZE: u64 = 1 << 20;

/// Default maximum size of one artifact: 128 GiB.
pub const DEFAULT_MAX_ARTIFACT_BYTES: u64 = 128 * 1024 * 1024 * 1024;

/// A range of bytes returned by a transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Payload starting at the requested offset. May be shorter than requested.
    pub bytes: Vec<u8>,
    /// Total length of the complete resource, as reported by the transport.
    pub total_len: u64,
}

/// A source of byte ranges.
///
/// Implementors are expected to honour `offset` exactly. Returning fewer bytes
/// than `max_len` is normal and drives the resume loop; returning zero bytes
/// while the download is incomplete is treated as a stall, not as completion.
pub trait ChunkTransport {
    /// Fetch up to `max_len` bytes of `url` beginning at `offset`.
    fn fetch(&self, url: &str, offset: u64, max_len: u64) -> Result<Chunk, ModelError>;
}

/// Blocking HTTPS byte-range transport.
///
/// Redirects are disabled, plaintext URLs are rejected before dispatch, every
/// response must be `206 Partial Content`, and `Content-Range` must begin at
/// the exact requested offset. Bodies are read through a hard byte limit.
#[derive(Debug, Clone)]
pub struct HttpTransport {
    /// Reused client and connection pool; redirects remain disabled globally.
    agent: ureq::Agent,
}

impl HttpTransport {
    /// Construct a bounded HTTPS transport with a reusable connection pool.
    #[must_use]
    pub fn new() -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(30)))
            .max_redirects(0)
            .build()
            .into();
        Self { agent }
    }
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkTransport for HttpTransport {
    fn fetch(&self, url: &str, offset: u64, max_len: u64) -> Result<Chunk, ModelError> {
        validate_https_url(url)?;
        if max_len == 0 {
            return Err(transport_error(url, "requested range must not be empty"));
        }
        let end = offset
            .checked_add(max_len - 1)
            .ok_or_else(|| transport_error(url, "requested range overflows u64"))?;
        let mut response = self
            .agent
            .get(url)
            .header("Range", format!("bytes={offset}-{end}"))
            .call()
            .map_err(|error| transport_error(url, error.to_string()))?;
        if response.status().as_u16() != 206 {
            return Err(transport_error(
                url,
                format!(
                    "expected HTTP 206 Partial Content, received {}",
                    response.status()
                ),
            ));
        }
        let content_range = response
            .headers()
            .get("content-range")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| transport_error(url, "missing valid Content-Range header"))?;
        let (start, range_end, total_len) = parse_content_range(url, content_range)?;
        if start != offset || range_end > end {
            return Err(transport_error(
                url,
                format!(
                    "Content-Range {content_range} does not match requested bytes {offset}-{end}"
                ),
            ));
        }
        if total_len > DEFAULT_MAX_ARTIFACT_BYTES {
            return Err(ModelError::ArtifactTooLarge {
                url: url.to_owned(),
                actual: total_len,
                maximum: DEFAULT_MAX_ARTIFACT_BYTES,
            });
        }
        let bytes = response
            .body_mut()
            .with_config()
            .limit(max_len.saturating_add(1))
            .read_to_vec()
            .map_err(|error| transport_error(url, error.to_string()))?;
        let expected_len = range_end - start + 1;
        let actual_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if actual_len != expected_len || actual_len > max_len {
            return Err(transport_error(
                url,
                format!(
                    "Content-Range declares {expected_len} bytes but body contains {actual_len}"
                ),
            ));
        }
        Ok(Chunk { bytes, total_len })
    }
}

/// Reject plaintext, relative, fragment-bearing, or authority-less URLs.
fn validate_https_url(url: &str) -> Result<(), ModelError> {
    let parsed: ureq::http::Uri = url
        .parse()
        .map_err(|_| transport_error(url, "URL is not a valid absolute URI"))?;
    let safe_authority = parsed
        .authority()
        .is_some_and(|authority| !authority.as_str().contains('@'));
    if parsed.scheme_str() != Some("https") || !safe_authority || url.contains('#') {
        return Err(transport_error(
            url,
            "only absolute HTTPS URLs without fragments are allowed",
        ));
    }
    Ok(())
}

/// Parse `bytes start-end/total`, rejecting wildcard and incoherent values.
fn parse_content_range(url: &str, value: &str) -> Result<(u64, u64, u64), ModelError> {
    let range = value
        .strip_prefix("bytes ")
        .ok_or_else(|| transport_error(url, format!("invalid Content-Range '{value}'")))?;
    let (span, total) = range
        .split_once('/')
        .ok_or_else(|| transport_error(url, format!("invalid Content-Range '{value}'")))?;
    let (start, end) = span
        .split_once('-')
        .ok_or_else(|| transport_error(url, format!("invalid Content-Range '{value}'")))?;
    let start = start
        .parse::<u64>()
        .map_err(|_| transport_error(url, format!("invalid Content-Range '{value}'")))?;
    let end = end
        .parse::<u64>()
        .map_err(|_| transport_error(url, format!("invalid Content-Range '{value}'")))?;
    let total = total
        .parse::<u64>()
        .map_err(|_| transport_error(url, format!("invalid Content-Range '{value}'")))?;
    if end < start || end >= total {
        return Err(transport_error(
            url,
            format!("incoherent Content-Range '{value}'"),
        ));
    }
    Ok((start, end, total))
}

/// Construct a transport error without repeating URL cloning at call sites.
fn transport_error(url: &str, detail: impl Into<String>) -> ModelError {
    ModelError::Transport {
        url: url.to_owned(),
        detail: detail.into(),
    }
}

/// What a download run did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadOutcome {
    /// The destination already held a verified copy; no bytes were fetched.
    AlreadyPresent,
    /// The artifact was written and verified.
    Downloaded {
        /// Offset the run started from — non-zero means it resumed.
        resumed_from: u64,
        /// Bytes fetched during this run.
        bytes_written: u64,
    },
}

/// A resumable download of one artifact to one destination path.
#[derive(Debug, Clone)]
pub struct ResumableDownload<'a> {
    /// Manifest entry describing the expected bytes.
    artifact: &'a Artifact,
    /// Final path, written only after verification.
    destination: PathBuf,
    /// Largest artifact this operation will accept.
    max_artifact_bytes: u64,
}

impl<'a> ResumableDownload<'a> {
    /// Prepare a download of `artifact` to `destination`.
    #[must_use]
    pub fn new(artifact: &'a Artifact, destination: impl Into<PathBuf>) -> Self {
        Self {
            artifact,
            destination: destination.into(),
            max_artifact_bytes: DEFAULT_MAX_ARTIFACT_BYTES,
        }
    }

    /// Override the maximum accepted artifact size for this operation.
    #[must_use]
    pub const fn with_max_size(mut self, max_artifact_bytes: u64) -> Self {
        self.max_artifact_bytes = max_artifact_bytes;
        self
    }

    /// The final path.
    #[must_use]
    pub fn destination(&self) -> &Path {
        &self.destination
    }

    /// The partial file backing an interrupted download.
    #[must_use]
    pub fn part_path(&self) -> PathBuf {
        let mut name = self.destination.as_os_str().to_owned();
        name.push(".part");
        PathBuf::from(name)
    }

    /// Bytes already on disk, i.e. the offset the next run resumes from.
    pub fn resume_offset(&self) -> Result<u64, ModelError> {
        let path = self.part_path();
        match std::fs::metadata(&path) {
            Ok(metadata) => Ok(metadata.len()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(source) => Err(ModelError::io(path, source)),
        }
    }

    /// Run the download to completion, resuming from any partial file.
    pub fn run<T: ChunkTransport + ?Sized>(
        &self,
        transport: &T,
    ) -> Result<DownloadOutcome, ModelError> {
        if self.artifact.size_bytes > self.max_artifact_bytes {
            return Err(ModelError::ArtifactTooLarge {
                url: self.artifact.url.clone(),
                actual: self.artifact.size_bytes,
                maximum: self.max_artifact_bytes,
            });
        }
        if self.destination.exists() {
            crate::verify::verify_artifact(&self.destination, self.artifact)?;
            return Ok(DownloadOutcome::AlreadyPresent);
        }
        if let Some(parent) = self.destination.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|source| ModelError::io(parent, source))?;
        }

        let part = self.part_path();
        let total = self.artifact.size_bytes;
        let resumed_from = self.resume_offset()?;
        if resumed_from > total {
            return Err(ModelError::ResumeBeyondEnd {
                path: part,
                have: resumed_from,
                total,
            });
        }

        let written = self.fetch_remaining(transport, &part, resumed_from, total)?;
        self.publish(&part)?;
        Ok(DownloadOutcome::Downloaded {
            resumed_from,
            bytes_written: written,
        })
    }

    /// Append byte ranges to the partial file until it reaches `total`.
    fn fetch_remaining<T: ChunkTransport + ?Sized>(
        &self,
        transport: &T,
        part: &Path,
        resumed_from: u64,
        total: u64,
    ) -> Result<u64, ModelError> {
        let mut have = resumed_from;
        if have == total {
            return Ok(0);
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(part)
            .map_err(|source| ModelError::io(part, source))?;

        while have < total {
            let remaining = total - have;
            let chunk = transport.fetch(&self.artifact.url, have, remaining.min(REQUEST_SIZE))?;
            if chunk.total_len != total {
                return Err(ModelError::SizeMismatch {
                    path: part.to_path_buf(),
                    expected: total,
                    actual: chunk.total_len,
                });
            }
            let len = u64::try_from(chunk.bytes.len()).unwrap_or(u64::MAX);
            if len == 0 {
                return Err(ModelError::TransportStalled {
                    url: self.artifact.url.clone(),
                    offset: have,
                    total,
                });
            }
            if len > remaining {
                return Err(ModelError::SizeMismatch {
                    path: part.to_path_buf(),
                    expected: total,
                    actual: have + len,
                });
            }
            file.write_all(&chunk.bytes)
                .map_err(|source| ModelError::io(part, source))?;
            have += len;
        }
        file.flush()
            .map_err(|source| ModelError::io(part, source))?;
        file.sync_all()
            .map_err(|source| ModelError::io(part, source))?;
        Ok(have - resumed_from)
    }

    /// Verify the partial file and move it into place.
    ///
    /// A failure here removes the partial file, so a corrupt resume cannot
    /// wedge the download into repeating the same failure forever.
    fn publish(&self, part: &Path) -> Result<(), ModelError> {
        let metadata = std::fs::metadata(part).map_err(|source| ModelError::io(part, source))?;
        if metadata.len() != self.artifact.size_bytes {
            let actual = metadata.len();
            discard(part);
            return Err(ModelError::SizeMismatch {
                path: part.to_path_buf(),
                expected: self.artifact.size_bytes,
                actual,
            });
        }
        let digest = hash_file(part)?;
        if digest != self.artifact.sha256 {
            discard(part);
            return Err(ModelError::HashMismatch {
                path: part.to_path_buf(),
                expected: self.artifact.sha256.to_hex(),
                actual: digest.to_hex(),
            });
        }
        let file = std::fs::File::open(part).map_err(|source| ModelError::io(part, source))?;
        file.sync_all()
            .map_err(|source| ModelError::io(part, source))?;
        std::fs::hard_link(part, &self.destination)
            .map_err(|source| ModelError::io(&self.destination, source))?;
        // The hard link is the atomic publication point. Cleanup cannot make
        // the verified destination less valid, so a stale `.part` link is
        // harmless and must not turn a successful publication into an error.
        let _ = std::fs::remove_file(part);
        sync_parent_best_effort(&self.destination);
        Ok(())
    }
}

/// Remove a poisoned partial file, ignoring an already-absent path.
fn discard(part: &Path) {
    let _ = std::fs::remove_file(part);
}

/// Persist the containing directory where the platform exposes directory
/// handles. Publication is already atomic; this is an additional durability
/// step and therefore best-effort.
fn sync_parent_best_effort(destination: &Path) {
    #[cfg(unix)]
    if let Some(parent) = destination.parent()
        && let Ok(directory) = std::fs::File::open(parent)
    {
        let _ = directory.sync_all();
    }
}

#[cfg(test)]
mod tests;
