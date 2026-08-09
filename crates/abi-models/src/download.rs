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
//!                            verify partial: size, then SHA-256
//!                                        │
//!                     mismatch ──► delete partial, error
//!                        ok     ──► rename partial to destination
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
//! No real transport ships here — see [`HttpTransport`].

use crate::error::ModelError;
use crate::manifest::Artifact;
use crate::verify::hash_file;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Bytes requested per transport call.
const REQUEST_SIZE: u64 = 1 << 20;

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

/// Placeholder for a real network transport.
///
/// **Proposed, not implemented.** Every call returns
/// [`ModelError::TransportNotImplemented`]. It returns an error rather than
/// panicking so a caller that reaches for it gets an honest refusal instead of
/// an abort, and so its absence cannot be mistaken for working code.
#[derive(Debug, Clone, Copy, Default)]
pub struct HttpTransport;

impl ChunkTransport for HttpTransport {
    fn fetch(&self, _url: &str, _offset: u64, _max_len: u64) -> Result<Chunk, ModelError> {
        Err(ModelError::TransportNotImplemented)
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
}

impl<'a> ResumableDownload<'a> {
    /// Prepare a download of `artifact` to `destination`.
    #[must_use]
    pub fn new(artifact: &'a Artifact, destination: impl Into<PathBuf>) -> Self {
        Self {
            artifact,
            destination: destination.into(),
        }
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
        std::fs::rename(part, &self.destination)
            .map_err(|source| ModelError::io(&self.destination, source))
    }
}

/// Remove a poisoned partial file, ignoring an already-absent path.
fn discard(part: &Path) {
    let _ = std::fs::remove_file(part);
}
