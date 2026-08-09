//! Streaming SHA-256 verification of downloaded artifacts.
//!
//! Hashing is streamed through a fixed buffer rather than reading the file into
//! memory, because weight shards are routinely larger than available RAM. The
//! same helper backs both post-download verification and the re-verification a
//! resumed download performs before it publishes a file.

use crate::error::ModelError;
use crate::manifest::{Artifact, Sha256Digest};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Bytes read per hasher update.
const CHUNK: usize = 64 * 1024;

/// Stream a reader through SHA-256 and return the digest.
pub fn hash_reader<R: Read>(reader: &mut R) -> Result<Sha256Digest, std::io::Error> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; CHUNK];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(Sha256Digest::from_bytes(hasher.finalize().into()))
}

/// Stream a file through SHA-256 and return the digest.
pub fn hash_file(path: &Path) -> Result<Sha256Digest, ModelError> {
    let mut file = File::open(path).map_err(|source| ModelError::io(path, source))?;
    hash_reader(&mut file).map_err(|source| ModelError::io(path, source))
}

/// Check a file's size and SHA-256 against the manifest entry.
///
/// Size is checked first so a truncated file reports the cheaper, more specific
/// [`ModelError::SizeMismatch`] instead of a bare hash mismatch.
pub fn verify_artifact(path: &Path, artifact: &Artifact) -> Result<(), ModelError> {
    let metadata = std::fs::metadata(path).map_err(|source| ModelError::io(path, source))?;
    if metadata.len() != artifact.size_bytes {
        return Err(ModelError::SizeMismatch {
            path: path.to_path_buf(),
            expected: artifact.size_bytes,
            actual: metadata.len(),
        });
    }
    let actual = hash_file(path)?;
    if actual == artifact.sha256 {
        Ok(())
    } else {
        Err(ModelError::HashMismatch {
            path: path.to_path_buf(),
            expected: artifact.sha256.to_hex(),
            actual: actual.to_hex(),
        })
    }
}
