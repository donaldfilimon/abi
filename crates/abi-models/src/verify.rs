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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{Scratch, WEIGHTS_BYTES, manifest};

    #[test]
    fn hashing_matches_a_published_test_vector() {
        // NIST SHA-256 vector for "abc". Checking against an external reference
        // proves the hasher is correct, not merely self-consistent.
        let digest = hash_reader(&mut &b"abc"[..]).expect("hashes");
        assert_eq!(
            digest.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hashing_an_empty_input_matches_the_published_vector() {
        let digest = hash_reader(&mut &b""[..]).expect("hashes");
        assert_eq!(
            digest.to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn hashing_streams_input_larger_than_one_buffer() {
        // Larger than CHUNK, so the multi-update path is the one measured.
        let scratch = Scratch::new("hash_large");
        let payload: Vec<u8> = (0..CHUNK * 2 + 7)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let path = scratch.write("large.bin", &payload);
        let streamed = hash_file(&path).expect("hashes");
        let in_memory = hash_reader(&mut payload.as_slice()).expect("hashes");
        assert_eq!(streamed, in_memory);
    }

    #[test]
    fn a_matching_file_verifies() {
        let scratch = Scratch::new("verify_ok");
        let manifest = manifest();
        let artifact = manifest.weights().next().expect("weights");
        let path = scratch.write("model.safetensors", WEIGHTS_BYTES);
        verify_artifact(&path, artifact).expect("fixture bytes match the manifest hash");
    }

    #[test]
    fn corrupted_bytes_fail_with_both_digests_reported() {
        let scratch = Scratch::new("verify_bad");
        let manifest = manifest();
        let artifact = manifest.weights().next().expect("weights");
        // Same length, different content: only the hash can catch this.
        let mut corrupt = WEIGHTS_BYTES.to_vec();
        corrupt[0] ^= 0xff;
        let path = scratch.write("model.safetensors", &corrupt);

        let error = verify_artifact(&path, artifact).expect_err("must reject corrupted bytes");
        match error {
            ModelError::HashMismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected, artifact.sha256.to_hex());
                assert_ne!(actual, expected);
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_truncated_file_fails_on_size_before_hashing() {
        let scratch = Scratch::new("verify_short");
        let manifest = manifest();
        let artifact = manifest.weights().next().expect("weights");
        let path = scratch.write("model.safetensors", &WEIGHTS_BYTES[..4]);

        let error = verify_artifact(&path, artifact).expect_err("must reject a short file");
        assert!(
            matches!(error, ModelError::SizeMismatch { actual: 4, .. }),
            "{error:?}"
        );
    }

    #[test]
    fn a_missing_file_is_an_io_error() {
        let scratch = Scratch::new("verify_absent");
        let manifest = manifest();
        let artifact = manifest.weights().next().expect("weights");
        let error = verify_artifact(&scratch.join("absent.bin"), artifact)
            .expect_err("must not silently succeed");
        assert!(matches!(error, ModelError::Io { .. }), "{error:?}");
    }
}
