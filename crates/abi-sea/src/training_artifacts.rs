//! Private bounded publication under an owned registry session. Unix ownership is
//! checked against a kernel-owned empty probe, never an environment UID. This is
//! a trusted same-user filesystem boundary, not hostile-path race isolation.
use crate::training_registry::{RegistryError, TrainingRegistry};
use abi_ai::training_contracts::{ArtifactRef, sha256_hex, validate_artifact};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
/// Maximum bounded publication size; streaming large weights is a later API.
pub const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
/// Artifact publication or verification refusal.
#[derive(Debug)]
pub enum ArtifactError {
    /// Unsafe path, schema, permissions, length or digest.
    Invalid(String),
    /// Filesystem operation failed.
    Io(std::io::Error),
    /// Registry session refused publication.
    Registry(RegistryError),
}
impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ArtifactError {}
impl From<std::io::Error> for ArtifactError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<RegistryError> for ArtifactError {
    fn from(e: RegistryError) -> Self {
        Self::Registry(e)
    }
}
fn invalid(s: &str) -> ArtifactError {
    ArtifactError::Invalid(s.into())
}
struct OwnedTemporary(PathBuf);
impl Drop for OwnedTemporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn temporary(root: &Path) -> Result<(OwnedTemporary, File), ArtifactError> {
    let path = root.join(format!(
        ".stage-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&path)?;
    Ok((OwnedTemporary(path), file))
}
#[cfg(unix)]
fn validate_owner(root_uid: u32, probe_uid: u32, probe_mode: u32) -> Result<(), ArtifactError> {
    if probe_mode & 0o7777 != 0o600 || root_uid != probe_uid {
        return Err(invalid(
            "root owner differs from kernel-created 0600 owner probe",
        ));
    }
    Ok(())
}
#[cfg(unix)]
pub(crate) fn private_root(root: &Path) -> Result<u32, ArtifactError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(root)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() || metadata.mode() & 0o7777 != 0o700
    {
        return Err(invalid("root must be a private real 0700 directory"));
    }
    let (probe_path, file) = temporary(root)?;
    let probe = file.metadata()?;
    validate_owner(metadata.uid(), probe.uid(), probe.mode())?;
    drop(file);
    fs::remove_file(&probe_path.0)?;
    Ok(probe.uid())
}
#[cfg(not(unix))]
pub(crate) fn private_root(_root: &Path) -> Result<u32, ArtifactError> {
    Err(invalid(
        "private artifact ownership unsupported on this platform",
    ))
}
fn checked_file(root: &Path, reference: &ArtifactRef) -> Result<File, ArtifactError> {
    validate_artifact(reference).map_err(|e| invalid(&e.to_string()))?;
    if reference.byte_length > MAX_ARTIFACT_BYTES {
        return Err(invalid("artifact exceeds bounded API"));
    }
    let owner = private_root(root)?;
    let path = root.join(&reference.relative_path);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(invalid("artifact must be a real file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != owner || metadata.mode() & 0o7777 != 0o600 || metadata.nlink() != 1 {
            return Err(invalid("artifact owner, permissions or hard links"));
        }
    }
    #[cfg(not(unix))]
    let _ = owner;
    let file = File::open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let opened = file.metadata()?;
        if opened.dev() != metadata.dev() || opened.ino() != metadata.ino() {
            return Err(invalid("artifact changed during open"));
        }
    }
    Ok(file)
}
/// Verifies physical bytes at the explicit private root; metadata is insufficient.
pub fn verify_artifact(root: &Path, reference: &ArtifactRef) -> Result<(), ArtifactError> {
    let file = checked_file(root, reference)?;
    let mut bytes = Vec::new();
    file.take(MAX_ARTIFACT_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != reference.byte_length || sha256_hex(&bytes) != reference.sha256 {
        return Err(invalid("artifact bytes differ from reference"));
    }
    Ok(())
}
/// Stages and syncs verified bytes, renames atomically, then publishes the registry
/// reference through the owned WAL session. An uncertain WAL error poisons it.
pub fn publish_artifact(
    registry: &mut TrainingRegistry,
    root: &Path,
    expected_sha256: &str,
    bytes: &[u8],
) -> Result<ArtifactRef, ArtifactError> {
    publish(registry, root, expected_sha256, bytes, Publication::Normal)
}
#[derive(Clone, Copy)]
enum Publication {
    Normal,
    #[cfg(test)]
    Interrupted,
}
impl Publication {
    fn interrupted(self) -> bool {
        match self {
            Self::Normal => false,
            #[cfg(test)]
            Self::Interrupted => true,
        }
    }
}
fn publish(
    registry: &mut TrainingRegistry,
    root: &Path,
    expected_sha256: &str,
    bytes: &[u8],
    publication: Publication,
) -> Result<ArtifactRef, ArtifactError> {
    registry.artifact_session_healthy()?;
    let reference = ArtifactRef {
        version: 1,
        relative_path: format!("{expected_sha256}.artifact"),
        sha256: expected_sha256.into(),
        byte_length: bytes.len() as u64,
    };
    validate_artifact(&reference).map_err(|e| invalid(&e.to_string()))?;
    if reference.byte_length > MAX_ARTIFACT_BYTES || sha256_hex(bytes) != expected_sha256 {
        return Err(invalid("publication digest or size"));
    }
    private_root(root)?;
    let destination = root.join(&reference.relative_path);
    match fs::symlink_metadata(&destination) {
        Ok(_) => verify_artifact(root, &reference)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let (staged, mut file) = temporary(root)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            if publication.interrupted() {
                return Err(invalid("test interrupted staged publication"));
            }
            private_root(root)?;
            fs::rename(&staged.0, &destination)?;
            File::open(root)?.sync_all()?;
            verify_artifact(root, &reference)?;
        }
        Err(error) => return Err(error.into()),
    }
    registry.register_artifact(reference.clone())?;
    Ok(reference)
}
#[cfg(test)]
mod tests;
