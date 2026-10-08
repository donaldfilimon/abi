//! Filesystem checks at registry admission, not hostile same-user race isolation.
use super::RegistryError;
use abi_wdbx::manifest::StorePaths;
#[cfg(unix)]
use std::{
    fs,
    path::{Component, Path},
};
#[cfg(unix)]
fn invalid(message: &str) -> RegistryError {
    RegistryError::Invalid(message.into())
}
#[cfg(unix)]
pub(super) fn validate_store(paths: &StorePaths) -> Result<(), RegistryError> {
    use std::os::unix::fs::MetadataExt;
    let mut base = Path::new(&paths.base).components();
    if !matches!(base.next(), Some(Component::Normal(_))) || base.next().is_some() {
        return Err(invalid(
            "store base must be one filename beneath the private root",
        ));
    }
    let components = validate_components(&paths.dir)?;
    let owner = crate::training_artifacts::private_root(&paths.dir)
        .map_err(|error| invalid(&error.to_string()))?;
    for metadata in components {
        validate_ancestor_owner(metadata.uid(), owner)?;
    }
    // WDBX derives its WAL, lock, manifest, mirror and segments from this base.
    // Checking all existing matching entries includes stale checkpoint segments.
    let prefix = format!("{}.", paths.base);
    for entry in
        fs::read_dir(&paths.dir).map_err(|error| RegistryError::Storage(error.to_string()))?
    {
        let entry = entry.map_err(|error| RegistryError::Storage(error.to_string()))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name != paths.base && !name.starts_with(&prefix) {
            continue;
        }
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|error| RegistryError::Storage(error.to_string()))?;
        validate_file(&metadata, owner)?;
    }
    Ok(())
}
#[cfg(unix)]
fn validate_file(metadata: &fs::Metadata, owner: u32) -> Result<(), RegistryError> {
    use std::os::unix::fs::MetadataExt;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.uid() != owner
        || metadata.nlink() != 1
    {
        return Err(invalid(
            "store files must be real owned files without hard links",
        ));
    }
    // A 0700 root blocks other-user traversal even when WDBX creates 0644 files.
    Ok(())
}
#[cfg(unix)]
fn validate_components(path: &Path) -> Result<Vec<fs::Metadata>, RegistryError> {
    use std::os::unix::fs::MetadataExt;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| RegistryError::Storage(error.to_string()))?
            .join(path)
    };
    let mut current = std::path::PathBuf::new();
    let mut metadata_chain = Vec::new();
    for component in absolute.components() {
        if matches!(component, Component::ParentDir | Component::CurDir) {
            return Err(invalid("store root must have normalized path components"));
        }
        current.push(component);
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| RegistryError::Storage(error.to_string()))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(invalid("store path components must be real directories"));
        }
        // System-owned sticky temporary ancestors are valid; the owned root is 0700.
        if metadata.mode() & 0o022 != 0 && metadata.mode() & 0o1000 == 0 {
            return Err(invalid(
                "store ancestor is writable by others without sticky protection",
            ));
        }
        metadata_chain.push(metadata);
    }
    Ok(metadata_chain)
}
#[cfg(unix)]
fn validate_ancestor_owner(uid: u32, owner: u32) -> Result<(), RegistryError> {
    if uid != 0 && uid != owner {
        return Err(invalid(
            "store ancestors must be owned by root or the caller",
        ));
    }
    Ok(())
}
#[cfg(not(unix))]
pub(super) fn validate_store(_paths: &StorePaths) -> Result<(), RegistryError> {
    Err(RegistryError::Invalid(
        "private registry ownership unsupported on this platform".into(),
    ))
}
#[cfg(all(test, unix))]
#[test]
fn owner_mismatch_refuses() {
    use std::os::unix::fs::MetadataExt;
    let path = std::env::current_exe().unwrap();
    let metadata = std::fs::symlink_metadata(path).unwrap();
    assert!(validate_file(&metadata, metadata.uid().wrapping_add(1)).is_err());
}

#[cfg(all(test, unix))]
#[test]
fn foreign_ancestor_ownership_refuses_even_if_not_publicly_writable() {
    validate_ancestor_owner(0, 501).unwrap();
    validate_ancestor_owner(501, 501).unwrap();
    assert!(validate_ancestor_owner(502, 501).is_err());
}
