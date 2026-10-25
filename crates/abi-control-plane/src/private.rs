//! Explicit private metadata files, never live user memory.
use crate::{Error, Result};
use std::{
    fs::File,
    path::{Path, PathBuf},
};
#[cfg(unix)]
pub(crate) fn prepare(path: &Path, initialize: bool) -> Result<(File, PathBuf)> {
    use std::os::unix::{
        ffi::OsStrExt,
        fs::{MetadataExt, OpenOptionsExt},
    };
    let parent = path.parent().ok_or(Error::Invalid)?.canonicalize()?;
    let owner = nix::unistd::Uid::effective().as_raw();
    let meta = std::fs::symlink_metadata(&parent)?;
    if !meta.is_dir() || meta.uid() != owner || meta.mode() & 0o077 != 0 {
        return Err(Error::Denied);
    }
    // SQLite must reopen by pathname. Every ancestor must prevent another user
    // replacing our private directory between file validation and SQLite open.
    // Root/current-owner sticky directories (e.g. /tmp) protect owned entries.
    for ancestor in parent.ancestors() {
        let meta = std::fs::symlink_metadata(ancestor)?;
        if !meta.is_dir()
            || (meta.uid() != owner && meta.uid() != 0)
            || (meta.mode() & 0o022 != 0 && meta.mode() & 0o1000 == 0)
        {
            return Err(Error::Denied);
        }
    }
    let canonical = parent.join(path.file_name().ok_or(Error::Invalid)?);
    if canonical.as_os_str().as_bytes().contains(&0) {
        return Err(Error::Invalid);
    }
    let mut options = std::fs::OpenOptions::new();
    options
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .mode(0o600);
    if initialize {
        options.create_new(true);
    }
    let file = options.open(&canonical)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.uid() != owner || meta.mode() & 0o077 != 0 || meta.nlink() != 1 {
        return Err(Error::Denied);
    }
    Ok((file, canonical))
}
#[cfg(not(unix))]
pub(crate) fn prepare(_path: &Path, _initialize: bool) -> Result<(File, PathBuf)> {
    Err(Error::Denied)
}
