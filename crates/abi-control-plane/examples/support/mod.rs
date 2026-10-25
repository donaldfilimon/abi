//! Private local node/client files. No live ABI store or remote provider credentials.
use std::{
    io::{Read as _, Write as _},
    path::Path,
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub fn now() -> Result<i64> {
    Ok(i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
    )?)
}
pub fn read_private<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        let m = file.metadata()?;
        if !m.is_file()
            || m.uid() != nix::unistd::Uid::effective().as_raw()
            || m.mode() & 0o077 != 0
            || m.nlink() != 1
            || m.len() > 65536
        {
            return Err("private local credential file required".into());
        }
        let mut bytes = Vec::new();
        file.take(65537).read_to_end(&mut bytes)?;
        if bytes.len() > 65536 {
            return Err("local file too large".into());
        }
        Ok(serde_json::from_slice(&bytes)?)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("platform ACL validation unavailable".into())
    }
}
pub fn write_private<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        file.write_all(&serde_json::to_vec_pretty(value)?)?;
        file.sync_all()?;
        std::fs::File::open(path.parent().ok_or("missing parent")?)?.sync_all()?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (path, value);
        Err("platform ACL validation unavailable".into())
    }
}
pub fn directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        std::fs::DirBuilder::new().mode(0o700).create(path)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("platform ACL validation unavailable".into())
    }
}
pub fn seed() -> [u8; 32] {
    let mut s = [0; 32];
    s[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    s[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    s
}
