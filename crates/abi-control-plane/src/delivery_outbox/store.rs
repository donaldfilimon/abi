//! Shared private SQLite lifecycle for encrypted delivery journals.
use crate::{Error, Result};
use rusqlite::{Connection, OpenFlags};
use std::{
    fs::File,
    path::{Path, PathBuf},
};

pub(crate) struct PrivateDb {
    pub db: Connection,
    file: File,
    path: PathBuf,
    lock: File,
    lock_path: PathBuf,
    pub poisoned: bool,
}
impl PrivateDb {
    pub fn sibling(&self) -> Result<Self> {
        self.check()?;
        let db = Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        db.execute_batch(
            "PRAGMA journal_mode=DELETE;PRAGMA synchronous=FULL;PRAGMA foreign_keys=ON;",
        )?;
        Ok(Self {
            db,
            file: self.file.try_clone()?,
            path: self.path.clone(),
            lock: self.lock.try_clone()?,
            lock_path: self.lock_path.clone(),
            poisoned: false,
        })
    }
    pub fn open(path: &Path, create: bool, ceiling: usize) -> Result<Self> {
        let (file, path) = crate::private::prepare(path, create)?;
        if file.metadata()?.len() > u64::try_from(ceiling).map_err(|_| Error::Invalid)? {
            return Err(Error::Full);
        }
        // SQLite may open recovery journals before the first transaction. Validate
        // every existing sidecar before giving it the database pathname.
        for suffix in ["-journal", "-wal", "-shm"] {
            let sidecar = path.with_file_name(format!(
                "{}{suffix}",
                path.file_name().ok_or(Error::Invalid)?.to_string_lossy()
            ));
            match std::fs::symlink_metadata(&sidecar) {
                Ok(_) => {
                    crate::private::prepare(&sidecar, false)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        let lock_path = path.with_file_name(format!(
            "{}.lock",
            path.file_name().ok_or(Error::Invalid)?.to_string_lossy()
        ));
        let (lock, lock_path) = crate::private::prepare(&lock_path, create)?;
        lock.try_lock().map_err(|_| Error::Denied)?;
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        #[cfg(test)]
        let db = if super::faults::applies(&path) {
            Connection::open_with_flags_and_vfs(&path, flags, "abi-outbox-fault-tests")?
        } else {
            Connection::open_with_flags(&path, flags)?
        };
        #[cfg(not(test))]
        let db = Connection::open_with_flags(&path, flags)?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        db.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON; PRAGMA secure_delete=ON;")?;
        let pages = i64::try_from(ceiling / 4096).map_err(|_| Error::Invalid)?;
        if db.pragma_query_value(None, "page_size", |r| r.get::<_, i64>(0))? != 4096
            || db.pragma_query_value(None, "page_count", |r| r.get::<_, i64>(0))? > pages
        {
            return Err(Error::Full);
        }
        db.pragma_update(None, "max_page_count", pages)?;
        let store = Self {
            db,
            file,
            path,
            lock,
            lock_path,
            poisoned: false,
        };
        store.check()?;
        if create {
            store.file.sync_all()?;
            File::open(store.path.parent().ok_or(Error::Invalid)?)?.sync_all()?;
        }
        Ok(store)
    }
    pub fn check(&self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Denied);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let owner = nix::unistd::Uid::effective().as_raw();
            let parent = self.path.parent().ok_or(Error::Invalid)?;
            for directory in parent.ancestors() {
                let metadata = std::fs::symlink_metadata(directory)?;
                if !metadata.is_dir()
                    || (metadata.uid() != owner && metadata.uid() != 0)
                    || (metadata.mode() & 0o022 != 0 && metadata.mode() & 0o1000 == 0)
                    || (directory == parent
                        && (metadata.uid() != owner || metadata.mode() & 0o077 != 0))
                {
                    return Err(Error::Denied);
                }
            }
            for (path, file) in [(&self.path, &self.file), (&self.lock_path, &self.lock)] {
                let now = std::fs::symlink_metadata(path)?;
                let original = file.metadata()?;
                if !now.is_file()
                    || now.dev() != original.dev()
                    || now.ino() != original.ino()
                    || now.nlink() != 1
                    || now.mode() & 0o077 != 0
                    || now.uid() != nix::unistd::Uid::effective().as_raw()
                {
                    return Err(Error::Denied);
                }
            }
        }
        Ok(())
    }
    pub fn write<T>(
        &mut self,
        operation: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        self.check()?;
        let tx = self
            .db
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let value = match operation(&tx) {
            Ok(v) => v,
            Err(e) => {
                if matches!(e, Error::Storage(_) | Error::Io(_)) {
                    self.poisoned = true;
                }
                return Err(e);
            }
        };
        if let Err(error) = tx.commit() {
            self.poisoned = true;
            return Err(error.into());
        }
        if let Err(error) = self.check() {
            self.poisoned = true;
            return Err(error);
        }
        Ok(value)
    }
}
