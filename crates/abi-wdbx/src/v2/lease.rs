//! Per-writer leases and exclusive maintenance barriers for v2 generations.

use super::V2Error;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const MAINTENANCE_FILE: &str = "MAINTENANCE";

pub(super) struct WriterLease {
    path: PathBuf,
    _file: File,
}

impl Drop for WriterLease {
    fn drop(&mut self) {
        // The advisory lock is released when `_file` drops after this method.
        // POSIX permits unlinking the held file; Windows may not, so a stale
        // pathname is harmless and maintenance later reclaims it.
        std::fs::remove_file(&self.path).ok();
    }
}

pub(super) struct MaintenanceGuard {
    path: PathBuf,
    _file: File,
}

impl Drop for MaintenanceGuard {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

pub(super) fn acquire_writer_lease(root: &Path, writer_id: Uuid) -> Result<WriterLease, V2Error> {
    let directory = root.join("leases");
    std::fs::create_dir_all(&directory).map_err(|error| io_error(&directory, &error))?;
    let maintenance = root.join(MAINTENANCE_FILE);
    if maintenance.exists() {
        return Err(V2Error::Maintenance(
            "generation maintenance is in progress".into(),
        ));
    }
    let path = directory.join(format!("{writer_id}.lease"));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| io_error(&path, &error))?;
    file.write_all(format!("writer_id={writer_id}\n").as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(|error| io_error(&path, &error))?;
    file.try_lock().map_err(|error| match error {
        TryLockError::WouldBlock => V2Error::Maintenance("writer lease is already held".into()),
        TryLockError::Error(error) => io_error(&path, &error),
    })?;
    let lease = WriterLease { path, _file: file };
    if maintenance.exists() {
        drop(lease);
        return Err(V2Error::Maintenance(
            "generation maintenance began while opening the writer".into(),
        ));
    }
    Ok(lease)
}

pub(super) fn begin_maintenance(root: &Path) -> Result<MaintenanceGuard, V2Error> {
    let path = root.join(MAINTENANCE_FILE);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                V2Error::Maintenance("generation maintenance is already in progress".into())
            } else {
                io_error(&path, &error)
            }
        })?;
    file.sync_all().map_err(|error| io_error(&path, &error))?;
    let guard = MaintenanceGuard { path, _file: file };
    let directory = root.join("leases");
    if !directory.is_dir() {
        return Ok(guard);
    }
    let mut leases: Vec<_> = std::fs::read_dir(&directory)
        .map_err(|error| io_error(&directory, &error))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "lease")
        })
        .collect();
    leases.sort();
    for lease in leases {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lease)
            .map_err(|error| io_error(&lease, &error))?;
        match file.try_lock() {
            Ok(()) => {
                drop(file);
                std::fs::remove_file(&lease).map_err(|error| io_error(&lease, &error))?;
            }
            Err(TryLockError::WouldBlock) => {
                return Err(V2Error::Maintenance(format!(
                    "active writer lease prevents maintenance: {}",
                    lease.display()
                )));
            }
            Err(TryLockError::Error(error)) => return Err(io_error(&lease, &error)),
        }
    }
    Ok(guard)
}

fn io_error(path: &Path, error: &std::io::Error) -> V2Error {
    V2Error::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writers_do_not_contend_but_maintenance_excludes_live_leases() {
        let root = abi_foundation::temp_path::temp_file_path("wdbx-v2-leases", "store");
        std::fs::create_dir_all(&root).unwrap();
        let left = acquire_writer_lease(&root, Uuid::new_v4()).unwrap();
        let right = acquire_writer_lease(&root, Uuid::new_v4()).unwrap();
        let Err(error) = begin_maintenance(&root) else {
            panic!("live writers must block maintenance");
        };
        assert!(error.to_string().contains("active writer lease"));
        assert!(!root.join(MAINTENANCE_FILE).exists());
        drop((left, right));

        let maintenance = begin_maintenance(&root).unwrap();
        let Err(error) = acquire_writer_lease(&root, Uuid::new_v4()) else {
            panic!("maintenance must block new writers");
        };
        assert!(error.to_string().contains("maintenance is in progress"));
        drop(maintenance);
        let writer = acquire_writer_lease(&root, Uuid::new_v4()).unwrap();
        drop(writer);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn maintenance_reclaims_crash_stale_unlocked_lease_files() {
        let root = abi_foundation::temp_path::temp_file_path("wdbx-v2-stale-lease", "store");
        let directory = root.join("leases");
        std::fs::create_dir_all(&directory).unwrap();
        let stale = directory.join(format!("{}.lease", Uuid::new_v4()));
        std::fs::write(&stale, b"stale\n").unwrap();
        let maintenance = begin_maintenance(&root).unwrap();
        assert!(!stale.exists());
        drop(maintenance);
        std::fs::remove_dir_all(root).unwrap();
    }
}
