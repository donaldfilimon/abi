//! Owner-protected durable worker control history.
//!
//! Checksums detect accidental corruption, not malicious rewriting or rollback
//! by the file owner. Keep the containing directory private and never replace
//! or unlink a live log. No torn-tail repair or empty-history fallback occurs.

use crate::{ControlEvent, JobControlStore, WorkerError};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::Path;

mod lock;

use lock::acquire_history_lock;

const HEADER: &[u8] = b"ABI-WORKER-LOG\x01\n";
const FRAME_OVERHEAD: u64 = 4 + 32;

// Keep durability at one I/O boundary. Tests can forward actual file writes
// through a per-instance faulting implementation without global syscall hooks.
trait ControlFile: std::io::Read + std::io::Write + std::io::Seek + Send {
    fn byte_len(&self) -> std::io::Result<u64>;
    fn sync_all(&self) -> std::io::Result<()>;
}

impl ControlFile for File {
    fn byte_len(&self) -> std::io::Result<u64> {
        Ok(self.metadata()?.len())
    }

    fn sync_all(&self) -> std::io::Result<()> {
        File::sync_all(self)
    }
}

/// Finite allocation and disk budgets for control history.
#[derive(Clone, Copy, Debug)]
pub struct FileControlStoreLimits {
    /// Maximum encoded event record size.
    pub record_bytes: usize,
    /// Maximum complete log size.
    pub log_bytes: u64,
    /// Maximum replayed events.
    pub events: usize,
}

impl Default for FileControlStoreLimits {
    fn default() -> Self {
        Self {
            // JSON byte vectors can take four encoded bytes per result byte.
            record_bytes: 8 * 1024 * 1024,
            log_bytes: 64 * 1024 * 1024,
            events: 100_000,
        }
    }
}

impl FileControlStoreLimits {
    fn validate(self) -> Result<(), WorkerError> {
        if self.record_bytes == 0
            || u32::try_from(self.record_bytes).is_err()
            || self.log_bytes < HEADER.len() as u64
            || self.events == 0
        {
            return Err(failure("invalid durable control budgets"));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct RecordRef<'a> {
    sequence: u64,
    previous: [u8; 32],
    event: &'a ControlEvent,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    sequence: u64,
    previous: [u8; 32],
    event: ControlEvent,
}

/// Exclusively held append-only history, independent of WDBX memory.
///
/// Unix files must belong to the current user with no group/other access.
/// Other platforms refuse until equivalent ACL validation is implemented.
/// An uncertain write poisons the handle; callers must stop dispatch and
/// recover through a new, fully validated open. Retain the path permanently:
/// an external replacement would invalidate the file's advisory lock boundary.
pub struct FileControlStore {
    file: Box<dyn ControlFile>,
    limits: FileControlStoreLimits,
    events: Vec<ControlEvent>,
    previous: [u8; 32],
    bytes: u64,
    poisoned: bool,
}

impl FileControlStore {
    /// Exclusively create and sync a new history in an existing private directory.
    /// Existing paths are never replaced; a failed creation may leave a file
    /// requiring operator inspection rather than silently resetting history.
    pub fn create(path: &Path, limits: FileControlStoreLimits) -> Result<Self, WorkerError> {
        limits.validate()?;
        let mut file = private_file(path, true)?;
        acquire_history_lock(&file)?;
        file.write_all(HEADER)
            .and_then(|()| file.sync_all())
            .map_err(|_| failure("cannot sync new history"))?;
        sync_parent(path)?;
        Ok(Self {
            file: Box::new(file),
            limits,
            events: Vec::new(),
            previous: [0; 32],
            bytes: HEADER.len() as u64,
            poisoned: false,
        })
    }

    /// Reopen and validate existing history before returning any replay evidence.
    /// Missing, oversized, corrupt, partial or unsupported history fails closed.
    pub fn open(path: &Path, limits: FileControlStoreLimits) -> Result<Self, WorkerError> {
        limits.validate()?;
        let mut file = private_file(path, false)?;
        acquire_history_lock(&file)?;
        let bytes = file
            .metadata()
            .map_err(|_| failure("cannot inspect history"))?
            .len();
        if bytes > limits.log_bytes || bytes < HEADER.len() as u64 {
            return Err(failure("history byte budget or header invalid"));
        }
        let mut header = [0; HEADER.len()];
        file.read_exact(&mut header)
            .map_err(|_| failure("incomplete history header"))?;
        if header != HEADER {
            return Err(failure("unsupported history header"));
        }
        let mut store = Self {
            file: Box::new(file),
            limits,
            events: Vec::new(),
            previous: [0; 32],
            bytes,
            poisoned: false,
        };
        let mut offset = HEADER.len() as u64;
        while offset < bytes {
            if store.events.len() >= limits.events || bytes - offset < FRAME_OVERHEAD {
                return Err(failure("event budget or incomplete history frame"));
            }
            let mut length = [0; 4];
            store
                .file
                .read_exact(&mut length)
                .map_err(|_| failure("incomplete record length"))?;
            let length = u32::from_be_bytes(length) as usize;
            if length == 0
                || length > limits.record_bytes
                || length as u64 > bytes - offset - FRAME_OVERHEAD
            {
                return Err(failure("record budget or incomplete history frame"));
            }
            let mut payload = vec![0; length];
            let mut checksum = [0; 32];
            store
                .file
                .read_exact(&mut payload)
                .and_then(|()| store.file.read_exact(&mut checksum))
                .map_err(|_| failure("incomplete history record"))?;
            let expected: [u8; 32] = Sha256::digest(&payload).into();
            if checksum != expected {
                return Err(failure("history checksum mismatch"));
            }
            let record: Record =
                serde_json::from_slice(&payload).map_err(|_| failure("invalid history record"))?;
            if record.sequence != store.events.len() as u64 || record.previous != store.previous {
                return Err(failure("history order or chain mismatch"));
            }
            store.previous = checksum;
            store.events.push(record.event);
            offset += length as u64 + FRAME_OVERHEAD;
        }
        Ok(store)
    }
}

impl JobControlStore for FileControlStore {
    fn load(&self) -> Result<Vec<ControlEvent>, WorkerError> {
        if self.poisoned {
            return Err(failure("history requires recovery"));
        }
        Ok(self.events.clone())
    }

    fn append(&mut self, event: &ControlEvent) -> Result<(), WorkerError> {
        if self.poisoned {
            return Err(failure("history requires recovery"));
        }
        if self.events.len() >= self.limits.events {
            return Err(failure("history event budget exhausted"));
        }
        let record = RecordRef {
            sequence: self.events.len() as u64,
            previous: self.previous,
            event,
        };
        let mut buffer = LimitedBuffer {
            bytes: Vec::new(),
            limit: self.limits.record_bytes,
        };
        serde_json::to_writer(&mut buffer, &record)
            .map_err(|_| failure("history record budget exceeded"))?;
        let payload = buffer.bytes;
        let length = u32::try_from(payload.len()).map_err(|_| failure("record too large"))?;
        let next_bytes = self
            .bytes
            .checked_add(u64::from(length) + FRAME_OVERHEAD)
            .filter(|bytes| *bytes <= self.limits.log_bytes)
            .ok_or_else(|| failure("history disk budget exhausted"))?;
        let checksum: [u8; 32] = Sha256::digest(&payload).into();
        // From this boundary onward, any error leaves the durability outcome uncertain.
        self.poisoned = true;
        if self
            .file
            .byte_len()
            .map_err(|_| failure("cannot inspect history"))?
            != self.bytes
        {
            return Err(failure("history was modified externally"));
        }
        self.file
            .seek(SeekFrom::Start(self.bytes))
            .and_then(|_| self.file.write_all(&length.to_be_bytes()))
            .and_then(|()| self.file.write_all(&payload))
            .and_then(|()| self.file.write_all(&checksum))
            .and_then(|()| self.file.sync_all())
            .map_err(|_| failure("history append durability uncertain"))?;
        self.events.push(event.clone());
        self.previous = checksum;
        self.bytes = next_bytes;
        self.poisoned = false;
        Ok(())
    }
}

struct LimitedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}

impl std::io::Write for LimitedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("record size limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(unix)]
fn private_file(path: &Path, create: bool) -> Result<File, WorkerError> {
    use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(create)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| failure("cannot open private regular history"))?;
    let metadata = file
        .metadata()
        .map_err(|_| failure("cannot inspect history permissions"))?;
    if !metadata.is_file()
        || metadata.uid() != nix::unistd::Uid::effective().as_raw()
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
    {
        return Err(failure(
            "history must be private, owned, regular and singly linked",
        ));
    }
    Ok(file)
}

#[cfg(not(unix))]
fn private_file(_path: &Path, _create: bool) -> Result<File, WorkerError> {
    Err(failure(
        "private history ACL validation unavailable on this platform",
    ))
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> Result<(), WorkerError> {
    File::open(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )
    .and_then(|parent| parent.sync_all())
    .map_err(|_| failure("cannot sync history directory"))
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> Result<(), WorkerError> {
    Err(failure(
        "history directory sync unavailable on this platform",
    ))
}

fn failure(message: &str) -> WorkerError {
    WorkerError::ControlStore(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi_foundation::temp_path::temp_file_path;

    #[cfg(unix)]
    #[test]
    fn actual_write_failure_poisoning_prevents_reuse() {
        let path = temp_file_path("worker-write-failure", "log");
        let mut store = FileControlStore::create(&path, FileControlStoreLimits::default()).unwrap();
        let before = std::fs::read(&path).unwrap();
        // A real read-only descriptor forces the production write syscall to fail.
        store.file = Box::new(File::open(&path).unwrap());
        let event = ControlEvent::Cancelled {
            job_id: uuid::Uuid::from_u128(1),
            request_digest: crate::ResultDigest::parse("0".repeat(64)).unwrap(),
            cancelled_at_unix_ms: 1,
        };
        assert!(store.append(&event).is_err());
        assert!(store.load().is_err());
        assert!(store.append(&event).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(not(unix))]
    #[test]
    fn unsupported_acl_validation_refuses_without_creating_state() {
        let path = temp_file_path("worker-unsupported-acl", "log");
        assert!(FileControlStore::create(&path, FileControlStoreLimits::default()).is_err());
        assert!(!path.exists());
        assert!(FileControlStore::open(&path, FileControlStoreLimits::default()).is_err());
    }
}

#[cfg(all(test, unix))]
#[path = "file_store/fault_tests.rs"]
mod fault_tests;
