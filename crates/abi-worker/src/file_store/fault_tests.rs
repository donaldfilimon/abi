//! Real scratch-file I/O with private, instance-owned fault injection.
use super::*;
use abi_foundation::temp_path::temp_file_path;
use std::io::{self, Read, Seek, Write};
use std::path::PathBuf;

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct FaultFile {
    inner: Box<dyn ControlFile>,
    max_write: usize,
    remaining: Option<usize>,
    zero_write: bool,
    fail_sync: bool,
}

impl Read for FaultFile {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.inner.read(bytes)
    }
}

impl Seek for FaultFile {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.inner.seek(from)
    }
}

impl Write for FaultFile {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.zero_write {
            return Ok(0);
        }
        if self.remaining == Some(0) {
            return Err(io::Error::other("injected disk failure"));
        }
        let count = bytes
            .len()
            .min(self.max_write)
            .min(self.remaining.unwrap_or(usize::MAX));
        let written = self.inner.write(&bytes[..count])?;
        if let Some(remaining) = &mut self.remaining {
            *remaining -= written;
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl ControlFile for FaultFile {
    fn byte_len(&self) -> io::Result<u64> {
        self.inner.byte_len()
    }

    fn sync_all(&self) -> io::Result<()> {
        self.inner.sync_all()?;
        if self.fail_sync {
            return Err(io::Error::other("injected uncertain sync"));
        }
        Ok(())
    }
}

fn fixture(
    max_write: usize,
    remaining: Option<usize>,
    zero_write: bool,
    fail_sync: bool,
) -> (Scratch, FileControlStore, ControlEvent) {
    let path = Scratch(temp_file_path("worker-io-fault", "log"));
    let mut store = FileControlStore::create(&path.0, FileControlStoreLimits::default()).unwrap();
    let inner = std::mem::replace(&mut store.file, Box::new(File::open(&path.0).unwrap()));
    store.file = Box::new(FaultFile {
        inner,
        max_write,
        remaining,
        zero_write,
        fail_sync,
    });
    let event = ControlEvent::Cancelled {
        job_id: uuid::Uuid::from_u128(1),
        request_digest: crate::ResultDigest::parse("0".repeat(64)).unwrap(),
        cancelled_at_unix_ms: 1,
    };
    (path, store, event)
}

#[test]
fn short_writes_complete_the_entire_frame_before_success() {
    let (path, mut store, event) = fixture(1, None, false, false);
    store.append(&event).unwrap();
    assert_eq!(store.load().unwrap(), vec![event.clone()]);
    drop(store);
    let recovered = FileControlStore::open(&path.0, FileControlStoreLimits::default()).unwrap();
    assert_eq!(recovered.load().unwrap(), vec![event]);
}

#[test]
fn partial_frame_failure_poisoning_refuses_reuse_and_reopen() {
    // Fail in the length, payload and checksum, each after real writes.
    let (_, mut sample, event) = fixture(usize::MAX, None, false, false);
    sample.append(&event).unwrap();
    let frame_bytes = usize::try_from(sample.bytes).unwrap() - HEADER.len();
    for accepted_bytes in [1, 5, frame_bytes - 1] {
        let (path, mut store, event) = fixture(3, Some(accepted_bytes), false, false);
        assert!(store.append(&event).is_err());
        assert_eq!(
            std::fs::metadata(&path.0).unwrap().len(),
            (HEADER.len() + accepted_bytes) as u64
        );
        assert_eq!(store.events, []);
        let before = std::fs::read(&path.0).unwrap();
        assert!(store.load().is_err());
        assert!(store.append(&event).is_err());
        assert_eq!(std::fs::read(&path.0).unwrap(), before);
        drop(store);
        assert!(FileControlStore::open(&path.0, FileControlStoreLimits::default()).is_err());
    }
}

#[test]
fn zero_write_never_acknowledges_or_reuses_the_handle() {
    let (path, mut store, event) = fixture(usize::MAX, None, true, false);
    assert!(store.append(&event).is_err());
    assert!(store.load().is_err());
    assert!(store.append(&event).is_err());
    assert_eq!(std::fs::read(&path.0).unwrap(), HEADER);
    drop(store);
    assert_eq!(
        FileControlStore::open(&path.0, FileControlStoreLimits::default())
            .unwrap()
            .load()
            .unwrap(),
        []
    );
}

#[test]
fn failed_sync_never_acknowledges_even_when_complete_frame_survives() {
    let (path, mut store, event) = fixture(usize::MAX, None, false, true);
    assert!(store.append(&event).is_err());
    assert_eq!(store.events, []);
    assert!(store.load().is_err());
    let bytes = std::fs::read(&path.0).unwrap();
    assert!(store.append(&event).is_err());
    assert_eq!(std::fs::read(&path.0).unwrap(), bytes);
    drop(store);
    assert_eq!(
        FileControlStore::open(&path.0, FileControlStoreLimits::default())
            .unwrap()
            .load()
            .unwrap(),
        vec![event]
    );
}
