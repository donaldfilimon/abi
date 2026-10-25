//! Real-file worker recovery, corruption, lock and process-exit contracts.
#![cfg(unix)]

#[allow(dead_code)]
mod support;

use abi_foundation::temp_path::temp_file_path;
use abi_worker::{
    Admission, FileControlStore, FileControlStoreLimits, JobControlStore, JobStatus, ResultChunk,
    WorkerLimits, WorkerManager,
};
use std::path::{Path, PathBuf};
use support::{NOW, capability, envelope, identity, trust};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        Self(temp_file_path("abi-worker-control", "log"))
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn manager(store: FileControlStore) -> WorkerManager<FileControlStore> {
    WorkerManager::new(
        identity(),
        [capability()],
        trust(),
        WorkerLimits::default(),
        store,
    )
    .expect("durable manager")
}

#[test]
fn completed_job_returns_cached_result_after_reopen() {
    let path = Scratch::new();
    let store = FileControlStore::create(path.path(), FileControlStoreLimits::default()).unwrap();
    let request = envelope();
    let mut worker = manager(store);
    let Admission::Leased(lease) = worker.submit(&request, NOW).unwrap() else {
        panic!("new lease")
    };
    let result = worker
        .complete(
            &lease,
            JobStatus::Succeeded,
            vec![ResultChunk {
                sequence: 0,
                bytes: b"done".to_vec(),
            }],
            NOW + 1,
        )
        .unwrap();
    drop(worker);
    let mut recovered =
        manager(FileControlStore::open(path.path(), FileControlStoreLimits::default()).unwrap());
    assert_eq!(
        recovered.submit(&request, NOW + 2).unwrap(),
        Admission::Cached(result)
    );
}

#[test]
fn default_store_retains_maximum_default_worker_result() {
    let path = Scratch::new();
    let store = FileControlStore::create(path.path(), FileControlStoreLimits::default()).unwrap();
    let mut worker = manager(store);
    let mut input = support::request();
    let bytes = WorkerLimits::default().result_bytes;
    input.max_result_bytes = u64::try_from(bytes).unwrap();
    let request = support::envelope_for(uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), input);
    let Admission::Leased(lease) = worker.submit(&request, NOW).unwrap() else {
        panic!("leased")
    };
    worker
        .complete(
            &lease,
            JobStatus::Succeeded,
            vec![ResultChunk {
                sequence: 0,
                bytes: vec![255; bytes],
            }],
            NOW + 1,
        )
        .unwrap();
    drop(worker);
    let mut recovered =
        manager(FileControlStore::open(path.path(), FileControlStoreLimits::default()).unwrap());
    let Admission::Cached(result) = recovered.submit(&request, NOW + 2).unwrap() else {
        panic!("cached")
    };
    assert_eq!(result.chunks[0].bytes.len(), bytes);
    assert_eq!(result.chunks[0].bytes[bytes - 1], 255);
}

#[test]
fn admitted_job_retry_never_becomes_a_new_lease_after_restart() {
    let path = Scratch::new();
    let request = envelope();
    let mut worker =
        manager(FileControlStore::create(path.path(), FileControlStoreLimits::default()).unwrap());
    let Admission::Leased(lease) = worker.submit(&request, NOW).unwrap() else {
        panic!("new lease")
    };
    drop(worker);
    let mut recovered =
        manager(FileControlStore::open(path.path(), FileControlStoreLimits::default()).unwrap());
    assert_eq!(
        recovered.submit(&request, NOW + 1).unwrap(),
        Admission::ExistingLease(lease)
    );
}

#[test]
fn existing_history_is_exclusive_and_never_reset() {
    let path = Scratch::new();
    let limits = FileControlStoreLimits::default();
    assert!(FileControlStore::open(path.path(), limits).is_err());
    assert!(!path.path().exists());
    let store = FileControlStore::create(path.path(), limits).unwrap();
    let before = std::fs::read(path.path()).unwrap();
    assert!(FileControlStore::create(path.path(), limits).is_err());
    assert!(FileControlStore::open(path.path(), limits).is_err());
    assert_eq!(std::fs::read(path.path()).unwrap(), before);
    drop(store);
    assert!(FileControlStore::open(path.path(), limits).is_ok());
}

#[test]
fn corrupted_and_partial_frames_refuse_recovery() {
    let path = Scratch::new();
    let limits = FileControlStoreLimits::default();
    let mut worker = manager(FileControlStore::create(path.path(), limits).unwrap());
    worker.submit(&envelope(), NOW).unwrap();
    drop(worker);
    let original = std::fs::read(path.path()).unwrap();
    let mut corrupted = original.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    std::fs::write(path.path(), &corrupted).unwrap();
    assert!(FileControlStore::open(path.path(), limits).is_err());
    std::fs::write(path.path(), &original[..last]).unwrap();
    assert!(FileControlStore::open(path.path(), limits).is_err());
    std::fs::write(path.path(), b"unknown-history-version").unwrap();
    assert!(FileControlStore::open(path.path(), limits).is_err());
}

#[test]
fn budgets_refuse_before_mutating_durable_history() {
    let path = Scratch::new();
    let limits = FileControlStoreLimits {
        events: 1,
        ..FileControlStoreLimits::default()
    };
    let mut store = FileControlStore::create(path.path(), limits).unwrap();
    // Use the actual manager's public control-event contract from a second durable log.
    let source = Scratch::new();
    let mut worker = manager(
        FileControlStore::create(source.path(), FileControlStoreLimits::default()).unwrap(),
    );
    worker.submit(&envelope(), NOW).unwrap();
    drop(worker);
    let event = FileControlStore::open(source.path(), FileControlStoreLimits::default())
        .unwrap()
        .load()
        .unwrap()
        .remove(0);
    store.append(&event).unwrap();
    let before = std::fs::read(path.path()).unwrap();
    assert!(store.append(&event).is_err());
    assert_eq!(std::fs::read(path.path()).unwrap(), before);
    drop(store);
    assert_eq!(
        FileControlStore::open(path.path(), limits)
            .unwrap()
            .load()
            .unwrap()
            .len(),
        1
    );
    let small = FileControlStoreLimits {
        log_bytes: 1,
        ..limits
    };
    assert!(FileControlStore::open(path.path(), small).is_err());
}

#[cfg(unix)]
#[test]
fn unsafe_permissions_and_symlinks_refuse() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let path = Scratch::new();
    let alias = Scratch::new();
    let limits = FileControlStoreLimits::default();
    drop(FileControlStore::create(path.path(), limits).unwrap());
    symlink(path.path(), alias.path()).unwrap();
    assert!(FileControlStore::open(alias.path(), limits).is_err());
    std::fs::set_permissions(path.path(), std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(FileControlStore::open(path.path(), limits).is_err());
}

#[test]
fn external_truncation_poisons_handle_and_blocks_further_appends() {
    let path = Scratch::new();
    let limits = FileControlStoreLimits::default();
    let mut worker = manager(FileControlStore::create(path.path(), limits).unwrap());
    worker.submit(&envelope(), NOW).unwrap();
    drop(worker);
    let mut store = FileControlStore::open(path.path(), limits).unwrap();
    let event = store.load().unwrap().remove(0);
    std::fs::OpenOptions::new()
        .write(true)
        .open(path.path())
        .unwrap()
        .set_len(1)
        .unwrap();
    assert!(store.append(&event).is_err());
    assert!(store.load().is_err());
    assert!(store.append(&event).is_err());
    assert_eq!(std::fs::metadata(path.path()).unwrap().len(), 1);
}

#[test]
fn individual_record_and_disk_limits_refuse_before_write() {
    let source = Scratch::new();
    let mut worker = manager(
        FileControlStore::create(source.path(), FileControlStoreLimits::default()).unwrap(),
    );
    worker.submit(&envelope(), NOW).unwrap();
    drop(worker);
    let event = FileControlStore::open(source.path(), FileControlStoreLimits::default())
        .unwrap()
        .load()
        .unwrap()
        .remove(0);
    for limits in [
        FileControlStoreLimits {
            record_bytes: 1,
            ..FileControlStoreLimits::default()
        },
        FileControlStoreLimits {
            log_bytes: 32,
            ..FileControlStoreLimits::default()
        },
    ] {
        let path = Scratch::new();
        let mut store = FileControlStore::create(path.path(), limits).unwrap();
        let before = std::fs::read(path.path()).unwrap();
        assert!(store.append(&event).is_err());
        assert_eq!(std::fs::read(path.path()).unwrap(), before);
        assert_eq!(store.load().unwrap(), []);
    }
}

#[test]
fn reordered_valid_frames_and_hostile_lengths_refuse() {
    let path = Scratch::new();
    let limits = FileControlStoreLimits::default();
    let mut worker = manager(FileControlStore::create(path.path(), limits).unwrap());
    let Admission::Leased(lease) = worker.submit(&envelope(), NOW).unwrap() else {
        panic!("new lease")
    };
    worker
        .complete(&lease, JobStatus::Succeeded, Vec::new(), NOW + 1)
        .unwrap();
    drop(worker);
    let original = std::fs::read(path.path()).unwrap();
    let header_end = original.iter().position(|b| *b == b'\n').unwrap() + 1;
    let frame_length =
        u32::from_be_bytes(original[header_end..header_end + 4].try_into().unwrap()) as usize + 36;
    let mut reordered = original[..header_end].to_vec();
    reordered.extend_from_slice(&original[header_end + frame_length..]);
    reordered.extend_from_slice(&original[header_end..header_end + frame_length]);
    std::fs::write(path.path(), reordered).unwrap();
    assert!(FileControlStore::open(path.path(), limits).is_err());
    let mut hostile = original;
    hostile[header_end..header_end + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    std::fs::write(path.path(), hostile).unwrap();
    assert!(FileControlStore::open(path.path(), limits).is_err());
}

fn fixed_request() -> abi_worker::JobEnvelope {
    support::envelope_for(
        uuid::Uuid::from_u128(41),
        uuid::Uuid::from_u128(42),
        support::request(),
    )
}

#[test]
fn child_commit_without_running_destructors() {
    use std::io::Read as _;
    let mut path = String::new();
    std::io::stdin().read_to_string(&mut path).unwrap();
    if path.is_empty() {
        return;
    }
    let mut worker = manager(
        FileControlStore::create(Path::new(&path), FileControlStoreLimits::default()).unwrap(),
    );
    worker.submit(&fixed_request(), NOW).unwrap();
    std::process::exit(0);
}

#[test]
fn process_exit_after_admission_preserves_non_dispatching_retry() {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let path = Scratch::new();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_commit_without_running_destructors"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(path.path().to_str().unwrap().as_bytes())
        .unwrap();
    assert!(child.wait().unwrap().success());
    let mut recovered =
        manager(FileControlStore::open(path.path(), FileControlStoreLimits::default()).unwrap());
    assert!(matches!(
        recovered.submit(&fixed_request(), NOW + 1).unwrap(),
        Admission::ExistingLease(_)
    ));
}
