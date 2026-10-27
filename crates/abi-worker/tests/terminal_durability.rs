//! A lost terminal acknowledgment never gives a provider another execution.
#![cfg(unix)]
#[allow(dead_code)]
mod support;

use abi_agent_runtime::{EchoProvider, ModelProvider, ModelRequest, RunContext, RuntimeError};
use abi_worker::{
    ControlEvent, FileControlStore, FileControlStoreLimits, InferenceOutcome, JobControlStore,
    RecordedExecution, WorkerError, WorkerLimits, WorkerManager,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use support::*;

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct TerminalFailure {
    inner: FileControlStore,
    commit_first: bool,
    poisoned: bool,
}

impl JobControlStore for TerminalFailure {
    fn load(&self) -> Result<Vec<ControlEvent>, WorkerError> {
        if self.poisoned {
            return Err(WorkerError::ControlStore("uncertain terminal".into()));
        }
        self.inner.load()
    }

    fn append(&mut self, event: &ControlEvent) -> Result<(), WorkerError> {
        if self.poisoned {
            return Err(WorkerError::ControlStore("uncertain terminal".into()));
        }
        if matches!(event, ControlEvent::Completed { .. }) {
            self.poisoned = true;
            if self.commit_first {
                self.inner.append(event)?;
            }
            return Err(WorkerError::ControlStore(
                "lost terminal acknowledgment".into(),
            ));
        }
        self.inner.append(event)
    }
}

struct Counted(AtomicUsize);
impl ModelProvider for Counted {
    fn id(&self) -> &'static str {
        "counted"
    }
    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        EchoProvider::new().run(request, run)
    }
}

fn worker<S: JobControlStore>(store: S) -> WorkerManager<S> {
    WorkerManager::new(
        identity(),
        [capability()],
        trust(),
        WorkerLimits::default(),
        store,
    )
    .unwrap()
}

fn terminal_failure(commit_first: bool) {
    let path = Scratch(abi_foundation::temp_path::temp_file_path(
        "worker-terminal-loss",
        "log",
    ));
    let provider = Counted(AtomicUsize::new(0));
    let request = envelope();
    let store = TerminalFailure {
        inner: FileControlStore::create(&path.0, FileControlStoreLimits::default()).unwrap(),
        commit_first,
        poisoned: false,
    };
    let mut manager = worker(store);
    assert!(matches!(
        manager.infer(&request, &provider, || NOW),
        Err(WorkerError::ControlStore(_))
    ));
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
    assert!(matches!(
        manager.infer(&request, &provider, || NOW).unwrap(),
        InferenceOutcome::InProgress(_)
    ));
    let before = std::fs::read(&path.0).unwrap();
    drop(manager);
    let mut recovered =
        worker(FileControlStore::open(&path.0, FileControlStoreLimits::default()).unwrap());
    if commit_first {
        let RecordedExecution::Completed(result) = recovered.recorded(&request).unwrap() else {
            panic!("committed terminal must be recovered")
        };
        assert_eq!(result.chunks[0].bytes, b"hello");
        assert_eq!(
            recovered.infer(&request, &provider, || NOW).unwrap(),
            InferenceOutcome::Cached(result)
        );
    } else {
        assert!(matches!(
            recovered.infer(&request, &provider, || NOW).unwrap(),
            InferenceOutcome::InProgress(_)
        ));
        assert_eq!(
            recovered
                .recover_interrupted(&request, NOW + 20_000)
                .unwrap(),
            RecordedExecution::Unresolved
        );
        assert!(recovered.infer(&request, &provider, || NOW).is_err());
    }
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
    if commit_first {
        assert_eq!(std::fs::read(&path.0).unwrap(), before);
    }
}

#[test]
fn persisted_terminal_with_lost_ack_reopens_as_cached_without_execution() {
    terminal_failure(true);
}

#[test]
fn missing_terminal_reopens_as_unresolved_without_execution() {
    terminal_failure(false);
}
