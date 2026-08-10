//! Independent append-only worker control storage.

use crate::{JobLease, JobResult, ResultDigest, WorkerError};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Durable control transition. No WDBX data-plane record is used here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlEvent {
    /// A signed job passed admission and received a lease.
    Admitted {
        /// Job identity.
        job_id: Uuid,
        /// Signed replay nonce.
        nonce: Uuid,
        /// Charged principal.
        principal: String,
        /// Exact capability identity.
        capability: String,
        /// Manager-derived request identity.
        request_digest: ResultDigest,
        /// Digest of the exact signed envelope fields admitted for this job.
        envelope_digest: ResultDigest,
        /// Finite worker authority.
        lease: JobLease,
        /// Absolute job deadline.
        deadline_unix_ms: i64,
        /// Signed per-job result byte ceiling.
        max_result_bytes: u64,
        /// Signed per-job result chunk ceiling.
        max_stream_chunks: u32,
    },
    /// A lease produced one hash-bound terminal result.
    Completed {
        /// Exact terminal result.
        result: JobResult,
    },
    /// A signed cancellation terminally stopped an admitted job.
    Cancelled {
        /// Job identity.
        job_id: Uuid,
        /// Exact manager-derived request identity.
        request_digest: ResultDigest,
        /// Cancellation time.
        cancelled_at_unix_ms: i64,
    },
}

/// Append-only persistence boundary for worker admission and replay state.
///
/// Implementations belong to the worker control plane. They must not treat
/// WDBX replication or membership state as job-control authority.
pub trait JobControlStore: Send {
    /// Replay every previously committed event in append order.
    fn load(&self) -> Result<Vec<ControlEvent>, WorkerError>;
    /// Durably append one event before the manager publishes it in memory.
    fn append(&mut self, event: &ControlEvent) -> Result<(), WorkerError>;
}

/// Deterministic in-memory control store for offline tests and embedding.
#[derive(Clone, Debug, Default)]
pub struct MemoryControlStore {
    events: Vec<ControlEvent>,
}

impl MemoryControlStore {
    /// Empty append-only store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current committed events in append order.
    #[must_use]
    pub fn events(&self) -> &[ControlEvent] {
        &self.events
    }
}

impl JobControlStore for MemoryControlStore {
    fn load(&self) -> Result<Vec<ControlEvent>, WorkerError> {
        Ok(self.events.clone())
    }

    fn append(&mut self, event: &ControlEvent) -> Result<(), WorkerError> {
        self.events.push(event.clone());
        Ok(())
    }
}
