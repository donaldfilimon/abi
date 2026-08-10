//! Authenticated, bounded worker-control foundations for ABI.
//!
//! This crate owns typed job, identity, capability, lease, cancellation,
//! result, digest, health, and replay contracts. A [`WorkerManager`] verifies a
//! coordinator signature, validates the complete request, derives the
//! idempotency digest from its own canonical representation, and only then
//! admits work. Exact retries return the recorded lease or result and never
//! execute twice.
//!
//! [`WorkerTransportSecurity`] reuses the WDBX gateway's owner-protected TLS
//! and mandatory-client-certificate loader. This crate does not implement a
//! second certificate parser, a network listener, scheduling, model execution,
//! or WDBX data replication. Job control is an independent [`JobControlStore`]
//! boundary; WDBX remains a separate data plane.

mod canonical;
mod crypto;
mod error;
mod manager;
mod store;
mod transport;
mod types;

pub use abi_agent_runtime::CancellationToken;
pub use crypto::CoordinatorTrustStore;
pub use error::WorkerError;
pub use manager::{Admission, WorkerManager};
pub use store::{ControlEvent, JobControlStore, MemoryControlStore};
pub use transport::WorkerTransportSecurity;
pub use types::{
    CancellationRequest, HealthReport, JobEnvelope, JobLease, JobRequest, JobResult, JobStatus,
    ResultChunk, ResultDigest, WorkerCapability, WorkerIdentity, WorkerLimits,
};
