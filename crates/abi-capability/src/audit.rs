use crate::{ApprovalLevel, Decision, Digest, ReasonCode};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Fixed redacted audit record. It intentionally has no raw-content field.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuditRecord {
    /// Attempt identifier.
    pub attempt_id: String,
    /// Request commitment.
    pub request_digest: Digest,
    /// Principal commitment.
    pub principal_digest: Digest,
    /// Scope commitment.
    pub scope_digest: Digest,
    /// Package commitment.
    pub package_digest: Digest,
    /// Grant commitment when a candidate grant was evaluated.
    pub grant_digest: Option<Digest>,
    /// Pipeline stage (0 through 16).
    pub stage: u8,
    /// Closed decision.
    pub decision: Decision,
    /// Closed reason.
    pub reason: ReasonCode,
    /// Parameter commitment.
    pub parameter_digest: Digest,
    /// Required approval level.
    pub approval_required: ApprovalLevel,
    /// Whether cancellation was observed.
    pub cancelled: bool,
    /// Whether the record is redacted.
    pub redacted: bool,
}

/// Fallible audit boundary. Failure must prevent actuation.
pub trait AuditSink {
    /// Persist one bounded record.
    fn record(&mut self, record: &AuditRecord) -> Result<(), AuditError>;
}

/// Bounded in-memory sink for tests, replay, and recording profiles only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundedMemoryAuditSink {
    max_records: usize,
    max_bytes: usize,
    bytes: usize,
    records: Vec<AuditRecord>,
}

impl BoundedMemoryAuditSink {
    /// Construct explicit count and serialized-byte ceilings.
    #[must_use]
    pub const fn new(max_records: usize, max_bytes: usize) -> Self {
        Self {
            max_records,
            max_bytes,
            bytes: 0,
            records: Vec::new(),
        }
    }

    /// Observe recorded redacted entries.
    #[must_use]
    pub fn records(&self) -> &[AuditRecord] {
        &self.records
    }
}

impl AuditSink for BoundedMemoryAuditSink {
    fn record(&mut self, record: &AuditRecord) -> Result<(), AuditError> {
        let encoded = serde_json::to_vec(record).map_err(|_| AuditError::Encoding)?;
        if self.records.len() >= self.max_records
            || self.bytes.saturating_add(encoded.len()) > self.max_bytes
        {
            return Err(AuditError::Capacity);
        }
        self.bytes += encoded.len();
        self.records.push(record.clone());
        Ok(())
    }
}

/// Closed audit failures without record contents.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AuditError {
    /// Record or byte capacity is exhausted.
    #[error("audit_capacity")]
    Capacity,
    /// Serialization failed.
    #[error("audit_encoding")]
    Encoding,
    /// Injected persistence failure.
    #[error("audit_unavailable")]
    Unavailable,
}
