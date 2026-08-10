//! Public worker protocol and evidence types.

use crate::WorkerError;
use ed25519_dalek::{Signer as _, SigningKey};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

pub(crate) const FORMAT_VERSION: u16 = 1;
pub(crate) const MAX_ID_BYTES: usize = 128;
pub(crate) const MAX_LABEL_BYTES: usize = 256;
pub(crate) const MAX_OPERATION_BYTES: usize = 256;
pub(crate) const MAX_REASON_BYTES: usize = 2_048;
pub(crate) const MAX_CAPABILITIES: usize = 256;
const MAX_INPUT_DEPTH: usize = 64;
const MAX_INPUT_NODES: usize = 100_000;
const MAX_INPUT_KEY_BYTES: usize = 1_024;

/// Immutable digest rendered as 64 lowercase hexadecimal characters.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ResultDigest(String);

impl ResultDigest {
    /// Parse a strict lowercase SHA-256 digest.
    pub fn parse(value: impl Into<String>) -> Result<Self, WorkerError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(WorkerError::Invalid(
                "digest must be 64 lowercase hexadecimal characters".into(),
            ));
        }
        Ok(Self(value))
    }

    pub(crate) fn from_hash(value: String) -> Self {
        Self(value)
    }

    /// Lowercase hexadecimal digest.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ResultDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ResultDigest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

/// One operation family and immutable implementation revision a worker offers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerCapability {
    /// Stable capability identifier.
    pub id: String,
    /// Immutable implementation or model revision.
    pub revision: String,
    /// Maximum simultaneously active jobs for this capability.
    pub max_concurrency: u32,
    /// Bounded descriptive labels; never credentials.
    pub labels: BTreeMap<String, String>,
}

impl WorkerCapability {
    /// Construct a capability and validate its stable identity.
    pub fn new(
        id: impl Into<String>,
        revision: impl Into<String>,
        max_concurrency: u32,
    ) -> Result<Self, WorkerError> {
        let capability = Self {
            id: id.into(),
            revision: revision.into(),
            max_concurrency,
            labels: BTreeMap::new(),
        };
        capability.validate()?;
        Ok(capability)
    }

    /// Add one bounded non-secret descriptive label.
    pub fn with_label(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, WorkerError> {
        self.labels.insert(key.into(), value.into());
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn validate(&self) -> Result<(), WorkerError> {
        validate_id("capability id", &self.id)?;
        validate_id("capability revision", &self.revision)?;
        if self.max_concurrency == 0 {
            return Err(WorkerError::Invalid(
                "capability concurrency must be nonzero".into(),
            ));
        }
        if self.labels.len() > 32
            || self.labels.iter().any(|(key, value)| {
                key.is_empty()
                    || key.len() > MAX_LABEL_BYTES
                    || value.len() > MAX_LABEL_BYTES
                    || key.chars().any(char::is_control)
                    || value.chars().any(char::is_control)
            })
        {
            return Err(WorkerError::Invalid(
                "capability labels exceed count or text bounds".into(),
            ));
        }
        Ok(())
    }
}

/// Startup-owned worker identity and accepted coordinator audiences.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerIdentity {
    /// Stable worker identity, also the required job audience.
    pub worker_id: String,
    /// Edition or deployment namespace.
    pub namespace: String,
    /// Coordinator identities this worker accepts as issuers.
    pub coordinator_audiences: BTreeSet<String>,
}

impl WorkerIdentity {
    /// Build an audience-bound worker identity.
    pub fn new(
        worker_id: impl Into<String>,
        namespace: impl Into<String>,
        audiences: impl IntoIterator<Item = String>,
    ) -> Result<Self, WorkerError> {
        let identity = Self {
            worker_id: worker_id.into(),
            namespace: namespace.into(),
            coordinator_audiences: audiences.into_iter().collect(),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub(crate) fn validate(&self) -> Result<(), WorkerError> {
        validate_id("worker id", &self.worker_id)?;
        validate_id("worker namespace", &self.namespace)?;
        if self.coordinator_audiences.is_empty()
            || self.coordinator_audiences.len() > 32
            || self
                .coordinator_audiences
                .iter()
                .any(|audience| validate_id("coordinator audience", audience).is_err())
        {
            return Err(WorkerError::Invalid(
                "worker must declare bounded coordinator audiences".into(),
            ));
        }
        Ok(())
    }
}

/// Validated operation request whose canonical form determines idempotency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobRequest {
    /// Exact capability identifier.
    pub capability: String,
    /// Exact capability revision; incompatible revisions fail closed.
    pub capability_revision: String,
    /// Startup-interpreted operation identifier, never an executable path.
    pub operation: String,
    /// Structured input. Object keys are sorted before hashing.
    pub input: serde_json::Value,
    /// Requested total result byte ceiling.
    pub max_result_bytes: u64,
    /// Requested result stream chunk ceiling.
    pub max_stream_chunks: u32,
}

impl JobRequest {
    /// Construct a structured request with explicit bounds.
    pub fn new(
        capability: impl Into<String>,
        capability_revision: impl Into<String>,
        operation: impl Into<String>,
        input: serde_json::Value,
        max_result_bytes: u64,
        max_stream_chunks: u32,
    ) -> Result<Self, WorkerError> {
        let request = Self {
            capability: capability.into(),
            capability_revision: capability_revision.into(),
            operation: operation.into(),
            input,
            max_result_bytes,
            max_stream_chunks,
        };
        request.validate()?;
        Ok(request)
    }

    pub(crate) fn validate(&self) -> Result<(), WorkerError> {
        validate_id("request capability", &self.capability)?;
        validate_id("request capability revision", &self.capability_revision)?;
        if self.operation.is_empty()
            || self.operation.len() > MAX_OPERATION_BYTES
            || self.operation.chars().any(char::is_control)
        {
            return Err(WorkerError::Invalid(
                "operation must be bounded non-control text".into(),
            ));
        }
        if self.max_result_bytes == 0 || self.max_stream_chunks == 0 {
            return Err(WorkerError::Invalid(
                "request result bounds must be nonzero".into(),
            ));
        }
        let mut nodes = 0;
        validate_json_complexity(&self.input, 0, &mut nodes)?;
        Ok(())
    }
}

/// Coordinator-authenticated job submission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobEnvelope {
    /// Frozen envelope format.
    pub format_version: u16,
    /// Coordinator-selected correlation identity.
    pub job_id: Uuid,
    /// Principal charged against quotas and recorded in control history.
    pub principal: String,
    /// Exact target worker identity.
    pub audience: String,
    /// Exact target namespace.
    pub namespace: String,
    /// Coordinator identity accepted by worker startup configuration.
    pub coordinator_audience: String,
    /// Wall-clock submission time.
    pub submitted_at_unix_ms: i64,
    /// Absolute execution deadline.
    pub deadline_unix_ms: i64,
    /// Unique signed replay identity.
    pub nonce: Uuid,
    /// Structured operation request.
    pub request: JobRequest,
    /// Configured coordinator signing-key identifier.
    pub signer_key_id: String,
    /// Ed25519 signature over the manager-canonicalized unsigned envelope.
    pub signature: Vec<u8>,
}

impl JobEnvelope {
    /// Sign a complete job envelope with one configured coordinator key.
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        job_id: Uuid,
        principal: impl Into<String>,
        audience: impl Into<String>,
        namespace: impl Into<String>,
        coordinator_audience: impl Into<String>,
        submitted_at_unix_ms: i64,
        deadline_unix_ms: i64,
        nonce: Uuid,
        request: JobRequest,
        signer_key_id: impl Into<String>,
        signing_key: &SigningKey,
    ) -> Result<Self, WorkerError> {
        let mut envelope = Self {
            format_version: FORMAT_VERSION,
            job_id,
            principal: principal.into(),
            audience: audience.into(),
            namespace: namespace.into(),
            coordinator_audience: coordinator_audience.into(),
            submitted_at_unix_ms,
            deadline_unix_ms,
            nonce,
            request,
            signer_key_id: signer_key_id.into(),
            signature: Vec::new(),
        };
        envelope.validate_shape()?;
        envelope.signature = signing_key
            .sign(&crate::canonical::envelope_bytes(&envelope)?)
            .to_bytes()
            .to_vec();
        Ok(envelope)
    }

    pub(crate) fn validate_shape(&self) -> Result<(), WorkerError> {
        if self.format_version != FORMAT_VERSION || self.job_id.is_nil() || self.nonce.is_nil() {
            return Err(WorkerError::Invalid(
                "unsupported format or nil job/nonce identity".into(),
            ));
        }
        validate_id("principal", &self.principal)?;
        validate_id("job audience", &self.audience)?;
        validate_id("job namespace", &self.namespace)?;
        validate_id("coordinator audience", &self.coordinator_audience)?;
        validate_id("signer key id", &self.signer_key_id)?;
        if self.deadline_unix_ms <= self.submitted_at_unix_ms {
            return Err(WorkerError::Invalid(
                "job deadline must follow submission time".into(),
            ));
        }
        self.request.validate()
    }
}

/// Worker-issued finite authority to complete one admitted job.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobLease {
    /// Job correlation identity.
    pub job_id: Uuid,
    /// Unpredictable lease identity.
    pub lease_id: Uuid,
    /// Manager-derived canonical request digest.
    pub request_digest: ResultDigest,
    /// Worker identity that issued this lease.
    pub worker_id: String,
    /// Absolute lease expiry, never later than the job deadline.
    pub expires_at_unix_ms: i64,
}

/// One ordered result-stream chunk.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultChunk {
    /// Zero-based contiguous stream sequence.
    pub sequence: u32,
    /// Opaque result bytes.
    pub bytes: Vec<u8>,
}

/// Terminal job state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Executor completed successfully.
    Succeeded,
    /// Executor completed with a bounded failure payload.
    Failed,
    /// Coordinator cancelled the job.
    Cancelled,
}

/// Hash-bound terminal result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobResult {
    /// Job correlation identity.
    pub job_id: Uuid,
    /// Manager-derived request identity.
    pub request_digest: ResultDigest,
    /// Result terminal state.
    pub status: JobStatus,
    /// Ordered bounded output stream.
    pub chunks: Vec<ResultChunk>,
    /// SHA-256 over the canonical terminal result fields and chunks.
    pub result_digest: ResultDigest,
    /// Completion timestamp.
    pub completed_at_unix_ms: i64,
}

impl JobResult {
    /// Verify that the published digest binds the complete terminal result.
    #[must_use]
    pub fn verify_digest(&self) -> bool {
        crate::canonical::result_bytes(
            self.job_id,
            self.request_digest.as_str(),
            self.status,
            &self.chunks,
            self.completed_at_unix_ms,
        )
        .is_ok_and(|canonical| self.result_digest.as_str() == crate::canonical::digest(&canonical))
    }
}

/// Signed coordinator request to cancel one exact admitted job.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancellationRequest {
    /// Frozen cancellation format.
    pub format_version: u16,
    /// Job correlation identity.
    pub job_id: Uuid,
    /// Exact manager-derived request digest.
    pub request_digest: ResultDigest,
    /// Principal that owns the job.
    pub principal: String,
    /// Exact worker audience.
    pub audience: String,
    /// Exact worker namespace.
    pub namespace: String,
    /// Coordinator audience cryptographically bound to the signing key.
    pub coordinator_audience: String,
    /// Cancellation time.
    pub requested_at_unix_ms: i64,
    /// Bounded operator-facing reason.
    pub reason: String,
    /// Configured coordinator key identity.
    pub signer_key_id: String,
    /// Ed25519 signature over canonical unsigned cancellation fields.
    pub signature: Vec<u8>,
}

impl CancellationRequest {
    /// Build and sign an exact cancellation request.
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        job_id: Uuid,
        request_digest: ResultDigest,
        principal: impl Into<String>,
        audience: impl Into<String>,
        namespace: impl Into<String>,
        coordinator_audience: impl Into<String>,
        requested_at_unix_ms: i64,
        reason: impl Into<String>,
        signer_key_id: impl Into<String>,
        signing_key: &SigningKey,
    ) -> Result<Self, WorkerError> {
        let mut request = Self {
            format_version: FORMAT_VERSION,
            job_id,
            request_digest,
            principal: principal.into(),
            audience: audience.into(),
            namespace: namespace.into(),
            coordinator_audience: coordinator_audience.into(),
            requested_at_unix_ms,
            reason: reason.into(),
            signer_key_id: signer_key_id.into(),
            signature: Vec::new(),
        };
        request.validate_shape()?;
        request.signature = signing_key
            .sign(&crate::canonical::cancellation_bytes(&request)?)
            .to_bytes()
            .to_vec();
        Ok(request)
    }

    pub(crate) fn validate_shape(&self) -> Result<(), WorkerError> {
        if self.format_version != FORMAT_VERSION || self.job_id.is_nil() {
            return Err(WorkerError::Invalid(
                "unsupported cancellation format or nil job identity".into(),
            ));
        }
        validate_id("cancellation principal", &self.principal)?;
        validate_id("cancellation audience", &self.audience)?;
        validate_id("cancellation namespace", &self.namespace)?;
        validate_id(
            "cancellation coordinator audience",
            &self.coordinator_audience,
        )?;
        validate_id("cancellation signer", &self.signer_key_id)?;
        if self.reason.is_empty()
            || self.reason.len() > MAX_REASON_BYTES
            || self.reason.chars().any(char::is_control)
        {
            return Err(WorkerError::Invalid(
                "cancellation reason exceeds text bounds".into(),
            ));
        }
        Ok(())
    }
}

/// Finite admission and output policy owned by worker startup configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerLimits {
    /// Maximum encoded signed envelope bytes.
    pub envelope_bytes: usize,
    /// Maximum canonical input JSON bytes.
    pub input_bytes: usize,
    /// Maximum result bytes for any one job.
    pub result_bytes: usize,
    /// Maximum result chunks for any one job.
    pub stream_chunks: u32,
    /// Maximum total active jobs.
    pub active_jobs: usize,
    /// Maximum active jobs charged to one principal.
    pub active_jobs_per_principal: usize,
    /// Maximum retained control events.
    pub control_events: usize,
    /// Lease lifetime in milliseconds.
    pub lease_ms: i64,
    /// Accepted future submission skew in milliseconds.
    pub clock_skew_ms: i64,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            envelope_bytes: 1024 * 1024,
            input_bytes: 512 * 1024,
            result_bytes: 1024 * 1024,
            stream_chunks: 128,
            active_jobs: 8,
            active_jobs_per_principal: 2,
            control_events: 100_000,
            lease_ms: 30_000,
            clock_skew_ms: 30_000,
        }
    }
}

impl WorkerLimits {
    pub(crate) fn validate(self) -> Result<(), WorkerError> {
        if self.envelope_bytes == 0
            || self.input_bytes == 0
            || self.result_bytes == 0
            || self.stream_chunks == 0
            || self.active_jobs == 0
            || self.active_jobs_per_principal == 0
            || self.control_events == 0
            || self.lease_ms <= 0
            || self.clock_skew_ms < 0
        {
            return Err(WorkerError::Invalid(
                "all worker limits must be positive except nonnegative clock skew".into(),
            ));
        }
        Ok(())
    }
}

/// Bounded operational snapshot of one manager.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HealthReport {
    /// Worker identity.
    pub worker_id: String,
    /// Worker namespace.
    pub namespace: String,
    /// Configured capabilities in stable order.
    pub capabilities: Vec<WorkerCapability>,
    /// Currently leased jobs.
    pub active_jobs: usize,
    /// Completed jobs retained for idempotent retries.
    pub completed_jobs: usize,
    /// Cancelled jobs retained for replay protection.
    pub cancelled_jobs: usize,
    /// Report generation time.
    pub observed_at_unix_ms: i64,
}

pub(crate) fn validate_id(label: &str, value: &str) -> Result<(), WorkerError> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
    {
        return Err(WorkerError::Invalid(format!(
            "{label} must be bounded printable non-whitespace ASCII"
        )));
    }
    Ok(())
}

fn validate_json_complexity(
    value: &serde_json::Value,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), WorkerError> {
    *nodes = nodes.saturating_add(1);
    if depth > MAX_INPUT_DEPTH || *nodes > MAX_INPUT_NODES {
        return Err(WorkerError::Invalid(
            "request input exceeds JSON depth or node bounds".into(),
        ));
    }
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                validate_json_complexity(value, depth + 1, nodes)?;
            }
        }
        serde_json::Value::Object(entries) => {
            if entries
                .keys()
                .any(|key| key.len() > MAX_INPUT_KEY_BYTES || key.chars().any(char::is_control))
            {
                return Err(WorkerError::Invalid(
                    "request input contains an oversized or control-character object key".into(),
                ));
            }
            for value in entries.values() {
                validate_json_complexity(value, depth + 1, nodes)?;
            }
        }
        _ => {}
    }
    Ok(())
}
