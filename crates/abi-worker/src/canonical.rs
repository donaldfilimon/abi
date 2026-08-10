//! Deterministic manager-owned request, signature, and result encodings.

use crate::{CancellationRequest, JobEnvelope, JobRequest, JobStatus, ResultChunk, WorkerError};
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;

#[derive(Serialize)]
struct CanonicalRequest<'a> {
    capability: &'a str,
    capability_revision: &'a str,
    operation: &'a str,
    input: Value,
    max_result_bytes: u64,
    max_stream_chunks: u32,
}

#[derive(Serialize)]
struct IdempotencyRequest<'a> {
    principal: &'a str,
    audience: &'a str,
    namespace: &'a str,
    coordinator_audience: &'a str,
    request: CanonicalRequest<'a>,
}

#[derive(Serialize)]
struct UnsignedEnvelope<'a> {
    format_version: u16,
    job_id: uuid::Uuid,
    principal: &'a str,
    audience: &'a str,
    namespace: &'a str,
    coordinator_audience: &'a str,
    submitted_at_unix_ms: i64,
    deadline_unix_ms: i64,
    nonce: uuid::Uuid,
    request: CanonicalRequest<'a>,
    signer_key_id: &'a str,
}

#[derive(Serialize)]
struct UnsignedCancellation<'a> {
    format_version: u16,
    job_id: uuid::Uuid,
    request_digest: &'a str,
    principal: &'a str,
    audience: &'a str,
    namespace: &'a str,
    coordinator_audience: &'a str,
    requested_at_unix_ms: i64,
    reason: &'a str,
    signer_key_id: &'a str,
}

#[derive(Serialize)]
struct CanonicalResult<'a> {
    job_id: uuid::Uuid,
    request_digest: &'a str,
    status: JobStatus,
    chunks: &'a [ResultChunk],
    completed_at_unix_ms: i64,
}

#[cfg(test)]
pub(crate) fn request_bytes(request: &JobRequest) -> Result<Vec<u8>, WorkerError> {
    serialize(&canonical_request(request), "canonical request")
}

pub(crate) fn idempotency_bytes(envelope: &JobEnvelope) -> Result<Vec<u8>, WorkerError> {
    serialize(
        &IdempotencyRequest {
            principal: &envelope.principal,
            audience: &envelope.audience,
            namespace: &envelope.namespace,
            coordinator_audience: &envelope.coordinator_audience,
            request: canonical_request(&envelope.request),
        },
        "canonical idempotency request",
    )
}

pub(crate) fn envelope_bytes(envelope: &JobEnvelope) -> Result<Vec<u8>, WorkerError> {
    serialize(
        &UnsignedEnvelope {
            format_version: envelope.format_version,
            job_id: envelope.job_id,
            principal: &envelope.principal,
            audience: &envelope.audience,
            namespace: &envelope.namespace,
            coordinator_audience: &envelope.coordinator_audience,
            submitted_at_unix_ms: envelope.submitted_at_unix_ms,
            deadline_unix_ms: envelope.deadline_unix_ms,
            nonce: envelope.nonce,
            request: canonical_request(&envelope.request),
            signer_key_id: &envelope.signer_key_id,
        },
        "unsigned job envelope",
    )
}

pub(crate) fn cancellation_bytes(request: &CancellationRequest) -> Result<Vec<u8>, WorkerError> {
    serialize(
        &UnsignedCancellation {
            format_version: request.format_version,
            job_id: request.job_id,
            request_digest: request.request_digest.as_str(),
            principal: &request.principal,
            audience: &request.audience,
            namespace: &request.namespace,
            coordinator_audience: &request.coordinator_audience,
            requested_at_unix_ms: request.requested_at_unix_ms,
            reason: &request.reason,
            signer_key_id: &request.signer_key_id,
        },
        "unsigned cancellation",
    )
}

pub(crate) fn result_bytes(
    job_id: uuid::Uuid,
    request_digest: &str,
    status: JobStatus,
    chunks: &[ResultChunk],
    completed_at_unix_ms: i64,
) -> Result<Vec<u8>, WorkerError> {
    serialize(
        &CanonicalResult {
            job_id,
            request_digest,
            status,
            chunks,
            completed_at_unix_ms,
        },
        "canonical result",
    )
}

pub(crate) fn digest(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in hash {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn canonical_request(request: &JobRequest) -> CanonicalRequest<'_> {
    CanonicalRequest {
        capability: &request.capability,
        capability_revision: &request.capability_revision,
        operation: &request.operation,
        input: canonical_value(&request.input),
        max_result_bytes: request.max_result_bytes,
        max_stream_chunks: request.max_stream_chunks,
    }
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonical_value).collect()),
        Value::Object(entries) => {
            let mut keys: Vec<_> = entries.keys().collect();
            keys.sort_unstable();
            let mut ordered = Map::with_capacity(entries.len());
            for key in keys {
                ordered.insert(key.clone(), canonical_value(&entries[key]));
            }
            Value::Object(ordered)
        }
        scalar => scalar.clone(),
    }
}

fn serialize(value: &impl Serialize, context: &str) -> Result<Vec<u8>, WorkerError> {
    serde_json::to_vec(value)
        .map_err(|error| WorkerError::Invalid(format!("{context} serialization failed: {error}")))
}

#[cfg(test)]
mod tests {
    use super::{digest, request_bytes};
    use crate::JobRequest;
    use serde_json::json;

    #[test]
    fn object_key_order_does_not_change_request_identity() {
        let left = JobRequest::new("model", "r1", "infer", json!({"b": 2, "a": 1}), 8, 1).unwrap();
        let right = JobRequest::new("model", "r1", "infer", json!({"a": 1, "b": 2}), 8, 1).unwrap();
        assert_eq!(
            request_bytes(&left).unwrap(),
            request_bytes(&right).unwrap()
        );
        assert_eq!(
            digest(&request_bytes(&left).unwrap()),
            digest(&request_bytes(&right).unwrap())
        );
    }
}
