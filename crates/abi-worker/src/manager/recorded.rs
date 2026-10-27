//! Historical evidence and explicit abandonment, independent of admission time.
use super::{JobState, WorkerManager};
use crate::canonical::{digest, envelope_bytes, idempotency_bytes};
use crate::{
    ControlEvent, JobControlStore, JobEnvelope, JobLease, JobResult, ResultDigest, WorkerError,
};
use uuid::Uuid;

/// Durable evidence for one exact authenticated historical envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordedExecution {
    /// No job, nonce, or canonical request was previously admitted.
    Missing,
    /// Admission exists; this evidence grants no new execution authority.
    Admitted(JobLease),
    /// Original committed result, including its original completion timestamp.
    Completed(JobResult),
    /// Signed cancellation was committed.
    Cancelled,
    /// Prior-process execution was abandoned with no known terminal result.
    Unresolved,
}

impl<S: JobControlStore> WorkerManager<S> {
    /// Authenticate exact historical identity without time checks or store writes.
    /// Current model configuration does not authorize or invalidate old evidence.
    pub fn recorded(&self, envelope: &JobEnvelope) -> Result<RecordedExecution, WorkerError> {
        envelope.validate_shape()?;
        let encoded = serde_json::to_vec(envelope)
            .map_err(|error| WorkerError::Invalid(format!("envelope serialization: {error}")))?;
        if encoded.len() > self.limits.envelope_bytes {
            return Err(WorkerError::Quota(
                "signed envelope exceeds transport byte ceiling".into(),
            ));
        }
        self.trust.verify_envelope(envelope)?;
        if envelope.audience != self.identity.worker_id
            || envelope.namespace != self.identity.namespace
            || !self
                .identity
                .coordinator_audiences
                .contains(&envelope.coordinator_audience)
        {
            return Err(WorkerError::Authentication(
                "historical envelope scope differs".into(),
            ));
        }
        let request = idempotency_bytes(envelope)?;
        if request.len() > self.limits.input_bytes {
            return Err(WorkerError::Quota(
                "canonical request exceeds input byte ceiling".into(),
            ));
        }
        let request_digest = ResultDigest::from_hash(digest(&request));
        let envelope_digest = ResultDigest::from_hash(digest(&envelope_bytes(envelope)?));
        let Some(entry) = self.jobs.get(&envelope.job_id) else {
            if self.nonces.contains_key(&envelope.nonce)
                || self.digests.contains_key(&request_digest)
            {
                return Err(WorkerError::Replay(
                    "historical identity belongs to another job".into(),
                ));
            }
            return Ok(RecordedExecution::Missing);
        };
        if entry.request_digest != request_digest
            || entry.envelope_digest != envelope_digest
            || entry.nonce != envelope.nonce
            || entry.principal != envelope.principal
        {
            return Err(WorkerError::Replay(
                "historical envelope differs from admission".into(),
            ));
        }
        Ok(match &entry.state {
            JobState::Active => RecordedExecution::Admitted(entry.lease.clone()),
            JobState::Completed(result) => RecordedExecution::Completed(result.clone()),
            JobState::Cancelled => RecordedExecution::Cancelled,
            JobState::Unresolved => RecordedExecution::Unresolved,
        })
    }

    /// Durably abandon only admissions replayed from a previous manager process.
    /// Call under exclusive startup ownership, never to stop live inference.
    pub fn recover_interrupted(
        &mut self,
        envelope: &JobEnvelope,
        observed_at_unix_ms: i64,
    ) -> Result<RecordedExecution, WorkerError> {
        let recorded = self.recorded(envelope)?;
        let RecordedExecution::Admitted(lease) = recorded else {
            return Ok(recorded);
        };
        if !self.jobs[&envelope.job_id].replayed {
            return Err(WorkerError::State(
                "current-process admission cannot be abandoned as replayed".into(),
            ));
        }
        self.ensure_event_capacity()?;
        self.store.append(&ControlEvent::Unresolved {
            job_id: lease.job_id,
            request_digest: lease.request_digest.clone(),
            observed_at_unix_ms,
        })?;
        self.apply_unresolved(lease.job_id, &lease.request_digest)?;
        self.event_count += 1;
        Ok(RecordedExecution::Unresolved)
    }

    pub(super) fn apply_unresolved(
        &mut self,
        job_id: Uuid,
        request_digest: &ResultDigest,
    ) -> Result<(), WorkerError> {
        let entry = self.jobs.get_mut(&job_id).ok_or_else(|| {
            WorkerError::ControlStore("unresolved history has no admission".into())
        })?;
        if entry.request_digest != *request_digest || !matches!(entry.state, JobState::Active) {
            return Err(WorkerError::ControlStore(
                "unresolved history conflicts with admission".into(),
            ));
        }
        entry.cancellation.cancel();
        entry.state = JobState::Unresolved;
        Ok(())
    }
}
