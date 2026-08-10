//! Authenticated admission, idempotency, lease, quota, and result state machine.

use crate::canonical::{digest, idempotency_bytes, result_bytes};
use crate::store::{ControlEvent, JobControlStore};
use crate::{
    CancellationRequest, CancellationToken, CoordinatorTrustStore, HealthReport, JobEnvelope,
    JobLease, JobResult, JobStatus, ResultChunk, ResultDigest, WorkerCapability, WorkerError,
    WorkerIdentity, WorkerLimits,
};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Outcome of submitting a verified job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Admission {
    /// New request received a finite lease and may execute once.
    Leased(JobLease),
    /// Exact signed retry found the same still-active lease.
    ExistingLease(JobLease),
    /// Exact canonical request already completed; do not execute again.
    Cached(JobResult),
}

#[derive(Clone, Debug)]
struct JobEntry {
    nonce: Uuid,
    principal: String,
    capability: String,
    request_digest: ResultDigest,
    envelope_digest: ResultDigest,
    lease: JobLease,
    deadline_unix_ms: i64,
    max_result_bytes: usize,
    max_stream_chunks: u32,
    cancellation: CancellationToken,
    state: JobState,
}

#[derive(Clone, Debug)]
enum JobState {
    Active,
    Completed(JobResult),
    Cancelled,
}

/// Startup-owned authenticated worker manager over independent control storage.
pub struct WorkerManager<S> {
    identity: WorkerIdentity,
    capabilities: BTreeMap<String, WorkerCapability>,
    trust: CoordinatorTrustStore,
    limits: WorkerLimits,
    store: S,
    jobs: BTreeMap<Uuid, JobEntry>,
    digests: BTreeMap<ResultDigest, Uuid>,
    nonces: BTreeMap<Uuid, Uuid>,
    event_count: usize,
}

impl<S: JobControlStore> WorkerManager<S> {
    /// Validate startup authority and replay the independent control log.
    pub fn new(
        identity: WorkerIdentity,
        capabilities: impl IntoIterator<Item = WorkerCapability>,
        trust: CoordinatorTrustStore,
        limits: WorkerLimits,
        store: S,
    ) -> Result<Self, WorkerError> {
        identity.validate()?;
        limits.validate()?;
        let mut capability_map = BTreeMap::new();
        for capability in capabilities {
            capability.validate()?;
            if capability_map
                .insert(capability.id.clone(), capability)
                .is_some()
            {
                return Err(WorkerError::Invalid(
                    "worker capability ids must be unique".into(),
                ));
            }
        }
        if capability_map.is_empty() || capability_map.len() > crate::types::MAX_CAPABILITIES {
            return Err(WorkerError::Invalid(
                "worker requires a bounded nonempty capability set".into(),
            ));
        }
        let events = store.load()?;
        if events.len() > limits.control_events {
            return Err(WorkerError::Quota(
                "control history exceeds configured event ceiling".into(),
            ));
        }
        let mut manager = Self {
            identity,
            capabilities: capability_map,
            trust,
            limits,
            store,
            jobs: BTreeMap::new(),
            digests: BTreeMap::new(),
            nonces: BTreeMap::new(),
            event_count: 0,
        };
        for event in events {
            manager.replay(event)?;
            manager.event_count += 1;
        }
        manager.validate_replayed_capacity()?;
        Ok(manager)
    }

    /// Authenticate and admit one job, or return prior exact retry evidence.
    pub fn submit(
        &mut self,
        envelope: &JobEnvelope,
        now_unix_ms: i64,
    ) -> Result<Admission, WorkerError> {
        envelope.validate_shape()?;
        let encoded = serde_json::to_vec(envelope)
            .map_err(|error| WorkerError::Invalid(format!("envelope serialization: {error}")))?;
        if encoded.len() > self.limits.envelope_bytes {
            return Err(WorkerError::Quota(
                "signed envelope exceeds transport byte ceiling".into(),
            ));
        }
        self.trust.verify_envelope(envelope)?;
        self.validate_authority(envelope)?;
        self.validate_time(envelope, now_unix_ms)?;
        let request_bytes = idempotency_bytes(envelope)?;
        if request_bytes.len() > self.limits.input_bytes {
            return Err(WorkerError::Quota(
                "canonical request exceeds input byte ceiling".into(),
            ));
        }
        let request_digest = ResultDigest::from_hash(digest(&request_bytes));
        let envelope_digest =
            ResultDigest::from_hash(digest(&crate::canonical::envelope_bytes(envelope)?));
        if let Some(admission) =
            self.retry(envelope, &request_digest, &envelope_digest, now_unix_ms)?
        {
            return Ok(admission);
        }
        self.validate_request_quotas(envelope)?;
        self.ensure_admission_capacity()?;
        let lease = JobLease {
            job_id: envelope.job_id,
            lease_id: Uuid::new_v4(),
            request_digest: request_digest.clone(),
            worker_id: self.identity.worker_id.clone(),
            expires_at_unix_ms: envelope
                .deadline_unix_ms
                .min(now_unix_ms.saturating_add(self.limits.lease_ms)),
        };
        let event = ControlEvent::Admitted {
            job_id: envelope.job_id,
            nonce: envelope.nonce,
            principal: envelope.principal.clone(),
            capability: envelope.request.capability.clone(),
            request_digest,
            envelope_digest,
            lease: lease.clone(),
            deadline_unix_ms: envelope.deadline_unix_ms,
            max_result_bytes: envelope.request.max_result_bytes,
            max_stream_chunks: envelope.request.max_stream_chunks,
        };
        self.store.append(&event)?;
        self.apply_admission(event)?;
        self.event_count += 1;
        Ok(Admission::Leased(lease))
    }

    /// Complete one exact live lease with a bounded ordered result stream.
    pub fn complete(
        &mut self,
        lease: &JobLease,
        status: JobStatus,
        chunks: Vec<ResultChunk>,
        now_unix_ms: i64,
    ) -> Result<JobResult, WorkerError> {
        if status == JobStatus::Cancelled {
            return Err(WorkerError::State(
                "cancelled status requires a signed cancellation request".into(),
            ));
        }
        let entry = self
            .jobs
            .get(&lease.job_id)
            .ok_or_else(|| WorkerError::State("completion references an unknown job".into()))?;
        if entry.lease != *lease || !matches!(entry.state, JobState::Active) {
            return Err(WorkerError::State(
                "completion lease is stale, mismatched, or terminal".into(),
            ));
        }
        if now_unix_ms > entry.lease.expires_at_unix_ms || now_unix_ms > entry.deadline_unix_ms {
            return Err(WorkerError::Deadline(
                "completion arrived after lease or job deadline".into(),
            ));
        }
        validate_chunks(
            &chunks,
            entry.max_stream_chunks.min(self.limits.stream_chunks),
            entry.max_result_bytes.min(self.limits.result_bytes),
        )?;
        self.ensure_event_capacity()?;
        let canonical = result_bytes(
            lease.job_id,
            lease.request_digest.as_str(),
            status,
            &chunks,
            now_unix_ms,
        )?;
        let result = JobResult {
            job_id: lease.job_id,
            request_digest: lease.request_digest.clone(),
            status,
            chunks,
            result_digest: ResultDigest::from_hash(digest(&canonical)),
            completed_at_unix_ms: now_unix_ms,
        };
        let event = ControlEvent::Completed {
            result: result.clone(),
        };
        self.store.append(&event)?;
        self.apply_completion(&result)?;
        self.event_count += 1;
        Ok(result)
    }

    /// Authenticate and terminally record cancellation for one exact job.
    pub fn cancel(
        &mut self,
        request: &CancellationRequest,
        now_unix_ms: i64,
    ) -> Result<(), WorkerError> {
        request.validate_shape()?;
        self.trust.verify_cancellation(request)?;
        if request.requested_at_unix_ms > now_unix_ms.saturating_add(self.limits.clock_skew_ms) {
            return Err(WorkerError::Deadline(
                "cancellation time is beyond accepted clock skew".into(),
            ));
        }
        if request.audience != self.identity.worker_id
            || request.namespace != self.identity.namespace
            || !self
                .identity
                .coordinator_audiences
                .contains(&request.coordinator_audience)
        {
            return Err(WorkerError::Authentication(
                "cancellation audience or namespace differs from worker identity".into(),
            ));
        }
        let entry = self
            .jobs
            .get(&request.job_id)
            .ok_or_else(|| WorkerError::State("cancellation references an unknown job".into()))?;
        if entry.principal != request.principal || entry.request_digest != request.request_digest {
            return Err(WorkerError::Authentication(
                "cancellation principal or request digest differs".into(),
            ));
        }
        match &entry.state {
            JobState::Active => {}
            JobState::Cancelled => return Ok(()),
            JobState::Completed(_) => {
                return Err(WorkerError::State(
                    "completed jobs cannot be retroactively cancelled".into(),
                ));
            }
        }
        self.ensure_event_capacity()?;
        let event = ControlEvent::Cancelled {
            job_id: request.job_id,
            request_digest: request.request_digest.clone(),
            cancelled_at_unix_ms: request.requested_at_unix_ms,
        };
        self.store.append(&event)?;
        self.apply_cancellation(request.job_id, &request.request_digest)?;
        self.event_count += 1;
        Ok(())
    }

    /// Bounded worker capability and control-state snapshot.
    #[must_use]
    pub fn health(&self, now_unix_ms: i64) -> HealthReport {
        let mut active_jobs = 0;
        let mut completed_jobs = 0;
        let mut cancelled_jobs = 0;
        for entry in self.jobs.values() {
            match entry.state {
                JobState::Active => active_jobs += 1,
                JobState::Completed(_) => completed_jobs += 1,
                JobState::Cancelled => cancelled_jobs += 1,
            }
        }
        HealthReport {
            worker_id: self.identity.worker_id.clone(),
            namespace: self.identity.namespace.clone(),
            capabilities: self.capabilities.values().cloned().collect(),
            active_jobs,
            completed_jobs,
            cancelled_jobs,
            observed_at_unix_ms: now_unix_ms,
        }
    }

    /// Borrow the independent control store for persistence or test evidence.
    #[must_use]
    pub const fn control_store(&self) -> &S {
        &self.store
    }

    /// Whether one exact admitted job has a durable cancellation marker.
    #[must_use]
    pub fn is_cancelled(&self, job_id: Uuid, request_digest: &ResultDigest) -> bool {
        self.jobs.get(&job_id).is_some_and(|entry| {
            entry.request_digest == *request_digest && matches!(entry.state, JobState::Cancelled)
        })
    }

    /// Clone the cooperative cancellation token for one exact live lease.
    pub fn cancellation_token(&self, lease: &JobLease) -> Result<CancellationToken, WorkerError> {
        let entry = self
            .jobs
            .get(&lease.job_id)
            .ok_or_else(|| WorkerError::State("lease references an unknown job".into()))?;
        if entry.lease != *lease || !matches!(entry.state, JobState::Active) {
            return Err(WorkerError::State(
                "lease is mismatched or no longer active".into(),
            ));
        }
        Ok(entry.cancellation.clone())
    }

    fn validate_authority(&self, envelope: &JobEnvelope) -> Result<(), WorkerError> {
        if envelope.audience != self.identity.worker_id
            || envelope.namespace != self.identity.namespace
            || !self
                .identity
                .coordinator_audiences
                .contains(&envelope.coordinator_audience)
        {
            return Err(WorkerError::Authentication(
                "job audience, namespace, or coordinator audience is not accepted".into(),
            ));
        }
        let capability = self
            .capabilities
            .get(&envelope.request.capability)
            .ok_or_else(|| WorkerError::Capability("unknown capability".into()))?;
        if capability.revision != envelope.request.capability_revision {
            return Err(WorkerError::Capability(
                "capability revision is incompatible".into(),
            ));
        }
        Ok(())
    }

    fn validate_time(&self, envelope: &JobEnvelope, now: i64) -> Result<(), WorkerError> {
        if envelope.submitted_at_unix_ms > now.saturating_add(self.limits.clock_skew_ms) {
            return Err(WorkerError::Deadline(
                "submission time is beyond accepted clock skew".into(),
            ));
        }
        if envelope.deadline_unix_ms <= now {
            return Err(WorkerError::Deadline("job deadline has expired".into()));
        }
        Ok(())
    }

    fn retry(
        &self,
        envelope: &JobEnvelope,
        digest: &ResultDigest,
        envelope_digest: &ResultDigest,
        now: i64,
    ) -> Result<Option<Admission>, WorkerError> {
        if let Some(existing_id) = self.nonces.get(&envelope.nonce)
            && *existing_id != envelope.job_id
        {
            return Err(WorkerError::Replay(
                "signed nonce was already used by another job".into(),
            ));
        }
        if let Some(entry) = self.jobs.get(&envelope.job_id) {
            if entry.request_digest != *digest
                || entry.envelope_digest != *envelope_digest
                || entry.nonce != envelope.nonce
                || entry.principal != envelope.principal
            {
                return Err(WorkerError::Replay(
                    "job identity conflicts with accepted canonical request".into(),
                ));
            }
            return match &entry.state {
                JobState::Completed(result) => Ok(Some(Admission::Cached(result.clone()))),
                JobState::Cancelled => Err(WorkerError::State(
                    "cancelled job cannot be resubmitted".into(),
                )),
                JobState::Active if now <= entry.lease.expires_at_unix_ms => {
                    Ok(Some(Admission::ExistingLease(entry.lease.clone())))
                }
                JobState::Active => Err(WorkerError::State(
                    "expired ambiguous lease requires explicit cancellation or recovery".into(),
                )),
            };
        }
        if let Some(existing_id) = self.digests.get(digest) {
            let entry = self
                .jobs
                .get(existing_id)
                .expect("digest index refers to a replayed job");
            return match &entry.state {
                JobState::Completed(result) => Ok(Some(Admission::Cached(result.clone()))),
                _ => Err(WorkerError::Replay(
                    "equivalent canonical request is already active or cancelled".into(),
                )),
            };
        }
        Ok(None)
    }

    fn validate_request_quotas(&self, envelope: &JobEnvelope) -> Result<(), WorkerError> {
        let requested_bytes = usize::try_from(envelope.request.max_result_bytes)
            .map_err(|_| WorkerError::Quota("result byte request does not fit this host".into()))?;
        if requested_bytes > self.limits.result_bytes
            || envelope.request.max_stream_chunks > self.limits.stream_chunks
        {
            return Err(WorkerError::Quota(
                "request exceeds result byte or stream chunk ceiling".into(),
            ));
        }
        let active = self
            .jobs
            .values()
            .filter(|entry| matches!(entry.state, JobState::Active));
        let total = active.clone().count();
        let principal = active
            .clone()
            .filter(|entry| entry.principal == envelope.principal)
            .count();
        let capability = active
            .filter(|entry| entry.capability == envelope.request.capability)
            .count();
        let capability_limit = self.capabilities[&envelope.request.capability].max_concurrency;
        if total >= self.limits.active_jobs
            || principal >= self.limits.active_jobs_per_principal
            || capability >= usize::try_from(capability_limit).unwrap_or(usize::MAX)
        {
            return Err(WorkerError::Quota(
                "active worker, principal, or capability concurrency is exhausted".into(),
            ));
        }
        Ok(())
    }

    fn ensure_event_capacity(&self) -> Result<(), WorkerError> {
        if self.event_count >= self.limits.control_events {
            return Err(WorkerError::Quota(
                "control event retention ceiling is exhausted".into(),
            ));
        }
        Ok(())
    }

    fn ensure_admission_capacity(&self) -> Result<(), WorkerError> {
        let active = self
            .jobs
            .values()
            .filter(|entry| matches!(entry.state, JobState::Active))
            .count();
        let required = self
            .event_count
            .checked_add(active)
            .and_then(|count| count.checked_add(2))
            .unwrap_or(usize::MAX);
        if required > self.limits.control_events {
            return Err(WorkerError::Quota(
                "control history cannot reserve admission and terminal events".into(),
            ));
        }
        Ok(())
    }

    fn validate_replayed_capacity(&self) -> Result<(), WorkerError> {
        let active: Vec<_> = self
            .jobs
            .values()
            .filter(|entry| matches!(entry.state, JobState::Active))
            .collect();
        let terminal_capacity = self.event_count.saturating_add(active.len());
        if active.len() > self.limits.active_jobs || terminal_capacity > self.limits.control_events
        {
            return Err(WorkerError::Quota(
                "replayed active jobs exceed concurrency or terminal-event capacity".into(),
            ));
        }
        let mut principals = BTreeMap::<&str, usize>::new();
        let mut capabilities = BTreeMap::<&str, usize>::new();
        for entry in active {
            *principals.entry(&entry.principal).or_default() += 1;
            *capabilities.entry(&entry.capability).or_default() += 1;
        }
        if principals
            .values()
            .any(|count| *count > self.limits.active_jobs_per_principal)
            || capabilities.iter().any(|(capability, count)| {
                *count
                    > usize::try_from(self.capabilities[*capability].max_concurrency)
                        .unwrap_or(usize::MAX)
            })
        {
            return Err(WorkerError::Quota(
                "replayed active jobs exceed principal or capability concurrency".into(),
            ));
        }
        Ok(())
    }

    fn replay(&mut self, event: ControlEvent) -> Result<(), WorkerError> {
        match event {
            admitted @ ControlEvent::Admitted { .. } => self.apply_admission(admitted),
            ControlEvent::Completed { result } => self.apply_completion(&result),
            ControlEvent::Cancelled {
                job_id,
                request_digest,
                ..
            } => self.apply_cancellation(job_id, &request_digest),
        }
    }

    fn apply_admission(&mut self, event: ControlEvent) -> Result<(), WorkerError> {
        let ControlEvent::Admitted {
            job_id,
            nonce,
            principal,
            capability,
            request_digest,
            envelope_digest,
            lease,
            deadline_unix_ms,
            max_result_bytes,
            max_stream_chunks,
        } = event
        else {
            return Err(WorkerError::ControlStore(
                "internal admission event mismatch".into(),
            ));
        };
        let max_result_bytes = usize::try_from(max_result_bytes).map_err(|_| {
            WorkerError::ControlStore("admission result byte ceiling does not fit this host".into())
        })?;
        if crate::types::validate_id("admission principal", &principal).is_err()
            || crate::types::validate_id("admission capability", &capability).is_err()
            || lease.job_id != job_id
            || lease.job_id.is_nil()
            || lease.lease_id.is_nil()
            || lease.request_digest != request_digest
            || lease.worker_id != self.identity.worker_id
            || lease.expires_at_unix_ms > deadline_unix_ms
            || self.jobs.contains_key(&job_id)
            || self.nonces.contains_key(&nonce)
            || self.digests.contains_key(&request_digest)
            || !self.capabilities.contains_key(&capability)
            || max_result_bytes == 0
            || max_result_bytes > self.limits.result_bytes
            || max_stream_chunks == 0
            || max_stream_chunks > self.limits.stream_chunks
        {
            return Err(WorkerError::ControlStore(
                "admission history is conflicting or malformed".into(),
            ));
        }
        self.nonces.insert(nonce, job_id);
        self.digests.insert(request_digest.clone(), job_id);
        self.jobs.insert(
            job_id,
            JobEntry {
                nonce,
                principal,
                capability,
                request_digest,
                envelope_digest,
                lease,
                deadline_unix_ms,
                max_result_bytes,
                max_stream_chunks,
                cancellation: CancellationToken::new(),
                state: JobState::Active,
            },
        );
        Ok(())
    }

    fn apply_completion(&mut self, result: &JobResult) -> Result<(), WorkerError> {
        let entry = self.jobs.get(&result.job_id).ok_or_else(|| {
            WorkerError::ControlStore("completion history has no admission".into())
        })?;
        if !matches!(entry.state, JobState::Active)
            || entry.request_digest != result.request_digest
            || result.status == JobStatus::Cancelled
            || result.completed_at_unix_ms > entry.lease.expires_at_unix_ms
            || result.completed_at_unix_ms > entry.deadline_unix_ms
        {
            return Err(WorkerError::ControlStore(
                "completion history conflicts with admission".into(),
            ));
        }
        validate_chunks(
            &result.chunks,
            entry.max_stream_chunks.min(self.limits.stream_chunks),
            entry.max_result_bytes.min(self.limits.result_bytes),
        )
        .map_err(|error| WorkerError::ControlStore(error.to_string()))?;
        if !result.verify_digest() {
            return Err(WorkerError::ControlStore(
                "completion result digest does not verify".into(),
            ));
        }
        let entry = self
            .jobs
            .get_mut(&result.job_id)
            .expect("entry was validated above");
        entry.state = JobState::Completed(result.clone());
        Ok(())
    }

    fn apply_cancellation(
        &mut self,
        job_id: Uuid,
        request_digest: &ResultDigest,
    ) -> Result<(), WorkerError> {
        let entry = self.jobs.get_mut(&job_id).ok_or_else(|| {
            WorkerError::ControlStore("cancellation history has no admission".into())
        })?;
        if entry.request_digest != *request_digest {
            return Err(WorkerError::ControlStore(
                "cancellation history digest differs".into(),
            ));
        }
        match entry.state {
            JobState::Active => {
                entry.cancellation.cancel();
                entry.state = JobState::Cancelled;
            }
            JobState::Cancelled => entry.cancellation.cancel(),
            JobState::Completed(_) => {
                return Err(WorkerError::ControlStore(
                    "cancellation history follows completion".into(),
                ));
            }
        }
        Ok(())
    }
}

fn validate_chunks(
    chunks: &[ResultChunk],
    maximum_chunks: u32,
    maximum_bytes: usize,
) -> Result<(), WorkerError> {
    if chunks.len() > usize::try_from(maximum_chunks).unwrap_or(usize::MAX) {
        return Err(WorkerError::Quota(
            "result stream exceeds chunk ceiling".into(),
        ));
    }
    let mut bytes = 0usize;
    for (index, chunk) in chunks.iter().enumerate() {
        if usize::try_from(chunk.sequence).ok() != Some(index) {
            return Err(WorkerError::Invalid(
                "result chunks must have contiguous zero-based sequence numbers".into(),
            ));
        }
        bytes = bytes
            .checked_add(chunk.bytes.len())
            .ok_or_else(|| WorkerError::Quota("result byte count overflowed this host".into()))?;
        if bytes > maximum_bytes {
            return Err(WorkerError::Quota(
                "result stream exceeds byte ceiling".into(),
            ));
        }
    }
    Ok(())
}
