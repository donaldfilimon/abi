//! User-node executor: decrypt locally, verify permit, reuse durable worker admission.
use crate::{
    Dispatch, DispatchPermit, Error, NodeCertificate, RelayKey, Result, SealedPayload,
    authority::hash, enrollment::json, seal,
};
use abi_agent_runtime::ModelProvider;
use abi_worker::{
    CoordinatorTrustStore, InferenceOutcome, JobControlStore, JobEnvelope, JobRequest,
    WorkerCapability, WorkerIdentity, WorkerLimits, WorkerManager,
};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
/// Plaintext request exists only on submitting clients and user nodes.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceJob {
    /// User prompt, never control-plane metadata.
    pub prompt: String,
    /// Bounded output token budget.
    pub tokens: u64,
    /// Startup-bound immutable model revision.
    pub model: String,
}
/// User-owned runtime with a pinned service, enrolled node and durable control history.
pub struct RelayRuntime<S: JobControlStore> {
    certificate: NodeCertificate,
    key: SigningKey,
    encryption: RelayKey,
    authority_key: [u8; 32],
    revision: String,
    capability: String,
    manager: WorkerManager<S>,
}
impl<S: JobControlStore> RelayRuntime<S> {
    /// Bind node keys, service key, model revision and user-hosted history at startup.
    pub fn new(
        certificate: NodeCertificate,
        key: SigningKey,
        encryption: RelayKey,
        authority_key: [u8; 32],
        revision: &str,
        store: S,
        now: i64,
    ) -> Result<Self> {
        Self::new_bound(
            certificate,
            key,
            encryption,
            authority_key,
            revision,
            "model.infer",
            store,
            now,
        )
    }
    /// Bind a governed-tool-only worker; no inference capability is granted.
    #[allow(clippy::too_many_arguments)]
    pub fn new_tools(
        certificate: NodeCertificate,
        key: SigningKey,
        encryption: RelayKey,
        authority_key: [u8; 32],
        revision: &str,
        store: S,
        now: i64,
    ) -> Result<Self> {
        Self::new_bound(
            certificate,
            key,
            encryption,
            authority_key,
            revision,
            "tool.execute",
            store,
            now,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn new_bound(
        certificate: NodeCertificate,
        key: SigningKey,
        encryption: RelayKey,
        authority_key: [u8; 32],
        revision: &str,
        capability: &str,
        store: S,
        now: i64,
    ) -> Result<Self> {
        certificate.verify(&authority_key, &certificate.audience, now)?;
        if certificate.claim.public_key != key.verifying_key().to_bytes()
            || certificate.claim.encryption_public_key != Some(encryption.public_key())
            || !certificate
                .claim
                .capabilities
                .iter()
                .any(|s| s == capability)
        {
            return Err(Error::Denied);
        }
        let mut trust = CoordinatorTrustStore::new();
        trust
            .insert(
                "local-node",
                "abi-relay-node",
                key.verifying_key().as_bytes(),
            )
            .map_err(|_| Error::Denied)?;
        let identity = WorkerIdentity::new(
            &certificate.node_id,
            &certificate.claim.tenant_id,
            ["abi-relay-node".into()],
        )
        .map_err(|_| Error::Denied)?;
        let worker_capability =
            WorkerCapability::new(capability, revision, 2).map_err(|_| Error::Invalid)?;
        let limits = WorkerLimits {
            input_bytes: 16384,
            result_bytes: 16384,
            stream_chunks: 16,
            ..WorkerLimits::default()
        };
        let manager = WorkerManager::new(identity, [worker_capability], trust, limits, store)
            .map_err(|_| Error::Denied)?;
        Ok(Self {
            certificate,
            key,
            encryption,
            authority_key,
            revision: revision.into(),
            capability: capability.into(),
            manager,
        })
    }
    /// Validate exact signed dispatch and a fresh permit before any local provider call.
    /// Existing admission returns a cached result or unresolved state, never another run.
    pub fn execute(
        &mut self,
        dispatch: &Dispatch,
        permit: &DispatchPermit,
        provider: &dyn ModelProvider,
        mut clock: impl FnMut() -> i64,
    ) -> Result<Option<SealedPayload>> {
        self.execute_streaming(
            dispatch,
            permit,
            provider,
            &mut clock,
            &abi_worker::CancellationToken::new(),
            |_| Ok(()),
        )
    }
    /// Execute with a cooperative stop token and encrypted, ordered incremental output.
    pub fn execute_streaming(
        &mut self,
        dispatch: &Dispatch,
        permit: &DispatchPermit,
        provider: &dyn ModelProvider,
        clock: impl FnMut() -> i64,
        cancel: &abi_worker::CancellationToken,
        mut emit: impl FnMut(crate::StreamFrame) -> Result<()> + Send,
    ) -> Result<Option<SealedPayload>> {
        let Some(result) =
            self.infer_dispatch(dispatch, permit, provider, clock, cancel, |_, frame| {
                emit(frame)
            })?
        else {
            return Ok(None);
        };
        let bytes = serde_json::to_vec(&result).map_err(|_| Error::Denied)?;
        let b = &dispatch.submission.binding;
        if !result.verify_digest() || bytes.len() > b.max_result_bytes {
            return Err(Error::Denied);
        }
        Ok(Some(seal(&b.reply_key, &b.aad("result")?, &bytes)?))
    }
    fn infer_dispatch(
        &mut self,
        dispatch: &Dispatch,
        permit: &DispatchPermit,
        provider: &dyn ModelProvider,
        mut clock: impl FnMut() -> i64,
        cancel: &abi_worker::CancellationToken,
        mut emit: impl FnMut(&abi_worker::ResultChunk, crate::StreamFrame) -> Result<()> + Send,
    ) -> Result<Option<abi_worker::JobResult>> {
        let now = clock();
        dispatch.verify(&self.authority_key)?;
        permit.verify(&self.authority_key, now)?;
        let b = &dispatch.submission.binding;
        self.certificate
            .verify(&self.authority_key, &b.audience, now)?;
        if b != &permit.binding
            || permit.submission_digest != hash(&json(&dispatch.submission)?)
            || b.protocol != "abi.relay-job.v1"
            || b.node_id != self.certificate.node_id
            || b.tenant_id != self.certificate.claim.tenant_id
            || b.account_id != self.certificate.claim.account_id
            || b.capability != "model.infer"
            || now < b.submitted_at
            || now >= b.expires_at
        {
            return Err(Error::Denied);
        }
        let envelope = self.envelope_for_dispatch(dispatch)?;
        if envelope.request.capability_revision != self.revision {
            return Err(Error::Denied);
        }
        // The clock and grant are rechecked immediately before durable admission.
        let admitted = clock();
        permit.verify(&self.authority_key, admitted)?;
        let outcome = self
            .manager
            .infer_observed(
                &envelope,
                provider,
                || clock() * 1000,
                |time| {
                    let seconds = time / 1000;
                    permit
                        .verify(&self.authority_key, seconds)
                        .and_then(|()| {
                            self.certificate
                                .verify(&self.authority_key, &b.audience, seconds)
                        })
                        .map_err(|_| {
                            abi_worker::WorkerError::Authentication(
                                "dispatch permit or node certificate no longer valid".into(),
                            )
                        })
                },
                cancel,
                |chunk| {
                    let frame =
                        crate::StreamFrame::seal(b, chunk.sequence, &chunk.bytes, &self.key)
                            .and_then(|frame| emit(chunk, frame));
                    frame.map_err(|_| abi_worker::WorkerError::State("relay stream refused".into()))
                },
            )
            .map_err(|_| Error::Denied)?;
        let result = match outcome {
            InferenceOutcome::Completed(r) | InferenceOutcome::Cached(r) => r,
            InferenceOutcome::InProgress(_) => return Ok(None),
        };
        Ok(Some(result))
    }
    pub(crate) fn envelope_for_dispatch(&self, dispatch: &Dispatch) -> Result<JobEnvelope> {
        dispatch.verify(&self.authority_key)?;
        let b = &dispatch.submission.binding;
        if b.node_id != self.certificate.node_id
            || b.tenant_id != self.certificate.claim.tenant_id
            || b.account_id != self.certificate.claim.account_id
            || b.audience != self.certificate.audience
            || b.capability != self.capability
            || b.protocol != "abi.relay-job.v1"
        {
            return Err(Error::Denied);
        }
        let plaintext = self
            .encryption
            .open(&b.aad("request")?, &dispatch.submission.payload)?;
        let request = if self.capability == "tool.execute" {
            let input: crate::ToolJob =
                serde_json::from_slice(&plaintext).map_err(|_| Error::Denied)?;
            input.validate()?;
            JobRequest::new(
                "tool.execute",
                &input.revision,
                &input.tool,
                input.parameters,
                u64::try_from(b.max_result_bytes.saturating_sub(4096).min(16_384))
                    .map_err(|_| Error::Invalid)?,
                16,
            )
            .map_err(|_| Error::Denied)?
        } else {
            let input: InferenceJob =
                serde_json::from_slice(&plaintext).map_err(|_| Error::Denied)?;
            if input.prompt.len() > 8192 || !(1..=1024).contains(&input.tokens) {
                return Err(Error::Denied);
            }
            JobRequest::new(
                "model.infer",
                &input.model,
                "infer",
                serde_json::json!({"prompt":input.prompt,"tokens":input.tokens}),
                u64::try_from(b.max_result_bytes.saturating_sub(4096) / 4)
                    .map_err(|_| Error::Invalid)?,
                16,
            )
            .map_err(|_| Error::Denied)?
        };
        let job = uuid::Uuid::parse_str(&b.job_id).map_err(|_| Error::Denied)?;
        let nonce = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, b.job_id.as_bytes());
        JobEnvelope::sign(
            job,
            &b.account_id,
            &b.node_id,
            &b.tenant_id,
            "abi-relay-node",
            b.submitted_at * 1000,
            b.expires_at * 1000,
            nonce,
            request,
            "local-node",
            &self.key,
        )
        .map_err(|_| Error::Denied)
    }
    #[cfg(test)]
    pub(crate) fn admit_recovery_fixture(&mut self, dispatch: &Dispatch, now: i64) -> Result<()> {
        self.manager
            .submit(&self.envelope_for_dispatch(dispatch)?, now * 1000)
            .map(|_| ())
            .map_err(|_| Error::Denied)
    }
    /// Inspect exact durable evidence without admitting or executing a job.
    pub fn recorded_dispatch(&self, dispatch: &Dispatch) -> Result<abi_worker::RecordedExecution> {
        self.manager
            .recorded(&self.envelope_for_dispatch(dispatch)?)
            .map_err(|_| Error::Denied)
    }
    pub(crate) fn recover_dispatch(
        &mut self,
        dispatch: &Dispatch,
        now: i64,
    ) -> Result<abi_worker::RecordedExecution> {
        self.manager
            .recover_interrupted(&self.envelope_for_dispatch(dispatch)?, now * 1000)
            .map_err(|_| Error::Denied)
    }
}

mod recovery;
pub use recovery::DurableRelayRuntime;

mod tools;
pub use tools::{GovernedToolContext, GovernedToolOutcome, ToolJob};
