//! Versioned encrypted relay metadata and action-bound node requests.
use crate::{
    Error, NodeCertificate, NodeChallenge, Result, SealedPayload,
    authority::{hash, secret},
    enrollment::json,
};
use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
/// Routing and authenticated-encryption context, derived from a verified account.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JobBinding {
    /// Versioned signature/encryption domain.
    pub protocol: String,
    /// Exact control-plane origin.
    pub audience: String,
    /// Client-generated UUID, never reused for another request.
    pub job_id: String,
    /// Owning account, checked against browser session.
    pub account_id: String,
    /// Owning tenant, checked before admission or lookup.
    pub tenant_id: String,
    /// Exact enrolled user node.
    pub node_id: String,
    /// Exact enrolled capability: model.infer, or tool.execute with durable delivery.
    pub capability: String,
    /// Exclusive Unix deadline; maximum five minutes ahead.
    pub expires_at: i64,
    /// Client submission timestamp, bound for deterministic durable worker envelopes.
    pub submitted_at: i64,
    /// Maximum serialized decrypted result bytes.
    pub max_result_bytes: usize,
    /// Client-held response key; control plane has no private half.
    pub reply_key: [u8; 32],
}
impl JobBinding {
    /// Bind encryption to all metadata plus the request/result direction.
    pub fn aad(&self, direction: &str) -> Result<Vec<u8>> {
        if !matches!(direction, "request" | "result") {
            return Err(Error::Invalid);
        }
        serde_json::to_vec(&(self, direction)).map_err(|_| Error::Invalid)
    }
}
/// Ciphertext-only browser submission.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Submission {
    /// Immutable account/node/job context.
    pub binding: JobBinding,
    /// Node-encrypted application request.
    pub payload: SealedPayload,
}
/// Durable routing state, without application payloads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobReceipt {
    /// Original immutable routing context.
    pub binding: JobBinding,
    /// queued, offered, dispatched, `cancel_requested`, completed, cancelled, expired or unresolved.
    pub state: String,
    /// Hash of ciphertext submission, never a plaintext prompt digest.
    pub submission_digest: String,
    /// Whether encrypted output is still held in transient relay memory.
    pub result_available: bool,
}
/// Authority-signed dispatch, persisted before network delivery.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dispatch {
    /// Encrypted job and immutable binding.
    pub submission: Submission,
    /// Authority signature over the exact dispatch ciphertext and context.
    pub signature: Vec<u8>,
}
impl Dispatch {
    /// Verify exact dispatch against the startup-pinned authority key.
    pub fn verify(&self, key: &[u8; 32]) -> Result<()> {
        VerifyingKey::from_bytes(key)
            .map_err(|_| Error::Denied)?
            .verify_strict(
                &self.bytes()?,
                &Signature::from_slice(&self.signature).map_err(|_| Error::Denied)?,
            )
            .map_err(|_| Error::Denied)
    }
    pub(crate) fn bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&("abi.relay-dispatch.v1", &self.submission)).map_err(|_| Error::Invalid)
    }
}
/// Action included in a signed node message.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
#[allow(clippy::large_enum_variant)] // Preserve the established ergonomic action constructors.
pub enum NodeAction {
    /// Outbound polling for at most one fresh job.
    Poll,
    /// Replay exact durable transport artifacts without execution authority.
    Recovery {
        /// Versioned replay-only request bound by this fresh node proof.
        request: crate::RecoveryRequest,
    },
    /// One ordered, node-signed client-encrypted incremental result.
    Stream {
        /// Original job UUID.
        job_id: String,
        /// Ordered client-encrypted delta.
        frame: crate::StreamFrame,
    },
    /// Cooperative provider has stopped after an owning account requested cancellation.
    Stopped {
        /// Original job UUID after the provider has returned.
        job_id: String,
    },
    /// Report one bounded encrypted terminal result.
    Complete {
        /// Job UUID.
        job_id: String,
        /// Client-encrypted output.
        payload: SealedPayload,
    },
    /// Record execution failure without provider diagnostics or content.
    Fail {
        /// Job UUID.
        job_id: String,
    },
    /// Check cancellation/revocation immediately before local execution.
    Inspect {
        /// Job UUID.
        job_id: String,
    },
}
/// Short-lived node proof binds the complete operation, not merely identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeMessage {
    /// Single-use expiring node proof.
    pub proof: NodeChallenge,
    /// Exact signed operation.
    pub action: NodeAction,
    /// Ed25519 node signature over proof including action commitment.
    pub signature: Vec<u8>,
}
impl NodeMessage {
    /// Sign an operation using the locally held enrollment key.
    pub fn sign(
        cert: &NodeCertificate,
        key: &SigningKey,
        action: NodeAction,
        now: i64,
    ) -> Result<Self> {
        let proof = NodeChallenge {
            protocol: "abi.node-proof.v1".into(),
            audience: cert.audience.clone(),
            node_id: cert.node_id.clone(),
            nonce: format!("{}.{}", secret(), hash(&json(&action)?)),
            expires_at: now + 60,
        };
        let signature = key.sign(&proof.signing_bytes()?).to_bytes().to_vec();
        Ok(Self {
            proof,
            action,
            signature,
        })
    }
    pub(crate) fn validate_action(&self) -> Result<()> {
        if self.proof.nonce.rsplit_once('.').map(|(_, h)| h)
            != Some(hash(&json(&self.action)?).as_str())
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
}
/// Bounded node response. Ciphertext is never stored in authority tables.
#[derive(Serialize, Deserialize)]
pub struct NodeReply {
    /// Signed immutable policy for a durable dispatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<crate::SignedDeliveryPolicy>,
    /// Typed replay-only response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<crate::RecoveryReply>,
    /// Fresh dispatch, at most once, or no queued job.
    pub dispatch: Option<Dispatch>,
    /// Status/result acknowledgement for a known job.
    pub receipt: Option<JobReceipt>,
    /// Short authority grant from a fresh pre-dispatch authority check.
    pub permit: Option<DispatchPermit>,
}
/// Owning browser's status and optional encrypted result.
#[derive(Serialize, Deserialize)]
pub struct JobView {
    /// Durable authority metadata.
    pub receipt: JobReceipt,
    /// Transient client-encrypted output, if available.
    pub result: Option<SealedPayload>,
    /// Ordered encrypted frames currently buffered in this relay epoch.
    #[serde(default)]
    pub frames: Vec<crate::StreamFrame>,
}
/// Startup relay quotas; defaults enforce 2 active and 10 queued per account.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RelayLimits {
    /// Per-account active dispatches.
    pub active_jobs: u32,
    /// Per-account queued requests.
    pub queued_jobs: u32,
    /// Globally retained job metadata entries; full means explicit refusal.
    pub job_records: usize,
    /// Global transient ciphertext memory ceiling.
    pub memory_bytes: usize,
}
impl Default for RelayLimits {
    fn default() -> Self {
        Self {
            active_jobs: 2,
            queued_jobs: 10,
            job_records: 10000,
            memory_bytes: 16 * 1024 * 1024,
        }
    }
}

/// A fresh authority check crosses the durable dispatch boundary before execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchPermit {
    /// Exact immutable job scope.
    pub binding: JobBinding,
    /// Commitment to the complete encrypted submission.
    pub submission_digest: String,
    /// Trusted authority issuance time.
    pub issued_at: i64,
    /// Exclusive expiry, at most five seconds after issuance.
    pub expires_at: i64,
    /// Authority signature over versioned fields.
    pub signature: Vec<u8>,
}
impl DispatchPermit {
    pub(crate) fn bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&(
            "abi.relay-permit.v1",
            &self.binding,
            &self.submission_digest,
            self.issued_at,
            self.expires_at,
        ))
        .map_err(|_| Error::Invalid)
    }
    /// Verify a bounded, exact dispatch authorization.
    pub fn verify(&self, key: &[u8; 32], now: i64) -> Result<()> {
        if now < self.issued_at
            || now >= self.expires_at
            || self.expires_at > self.issued_at + 5
            || self.expires_at > self.binding.expires_at
        {
            return Err(Error::Denied);
        }
        VerifyingKey::from_bytes(key)
            .map_err(|_| Error::Denied)?
            .verify_strict(
                &self.bytes()?,
                &Signature::from_slice(&self.signature).map_err(|_| Error::Denied)?,
            )
            .map_err(|_| Error::Denied)
    }
}
