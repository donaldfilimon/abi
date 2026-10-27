//! Additive durable delivery protocol. Recovery grants never authorize execution.
use crate::{
    Error, JobBinding, RelayKey, Result, SealedPayload, StreamFrame, Submission, authority::hash,
    enrollment::json, seal,
};
use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(hash(&json(value)?))
}
fn verify(key: &[u8; 32], bytes: &[u8], signature: &[u8]) -> Result<()> {
    VerifyingKey::from_bytes(key)
        .map_err(|_| Error::Denied)?
        .verify_strict(
            bytes,
            &Signature::from_slice(signature).map_err(|_| Error::Denied)?,
        )
        .map_err(|_| Error::Denied)
}
/// Immutable transport retention, independent of execution authorization.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeliveryPolicy {
    /// Exclusive immutable transport retention deadline.
    pub until_unix_s: i64,
    /// Commitment to the original version-one job binding.
    pub binding_digest: String,
    /// Commitment to the original encrypted submission.
    pub submission_digest: String,
}
impl DeliveryPolicy {
    /// Construct commitments for an opted-in encrypted submission.
    pub fn for_submission(s: &Submission, until_unix_s: i64) -> Result<Self> {
        Ok(Self {
            until_unix_s,
            binding_digest: digest(&s.binding)?,
            submission_digest: digest(s)?,
        })
    }
    /// Validate exact submission commitments and retention bounds.
    pub fn validate(&self, s: &Submission) -> Result<()> {
        if *self != Self::for_submission(s, self.until_unix_s)?
            || self.until_unix_s < s.binding.submitted_at.saturating_add(60)
            || self.until_unix_s > s.binding.submitted_at.saturating_add(7 * 86400)
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
}
/// Authority attestation of immutable policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedDeliveryPolicy {
    /// Immutable authority-attested delivery policy.
    pub policy: DeliveryPolicy,
    /// Ed25519 signature over all versioned signed fields.
    pub signature: Vec<u8>,
}
impl SignedDeliveryPolicy {
    pub(crate) fn sign(policy: DeliveryPolicy, key: &SigningKey) -> Result<Self> {
        let mut s = Self {
            policy,
            signature: vec![],
        };
        s.signature = key.sign(&s.bytes()?).to_bytes().to_vec();
        Ok(s)
    }
    fn bytes(&self) -> Result<Vec<u8>> {
        Ok(json(&("abi.relay-delivery-policy.v1", &self.policy))?.into_bytes())
    }
    /// Verify signature, immutable context and applicable protocol bounds.
    pub fn verify(
        &self,
        key: &[u8; 32],
        binding: &JobBinding,
        submission_digest: &str,
    ) -> Result<()> {
        if self.policy.binding_digest != digest(binding)?
            || self.policy.submission_digest != submission_digest
            || self.policy.until_unix_s < binding.submitted_at.saturating_add(60)
            || self.policy.until_unix_s > binding.submitted_at.saturating_add(7 * 86400)
        {
            return Err(Error::Denied);
        }
        verify(key, &self.bytes()?, &self.signature)
    }
}
/// Explicit durable opt-in; v1 submission bytes are unchanged.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableSubmission {
    /// Original byte-compatible version-one submission.
    pub submission: Submission,
    /// Requested immutable transport retention.
    pub delivery: DeliveryPolicy,
}
/// Distinct packet identity spaces.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PacketKind {
    /// Ordered independently encrypted provider delta.
    Delta,
    /// Sealed terminal result packet.
    Terminal,
}
/// Commitment to exact encoded ciphertext packet.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PacketId {
    /// Exact original job UUID.
    pub job_id: String,
    /// Packet identity namespace.
    pub kind: PacketKind,
    /// Zero-based delta sequence, or zero for terminal packets.
    pub sequence: u32,
    /// Commitment to exact serialized packet bytes.
    pub digest: String,
}
/// Exact signed terminal ciphertext, sealed once at the node.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalPacket {
    /// Client-encrypted application result.
    pub payload: SealedPayload,
    /// Ed25519 signature over all versioned signed fields.
    pub signature: Vec<u8>,
}
impl TerminalPacket {
    fn bytes(&self, b: &JobBinding) -> Result<Vec<u8>> {
        Ok(json(&("abi.relay-terminal.v1", b, &self.payload))?.into_bytes())
    }
    /// Encrypt once and sign the exact terminal packet.
    pub fn seal(b: &JobBinding, bytes: &[u8], key: &SigningKey) -> Result<Self> {
        if bytes.len() > b.max_result_bytes {
            return Err(Error::Invalid);
        }
        let mut p = Self {
            payload: seal(&b.reply_key, &b.aad("result")?, bytes)?,
            signature: vec![],
        };
        p.signature = key.sign(&p.bytes(b)?).to_bytes().to_vec();
        Ok(p)
    }
    /// Verify signature, immutable context and applicable protocol bounds.
    pub fn verify(&self, b: &JobBinding, key: &[u8; 32]) -> Result<()> {
        self.payload.validate()?;
        if self.payload.ciphertext.len() > b.max_result_bytes + 16 {
            return Err(Error::Invalid);
        }
        verify(key, &self.bytes(b)?, &self.signature)
    }
    /// Verify the node signature before decrypting the terminal result.
    pub fn open(&self, b: &JobBinding, node_key: &[u8; 32], key: &RelayKey) -> Result<Vec<u8>> {
        self.verify(b, node_key)?;
        key.open(&b.aad("result")?, &self.payload)
    }
}
/// Exact transport artifact retained by the user node.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutboxPacket {
    /// Ordered independently encrypted provider delta.
    Delta {
        /// Exact signed encrypted delta.
        frame: StreamFrame,
    },
    /// Sealed terminal result packet.
    Terminal {
        /// Exact signed encrypted terminal result.
        packet: TerminalPacket,
    },
}
impl OutboxPacket {
    /// Derive a stable identity for exact saved packet bytes.
    pub fn id(&self, b: &JobBinding) -> Result<PacketId> {
        let (kind, sequence, digest) = match self {
            Self::Delta { frame } => (PacketKind::Delta, frame.sequence, digest(frame)?),
            Self::Terminal { packet } => (PacketKind::Terminal, 0, digest(packet)?),
        };
        Ok(PacketId {
            job_id: b.job_id.clone(),
            kind,
            sequence,
            digest,
        })
    }
    /// Verify signature, immutable context and applicable protocol bounds.
    pub fn verify(&self, b: &JobBinding, key: &[u8; 32]) -> Result<()> {
        match self {
            Self::Delta { frame } => frame.verify(b, key),
            Self::Terminal { packet } => packet.verify(b, key),
        }
    }
    /// Return actual JSON transport encoding size.
    pub fn encoded_bytes(&self) -> Result<usize> {
        Ok(serde_json::to_vec(self).map_err(|_| Error::Invalid)?.len())
    }
}
/// Recorded execution facts; unresolved never asserts successful completion.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecordedOutcome {
    /// Durably recorded terminal completion within execution time.
    Completed,
    /// Durably observed stopped or cancelled execution.
    Cancelled,
    /// Execution outcome remains uncertain; no provider resume.
    Unresolved,
    /// Prepared job without an execution admission.
    Abandoned,
}
/// Signed ordered prefix and immutable terminal evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalManifest {
    /// Exact original account, tenant, node and job scope.
    pub binding: JobBinding,
    /// Commitment to the original encrypted submission.
    pub submission_digest: String,
    /// Ordered commitments to the saved encrypted delta prefix.
    pub frame_digests: Vec<String>,
    /// Exact signed terminal ciphertext commitment, if complete.
    pub terminal_digest: Option<String>,
    /// Durably recorded execution outcome.
    pub outcome: RecordedOutcome,
    /// Original durable completion timestamp, if complete.
    pub completed_at: Option<i64>,
    /// Ed25519 signature over all versioned signed fields.
    pub signature: Vec<u8>,
}
impl TerminalManifest {
    fn bytes(&self) -> Result<Vec<u8>> {
        Ok(json(&(
            "abi.relay-manifest.v1",
            &self.binding,
            &self.submission_digest,
            &self.frame_digests,
            &self.terminal_digest,
            &self.outcome,
            self.completed_at,
        ))?
        .into_bytes())
    }
    /// Sign this exact manifest revision with the enrolled node key.
    pub fn sign(mut self, key: &SigningKey) -> Result<Self> {
        self.signature = key.sign(&self.bytes()?).to_bytes().to_vec();
        Ok(self)
    }
    /// Verify signature, immutable context and applicable protocol bounds.
    pub fn verify(&self, key: &[u8; 32]) -> Result<()> {
        let valid_digest = |d: &str| {
            d.len() == 43
                && d.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        };
        if self.frame_digests.iter().any(|d| !valid_digest(d))
            || self
                .terminal_digest
                .as_ref()
                .is_some_and(|d| !valid_digest(d))
            || self.frame_digests.len() > 16
            || (self.outcome == RecordedOutcome::Completed
                && (self.terminal_digest.is_none()
                    || self.completed_at.is_none_or(|t| {
                        t < self.binding.submitted_at || t > self.binding.expires_at
                    })))
            || (self.outcome != RecordedOutcome::Completed && self.terminal_digest.is_some())
        {
            return Err(Error::Denied);
        }
        verify(key, &self.bytes()?, &self.signature)
    }
    /// Commit to the entire signed manifest.
    pub fn digest(&self) -> Result<String> {
        digest(self)
    }
    /// List the ordered packet identities committed by this manifest.
    #[must_use]
    pub fn packets(&self) -> Vec<PacketId> {
        let mut p: Vec<_> = self
            .frame_digests
            .iter()
            .enumerate()
            .map(|(i, d)| PacketId {
                job_id: self.binding.job_id.clone(),
                kind: PacketKind::Delta,
                sequence: u32::try_from(i).unwrap_or(u32::MAX),
                digest: d.clone(),
            })
            .collect();
        if let Some(d) = &self.terminal_digest {
            p.push(PacketId {
                job_id: self.binding.job_id.clone(),
                kind: PacketKind::Terminal,
                sequence: 0,
                digest: d.clone(),
            });
        }
        p
    }
}
/// A short replay-only authorization tied to one authority process epoch.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayGrant {
    /// Exact original account, tenant, node and job scope.
    pub binding: JobBinding,
    /// Immutable authority-attested delivery policy.
    pub policy: DeliveryPolicy,
    /// Commitment to the exact signed manifest revision.
    pub manifest_digest: String,
    /// Current transient relay process identity.
    pub epoch: String,
    /// Exact authorized or returned encrypted packet identities.
    pub packets: Vec<PacketId>,
    /// Sticky owning-account cancellation observation.
    pub cancelled: bool,
    /// Authority issuance timestamp.
    pub issued_at: i64,
    /// Exclusive replay grant expiry, bounded by retention.
    pub expires_at: i64,
    /// Ed25519 signature over all versioned signed fields.
    pub signature: Vec<u8>,
}
impl ReplayGrant {
    pub(crate) fn bytes(&self) -> Result<Vec<u8>> {
        Ok(json(&(
            "abi.relay-replay-grant.v1",
            &self.binding,
            &self.policy,
            &self.manifest_digest,
            &self.epoch,
            &self.packets,
            self.cancelled,
            self.issued_at,
            self.expires_at,
        ))?
        .into_bytes())
    }
    /// Verify signature, immutable context and applicable protocol bounds.
    pub fn verify(&self, key: &[u8; 32], now: i64) -> Result<()> {
        if now < self.issued_at
            || now >= self.expires_at
            || self.expires_at > self.issued_at + 30
            || self.expires_at > self.policy.until_unix_s
        {
            return Err(Error::Denied);
        }
        verify(key, &self.bytes()?, &self.signature)
    }
}
/// Replay-only inspection and exact packet publication.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveryAction {
    /// Reconcile immutable manifest commitments and request a replay grant.
    Inspect {
        /// Exact original job UUID.
        job_id: String,
        /// Signed node statement of the ordered prefix and terminal facts.
        manifest: TerminalManifest,
    },
    /// Restore one exact encrypted packet using a replay-only grant.
    Publish {
        /// Optional short-lived replay-only authorization.
        grant: ReplayGrant,
        /// Exact saved signed ciphertext artifact.
        packet: OutboxPacket,
    },
}
/// Versioned inner request carried by an action-bound node proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryRequest {
    /// Versioned recovery protocol domain.
    pub protocol: String,
    /// Replay-only operation bound into the fresh node proof.
    pub action: RecoveryAction,
}
/// Typed recovery outcome independent of generic transport errors.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStatus {
    /// Current authorization or protocol refuses recovery.
    Refused,
    /// No opted-in durable delivery history exists for this job.
    Unavailable,
    /// Immutable delivery retention has expired.
    Expired,
    /// Transient relay capacity is exhausted; retain local artifacts.
    CapacityFull,
    /// Exact packet commitment and transient packet are acknowledged.
    Recorded,
    /// Committed packets are absent from this relay epoch.
    Missing,
    /// All committed packets are present in this relay epoch.
    Cached,
    /// Metadata conflicts with a previously committed manifest.
    Contradictory,
}
/// Commitment and transient-cache reconciliation for one fresh node request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoveryReply {
    /// Explicit recovery classification.
    pub status: RecoveryStatus,
    /// Current transient relay process identity.
    pub epoch: String,
    /// Packet commitments durably recorded by the authority.
    pub committed: Vec<PacketId>,
    /// Exact packets currently held in transient relay memory.
    pub cached: Vec<PacketId>,
    /// Optional short-lived replay-only authorization.
    pub grant: Option<ReplayGrant>,
    /// Sticky owning-account cancellation observation.
    pub cancelled: bool,
}
/// Owning client encrypted delivery page, bounded by actual encoded size.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliveryPage {
    /// Authoritative original job scope and execution state.
    pub receipt: crate::JobReceipt,
    /// Immutable authority-attested delivery policy.
    pub policy: SignedDeliveryPolicy,
    /// Signed node statement of the ordered prefix and terminal facts.
    pub manifest: Option<TerminalManifest>,
    /// Exact authorized or returned encrypted packet identities.
    pub packets: Vec<OutboxPacket>,
    /// Next zero-based delta cursor.
    pub next: u32,
    /// Current transient relay process identity.
    pub epoch: String,
    /// Sticky owning-account cancellation observation.
    pub cancelled: bool,
}
/// Exact verified import checkpoint supplied by the owning client.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryAck {
    /// Commitment to the exact signed manifest revision.
    pub manifest_digest: String,
    /// Number of verified contiguous deltas imported by the client.
    pub contiguous: u32,
    /// Exact signed terminal ciphertext commitment, if complete.
    pub terminal_digest: Option<String>,
}

/// Durable outbox manifest.
pub type OutboxManifest = TerminalManifest;
/// Delivery response classification.
pub type DeliveryStatus = RecoveryStatus;

#[cfg(test)]
mod tests;
