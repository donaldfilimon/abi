//! Versioned metadata contracts. No conversation or provider-secret fields.
use serde::{Deserialize, Serialize};
/// Authority result.
pub type Result<T> = std::result::Result<T, Error>;
/// Fail-closed authority errors. HTTP responses never expose storage diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Authentication, ownership, expiry or state preconditions failed.
    #[error("authority denied request")]
    Denied,
    /// A transient outbound network or service failure; inference is never retried.
    #[error("relay temporarily unavailable")]
    Unavailable,
    /// Bounded metadata admission is full.
    #[error("metadata quota exceeded")]
    Full,
    /// Invalid configuration or request.
    #[error("invalid authority configuration or request")]
    Invalid,
    /// Durable storage unavailable.
    #[error("authority storage unavailable")]
    Storage(#[from] rusqlite::Error),
    /// Private filesystem unavailable.
    #[error("private authority filesystem unavailable")]
    Io(#[from] std::io::Error),
}
pub(crate) struct VerifiedIdentity {
    pub issuer: String,
    pub subject: String,
}
/// Opaque account and tenant identities; quotas are service admission defaults.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Account {
    /// Account identifier.
    pub account_id: String,
    /// Private-workload tenant identifier.
    pub tenant_id: String,
    /// Maximum simultaneously active jobs.
    pub active_job_limit: u32,
    /// Maximum queued jobs.
    pub queued_job_limit: u32,
}
/// Browser session credentials; bearer material is never stored verbatim.
pub(crate) struct Session {
    pub token: String,
    pub csrf: String,
}
/// Request made by an unpaired node. The private key stays on the node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairRequest {
    /// Ed25519 public key bytes.
    pub public_key: [u8; 32],
    /// Optional X25519 relay key. Legacy nodes without it cannot accept encrypted jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encryption_public_key: Option<[u8; 32]>,
    /// Requested private workload capabilities, sorted and unique.
    pub capabilities: Vec<String>,
}
/// One-time browser code and private node polling secret.
#[derive(Debug, Serialize, Deserialize)]
pub struct PairTicket {
    /// Pending enrollment identifier.
    pub enrollment_id: String,
    /// High-entropy code to enter in the browser.
    pub user_code: String,
    /// Node-only bearer secret.
    pub poll_secret: String,
    /// Exclusive Unix expiry.
    pub expires_at: i64,
}
/// Exact local consent preview. Sign these bytes only after local approval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairClaim {
    /// Protocol and signature domain.
    pub protocol: String,
    /// Pending enrollment identifier.
    pub enrollment_id: String,
    /// Bound browser account.
    pub account_id: String,
    /// Bound private workload namespace.
    pub tenant_id: String,
    /// Node public key.
    pub public_key: [u8; 32],
    /// Optional X25519 relay key. Legacy nodes without it cannot accept encrypted jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encryption_public_key: Option<[u8; 32]>,
    /// Approved capabilities.
    pub capabilities: Vec<String>,
    /// Exclusive Unix expiry.
    pub expires_at: i64,
}
impl PairClaim {
    /// Canonical, version-bound local approval bytes.
    pub fn signing_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|_| Error::Invalid)
    }
}
/// Issued application credential. This is not an X.509 or transport TLS certificate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NodeCertificate {
    /// Signature domain and version.
    pub protocol: String,
    /// Control-plane audience, preventing use at another service.
    pub audience: String,
    /// Stable enrolled node identifier.
    pub node_id: String,
    /// Tenant and account binding and immutable capability consent.
    pub claim: PairClaim,
    /// Exclusive credential expiry.
    pub expires_at: i64,
    /// Ed25519 authority signature over all preceding fields.
    pub signature: Vec<u8>,
}
impl NodeCertificate {
    /// Canonical signature input with the signature cleared.
    pub fn signing_bytes(&self) -> Result<Vec<u8>> {
        let mut unsigned = self.clone();
        unsigned.signature.clear();
        serde_json::to_vec(&unsigned).map_err(|_| Error::Invalid)
    }
    /// Verify authority signature, audience, protocol and validity.
    pub fn verify(&self, key: &[u8; 32], audience: &str, now: i64) -> Result<()> {
        use ed25519_dalek::{Signature, VerifyingKey};
        if self.protocol != "abi.node-certificate.v1"
            || self.audience != audience
            || self.expires_at <= now
        {
            return Err(Error::Denied);
        }
        VerifyingKey::from_bytes(key)
            .map_err(|_| Error::Denied)?
            .verify_strict(
                &self.signing_bytes()?,
                &Signature::from_slice(&self.signature).map_err(|_| Error::Denied)?,
            )
            .map_err(|_| Error::Denied)
    }
}
/// Node metadata visible only to its owning browser account.
#[derive(Debug, Serialize, Deserialize)]
pub struct Node {
    /// Node identifier.
    pub node_id: String,
    /// Exact granted capabilities.
    pub capabilities: Vec<String>,
    /// Exclusive credential expiry.
    pub expires_at: i64,
    /// Revocation state.
    pub revoked: bool,
    /// Locally approved relay encryption key, if enrolled.
    pub encryption_public_key: Option<[u8; 32]>,
}
/// Single-use proof of possession challenge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeChallenge {
    /// Protocol signature domain.
    pub protocol: String,
    /// Control-plane audience.
    pub audience: String,
    /// Node identifier.
    pub node_id: String,
    /// Unpredictable single-use challenge.
    pub nonce: String,
    /// Exclusive expiry.
    pub expires_at: i64,
}
impl NodeChallenge {
    /// Bytes signed by the enrolled node.
    pub fn signing_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|_| Error::Invalid)
    }
}
