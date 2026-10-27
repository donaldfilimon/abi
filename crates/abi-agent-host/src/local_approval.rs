//! Trusted local approval ownership. Raw previews stay on the user node.
use abi_capability::lifecycle::{
    AuthorizationBinding, DurableLifecycle, ExactApproval, LifecycleError,
};
use abi_capability::{Digest, Request};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// Exact human-visible local preview. Never send this value to a central relay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LocalApprovalPreview {
    /// Current local preview schema.
    pub version: u8,
    /// Startup-pinned tenant.
    pub tenant_id: String,
    /// Startup-pinned account.
    pub account_id: String,
    /// Startup-pinned user node.
    pub node_id: String,
    /// Startup-authenticated local user.
    pub user_id: String,
    /// Exact canonical capability request.
    pub request: Request,
    /// Complete current authorization commitment.
    pub binding: AuthorizationBinding,
    /// Selected exact grant identifier.
    pub grant_id: String,
    /// Registered tool identifier.
    pub tool_id: String,
    /// Human-visible operation.
    pub operation: String,
    /// Human-visible exact resource selector.
    pub resource: String,
    /// Canonical parameters; local persistence only.
    pub parameters: serde_json::Value,
    /// Expected visible effects.
    pub expected_effects: Vec<String>,
    /// Trusted creation time.
    pub created_at_ms: u64,
    /// Exclusive expiry.
    pub expires_at_ms: u64,
}
impl LocalApprovalPreview {
    /// Commit every visible field, including owner and expiry.
    #[must_use]
    pub fn digest(&self) -> Digest {
        let mut hasher = Sha256::new();
        hasher.update(b"abi-local-approval-preview-v1\0");
        hasher.update(serde_json::to_vec(self).expect("closed preview serializes"));
        Digest::from_bytes(hasher.finalize().into())
    }
}
/// Durable decision state; pending never grants authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalApprovalState {
    /// Awaiting explicit local user decision.
    Pending,
    /// Exact local user approval.
    Approved,
    /// Explicit refusal.
    Denied,
    /// Expired before use.
    Expired,
    /// Explicit cancellation.
    Cancelled,
    /// Already used for a durable intent.
    Consumed,
}
/// Preview commitment and its durable state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalApprovalStatus {
    /// Opaque local decision identity, independent of relay job identity.
    pub call_id: String,
    /// Commitment of the complete stored preview.
    pub preview_digest: Digest,
    /// Current durable state.
    pub state: LocalApprovalState,
}
/// Startup-authenticated local authority, never a network-supplied identity.
pub trait LocalApprovalStore: DurableLifecycle {
    /// Read the complete private preview for the startup-authenticated local owner.
    fn read_local_approval(
        &self,
        call_id: &str,
    ) -> Result<Option<LocalApprovalPreview>, LifecycleError>;
    /// Persist an immutable complete preview; changed calls fail closed.
    fn request_local_approval(
        &self,
        preview: &LocalApprovalPreview,
    ) -> Result<LocalApprovalStatus, LifecycleError>;
    /// Decide as the authority's pinned local principal.
    fn decide_local_approval(
        &self,
        call_id: &str,
        preview_digest: Digest,
        decision_id: &str,
        approved: bool,
        now_ms: u64,
    ) -> Result<LocalApprovalStatus, LifecycleError>;
    /// Reconstruct an exact approval exclusively from the durable local decision.
    fn exact_local_approval(
        &self,
        call_id: &str,
        preview_digest: Digest,
        now_ms: u64,
    ) -> Result<Option<ExactApproval>, LifecycleError>;
}
