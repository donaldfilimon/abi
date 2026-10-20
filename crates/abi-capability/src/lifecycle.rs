//! Additive internal durable admission contracts, independent of wire envelopes.
//! Implementations must use the owning ledger transaction for consume + intent + audit.
use crate::recording::digest_json;
use crate::{
    Approval, ApprovalLevel, AuthorizationEvaluation, Decision, Digest, ReasonCode, Request,
};
use serde::{Deserialize, Serialize};

/// Server-enforced approval lifetime ceiling.
pub const MAX_APPROVAL_TTL_MS: u64 = 15 * 60 * 1_000;

/// Complete redacted authority commitment shown in an approval preview.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthorizationBinding {
    /// Typed request commitment, including parameters and call identity.
    pub request_digest: Digest,
    /// Complete principal commitment.
    pub principal_digest: Digest,
    /// Complete exact scope commitment.
    pub scope_digest: Digest,
    /// Compiled source package commitment.
    pub package_digest: Digest,
    /// Complete selected grant commitment.
    pub grant_digest: Digest,
    /// Policy versions and revocation epoch commitment.
    pub policy_digest: Digest,
    /// Startup schema commitment.
    pub schema_digest: Digest,
    /// Fresh precondition and pre-state commitment.
    pub precondition_digest: Digest,
    /// Effective approval floor.
    pub required_approval: ApprovalLevel,
}
impl AuthorizationBinding {
    /// Build a preview for a selected grant, including pending approval decisions.
    #[must_use]
    pub fn from_evaluation(
        request: &Request,
        evaluation: &AuthorizationEvaluation,
        schema_digest: Digest,
        precondition_digest: Digest,
    ) -> Option<Self> {
        if !matches!(
            evaluation.decision,
            Decision::Allow | Decision::ApprovalRequired
        ) {
            return None;
        }
        Some(Self {
            request_digest: request.request_digest,
            principal_digest: digest_json(&request.principal),
            scope_digest: digest_json(&request.scope),
            package_digest: evaluation.package_digest,
            grant_digest: evaluation.grant_digest?,
            policy_digest: digest_json(&(
                "abi.policy.v1",
                &request.policy_version,
                &request.guild_constitution_version,
                &request.safety_policy_version,
                request.revocation_epoch,
            )),
            schema_digest,
            precondition_digest,
            required_approval: evaluation.required_approval,
        })
    }
}

/// Approval of the complete preview, never merely a tool name.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExactApproval {
    /// Existing role and call-bound approval envelope.
    pub approval: Approval,
    /// Exact preview accepted by the approver.
    pub binding: AuthorizationBinding,
    /// Trusted server issue time.
    pub issued_at_ms: u64,
}
impl ExactApproval {
    /// Binding and TTL validation; role/state/replay checks remain canonical kernel checks.
    #[must_use]
    pub fn matches(&self, binding: &AuthorizationBinding, now_ms: u64) -> bool {
        self.binding == *binding
            && self.issued_at_ms <= now_ms
            && self.approval.expires_at_ms > now_ms
            && self
                .approval
                .expires_at_ms
                .checked_sub(self.issued_at_ms)
                .is_some_and(|ttl| ttl > 0 && ttl <= MAX_APPROVAL_TTL_MS)
    }
}

/// Transaction input. All IDs must be bounded opaque references (1..=128 bytes).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrepareRequest {
    /// Stable call identity in the owning ledger.
    pub call_id: String,
    /// Startup registered tool identity.
    pub tool_id: String,
    /// Unique proposed execution identity; duplicate admission must refuse.
    pub execution_id: Digest,
    /// Exact evaluated authority.
    pub binding: AuthorizationBinding,
    /// Required above A0, consumed in the same transaction as intent and stage11.
    pub approval: Option<ExactApproval>,
    /// Trusted clock instant rechecked inside the transaction.
    pub now_ms: u64,
}
impl PrepareRequest {
    /// Structural bounds required before opening a transaction.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        [&self.call_id, &self.tool_id].iter().all(|id| {
            !id.is_empty()
                && id.len() <= 128
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
        }) && self.approval.as_ref().map_or(
            self.binding.required_approval == ApprovalLevel::A0None,
            |a| {
                a.matches(&self.binding, self.now_ms)
                    && [
                        &a.approval.decision_id,
                        &a.approval.grant_id,
                        &a.approval.approver.id,
                    ]
                    .iter()
                    .all(|id| bounded_opaque(id))
                    && a.approval
                        .coapprover
                        .as_ref()
                        .is_none_or(|p| bounded_opaque(&p.id))
                    && a.approval.call_id == self.call_id
                    && a.approval.call_digest == self.binding.request_digest
                    && a.approval.package_digest == self.binding.package_digest
                    && a.approval.state == crate::ApprovalState::Approved
                    && a.approval.level >= self.binding.required_approval
            },
        )
    }
}

/// Proof returned only after durable atomic admission; construction is a trusted-store boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedExecution {
    execution_id: Digest,
    binding: AuthorizationBinding,
}
impl PreparedExecution {
    /// Store implementors call only after committing consume + intent + stage11 audit.
    #[must_use]
    pub fn after_commit(execution_id: Digest, binding: AuthorizationBinding) -> Self {
        Self {
            execution_id,
            binding,
        }
    }
    /// Ledger identity.
    #[must_use]
    pub const fn execution_id(&self) -> Digest {
        self.execution_id
    }
    /// Exact admitted authority.
    #[must_use]
    pub const fn binding(&self) -> &AuthorizationBinding {
        &self.binding
    }
}

/// Admission result; only Prepared permits dispatch, and never on a duplicate lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // Fixed-size authorization proof; no heap required.
pub enum PrepareOutcome {
    /// New durable intent committed.
    Prepared(PreparedExecution),
    /// Human approval still required.
    Pending,
    /// Closed refusal.
    Refused(ReasonCode),
    /// Prior intent or ambiguous commit; fresh call and approval required.
    Uncertain,
}

/// Redacted lifecycle states; uncertainty never implies success.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeState {
    /// Intent exists, outcome not yet observed.
    Prepared,
    /// Authorization awaits a human.
    Pending,
    /// No admission.
    Refused,
    /// Verified postconditions.
    Succeeded,
    /// Known failed postconditions or adapter result.
    Failed,
    /// Postconditions cannot be verified; this is not success.
    Unverifiable,
    /// Interrupted or ambiguous effect, never automatically replayed.
    Uncertain,
    /// Compensation verified.
    Compensated,
    /// Compensation remains incomplete.
    CompensationFailed,
}

/// Versioned fixed-size content-free outcome record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TerminalOutcome {
    /// Internal record version; currently 1.
    pub version: u8,
    /// Closed observed state.
    pub state: OutcomeState,
    /// Optional evidence commitment; never fabricated for uncertainty.
    pub result_digest: Option<Digest>,
    /// Closed reason.
    pub reason: LifecycleReason,
}

/// Fixed-size bounded audit for one of the 17 stages.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageAudit {
    /// Internal version, currently 1.
    pub version: u8,
    /// Digest-only call/execution identity.
    pub execution_id: Digest,
    /// Stage 0 through 16.
    pub stage: u8,
    /// Exact authority commitment.
    pub binding_digest: Digest,
    /// Closed policy decision.
    pub decision: Decision,
    /// Closed reason.
    pub reason: LifecycleReason,
}
/// Redacted infrastructure failure.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
pub enum LifecycleError {
    /// Durable write/read failed.
    #[error("storage_unavailable")]
    Unavailable,
    /// Bounded audit capacity exhausted.
    #[error("audit_capacity")]
    Capacity,
    /// Corrupt or invalid state, never silently recreated.
    #[error("invalid_lifecycle_state")]
    InvalidState,
}

/// Owning durable ledger interface. Observational audit cannot implement admission.
pub trait DurableLifecycle {
    /// Fallible bounded stage append; validate version/stage before persistence.
    fn append_stage(&mut self, record: &StageAudit) -> Result<(), LifecycleError>;
    /// Atomically validate exact approval, consume it, reserve unique intent, append stage11,
    /// and commit. Duplicate calls/executions must never return Prepared again.
    fn prepare(&mut self, request: &PrepareRequest) -> Result<PrepareOutcome, LifecycleError>;
    /// Persist terminal state bound to the original intent; reject invalid transitions.
    fn finish(
        &mut self,
        prepared: &PreparedExecution,
        outcome: &TerminalOutcome,
    ) -> Result<(), LifecycleError>;
    /// Observational lookup, never an execution permit. Reopened unfinished intent is uncertain.
    fn lookup(&mut self, execution_id: Digest) -> Result<Option<TerminalOutcome>, LifecycleError>;
}

fn bounded_opaque(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
}

/// Internal lifecycle reason taxonomy, independent of frozen authority wire reasons.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleReason {
    /// Canonical authorization decision reason.
    Authorization(ReasonCode),
    /// Cancellation observed before dispatch.
    CancelledBeforeDispatch,
    /// Adapter returned a known failure.
    AdapterFailure,
    /// Observed postconditions failed.
    PostconditionFailed,
    /// Postconditions could not be verified.
    PostconditionUnverifiable,
    /// Process interruption left durable intent unfinished.
    Interrupted,
    /// The actual effect is unknown.
    OutcomeUnknown,
    /// Compensation did not complete.
    CompensationFailed,
    /// Durable infrastructure unavailable.
    StorageUnavailable,
}
