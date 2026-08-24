//! Immutable `ChangeSet` and approval vocabulary.

use crate::types::{
    ApprovalLevel, ApprovalState, Digest, Principal, PrincipalKind, Reversibility, RiskClass,
    Scope, TypeError,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// Closed compensation strength used by immutable executable proposals.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompensationClass {
    /// The complete prior observable state can be restored exactly.
    ExactRestore,
    /// Configuration can be compensated, but observed consequences may remain.
    BestEffort,
    /// No compensation is available.
    None,
}

impl CompensationClass {
    /// Map the execution vocabulary onto the existing package vocabulary.
    #[must_use]
    pub const fn reversibility(self) -> Reversibility {
        match self {
            Self::ExactRestore => Reversibility::Reversible,
            Self::BestEffort => Reversibility::ReversibleWithLoss,
            Self::None => Reversibility::Irreversible,
        }
    }
}

/// Immutable, content-free proposal for one exact platform change.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChangeSet {
    /// One-use operation identifier.
    pub operation_id: String,
    /// Canonical commitment over the complete proposal.
    pub change_set_digest: Digest,
    /// Human or service that requested consideration.
    pub requested_by: Principal,
    /// Abbey service identity that generated the immutable proposal.
    pub proposed_by: Principal,
    /// Exact guild/resource/subject scope.
    pub scope: Scope,
    /// Exact capability identifier.
    pub capability_id: String,
    /// Exact capability version.
    pub capability_version: String,
    /// Exact compiled package commitment.
    pub package_digest: Digest,
    /// Compensation strength independently reviewed for this proposal.
    pub compensation_class: CompensationClass,
    /// Reviewed risk class.
    pub risk: RiskClass,
    /// Required approval floor.
    pub required_approval: ApprovalLevel,
    /// Immutable precondition commitment.
    pub precondition_digest: Digest,
    /// Immutable expected-postcondition commitment.
    pub expected_postcondition_digest: Digest,
    /// Immutable rollback-plan commitment.
    pub rollback_digest: Digest,
    /// Complete prior-state commitment.
    pub snapshot_digest: Digest,
    /// Generator build/configuration commitment.
    pub generator_digest: Digest,
    /// Creation time from an injected clock.
    pub created_at_ms: u64,
    /// Exclusive proposal expiry.
    pub expires_at_ms: u64,
    /// Maximum lifetime of a subsequently prepared execution.
    pub prepared_ttl_ms: u64,
}

impl ChangeSet {
    /// Construct a bounded immutable proposal and compute its commitment.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operation_id: impl Into<String>,
        requested_by: Principal,
        proposed_by: Principal,
        scope: Scope,
        capability_id: impl Into<String>,
        capability_version: impl Into<String>,
        package_digest: Digest,
        compensation_class: CompensationClass,
        risk: RiskClass,
        required_approval: ApprovalLevel,
        precondition_digest: Digest,
        expected_postcondition_digest: Digest,
        rollback_digest: Digest,
        snapshot_digest: Digest,
        generator_digest: Digest,
        created_at_ms: u64,
        expires_at_ms: u64,
        prepared_ttl_ms: u64,
    ) -> Result<Self, TypeError> {
        let capability_id = capability_id.into();
        let capability_version = capability_version.into();
        if proposed_by.kind != PrincipalKind::Service
            || proposed_by.id == requested_by.id
            || !valid_capability_id(&capability_id)
            || !valid_semver(&capability_version)
            || expires_at_ms <= created_at_ms
            || expires_at_ms.saturating_sub(created_at_ms) > 300_000
            || !(1..=120_000).contains(&prepared_ttl_ms)
            || (compensation_class == CompensationClass::ExactRestore
                && rollback_digest == Digest::default())
        {
            return Err(TypeError::InvalidField);
        }
        let mut change_set = Self {
            operation_id: bounded_id(operation_id.into())?,
            change_set_digest: Digest::default(),
            requested_by,
            proposed_by,
            scope,
            capability_id,
            capability_version,
            package_digest,
            compensation_class,
            risk,
            required_approval,
            precondition_digest,
            expected_postcondition_digest,
            rollback_digest,
            snapshot_digest,
            generator_digest,
            created_at_ms,
            expires_at_ms,
            prepared_ttl_ms,
        };
        change_set.change_set_digest = change_set.computed_digest();
        Ok(change_set)
    }

    /// Recompute the domain-separated commitment over every immutable field.
    #[must_use]
    pub fn computed_digest(&self) -> Digest {
        let mut committed = self.clone();
        committed.change_set_digest = Digest::default();
        let encoded = serde_json::to_vec(&committed).expect("closed change-set fields serialize");
        let mut hasher = Sha256::new();
        hasher.update(b"abi-capability-change-set-v1\0");
        hasher.update(encoded);
        let mut output = [0_u8; 32];
        output.copy_from_slice(&hasher.finalize());
        Digest::from_bytes(output)
    }

    /// True only while the immutable proposal can still be approved.
    #[must_use]
    pub const fn is_live_at(&self, now_ms: u64) -> bool {
        now_ms >= self.created_at_ms && now_ms < self.expires_at_ms
    }
}

/// Human decision bound to one exact immutable [`ChangeSet`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChangeApproval {
    /// Unique decision identifier.
    pub decision_id: String,
    /// Exact proposal commitment; no edited proposal can reuse this decision.
    pub change_set_digest: Digest,
    /// Distinct authorized human approver.
    pub approved_by: Principal,
    /// Optional second distinct human for dual control.
    pub coapproved_by: Option<Principal>,
    /// Satisfied approval level.
    pub level: ApprovalLevel,
    /// Exclusive decision expiry.
    pub expires_at_ms: u64,
    /// Single-use lifecycle state.
    pub state: ApprovalState,
}

impl ChangeApproval {
    /// Bind an approval to one exact live proposal.
    pub fn approve(
        decision_id: impl Into<String>,
        change_set: &ChangeSet,
        approved_by: Principal,
        coapproved_by: Option<Principal>,
        level: ApprovalLevel,
        expires_at_ms: u64,
        now_ms: u64,
    ) -> Result<Self, TypeError> {
        let human = approved_by.kind != PrincipalKind::Service;
        let identities_are_distinct = approved_by.id != change_set.requested_by.id
            && approved_by.id != change_set.proposed_by.id
            && coapproved_by.as_ref().is_none_or(|second| {
                second.kind != PrincipalKind::Service
                    && second.id != approved_by.id
                    && second.id != change_set.requested_by.id
                    && second.id != change_set.proposed_by.id
            });
        let dual_control_matches =
            (level == ApprovalLevel::A5DualControl) == coapproved_by.is_some();
        let roles_satisfy = principal_satisfies_approval(&approved_by, level)
            && coapproved_by
                .as_ref()
                .is_none_or(|second| principal_satisfies_approval(second, level));
        if !change_set.is_live_at(now_ms)
            || !human
            || !identities_are_distinct
            || !dual_control_matches
            || !roles_satisfy
            || level < change_set.required_approval
            || expires_at_ms <= now_ms
            || expires_at_ms > change_set.expires_at_ms
        {
            return Err(TypeError::InvalidField);
        }
        Ok(Self {
            decision_id: bounded_id(decision_id.into())?,
            change_set_digest: change_set.change_set_digest,
            approved_by,
            coapproved_by,
            level,
            expires_at_ms,
            state: ApprovalState::Approved,
        })
    }
}

fn bounded_id(value: String) -> Result<String, TypeError> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
    {
        return Err(TypeError::InvalidField);
    }
    Ok(value)
}

fn valid_capability_id(value: &str) -> bool {
    value.len() <= 128
        && value.contains('.')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'.')
}

fn valid_semver(value: &str) -> bool {
    value.split('.').count() == 3
        && value
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

fn principal_satisfies_approval(principal: &Principal, level: ApprovalLevel) -> bool {
    match level {
        ApprovalLevel::A0None | ApprovalLevel::A1Actor => principal.kind != PrincipalKind::Service,
        ApprovalLevel::A2Manager => matches!(
            principal.kind,
            PrincipalKind::GuildManager
                | PrincipalKind::GuildAdministrator
                | PrincipalKind::GuildOwner
                | PrincipalKind::OrganizationOwner
        ),
        ApprovalLevel::A3Admin | ApprovalLevel::A5DualControl => matches!(
            principal.kind,
            PrincipalKind::GuildAdministrator
                | PrincipalKind::GuildOwner
                | PrincipalKind::OrganizationOwner
        ),
        ApprovalLevel::A4Owner => matches!(
            principal.kind,
            PrincipalKind::GuildOwner | PrincipalKind::OrganizationOwner
        ),
    }
}
