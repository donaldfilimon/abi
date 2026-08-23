use crate::audit::{AuditError, AuditRecord, AuditSink};
use crate::recording::{RecordingActuator, digest_json};
use crate::{
    Approval, ApprovalLevel, ApprovalState, CapabilityPackage, CapabilityRegistry, Clock, Decision,
    Grant, ReasonCode, Receipt, Request, Reversibility, RevocationState, RiskClass,
};
use abi_agent_runtime::CancellationToken;
use std::collections::BTreeSet;
use thiserror::Error;

/// All mutable authority facts for exactly one request.
pub struct AuthorizationContext<'a> {
    /// Injected deterministic clock.
    pub now_ms: &'a dyn Clock,
    /// Shared cancellation latch.
    pub cancellation: &'a CancellationToken,
    /// Request-scoped grant snapshot.
    pub grants: &'a [Grant],
    /// Optional single-use approval.
    pub approval: Option<&'a Approval>,
    /// Decision IDs already consumed by the caller's durable boundary.
    pub used_decision_ids: &'a BTreeSet<String>,
    /// Fresh platform and policy facts, rechecked immediately before recording.
    pub platform: &'a crate::PlatformFacts,
}

/// Deterministic authorization kernel over an immutable registry.
pub struct Kernel<'a> {
    registry: &'a CapabilityRegistry,
}

impl<'a> Kernel<'a> {
    /// Bind a kernel to a startup-compiled registry.
    #[must_use]
    pub const fn new(registry: &'a CapabilityRegistry) -> Self {
        Self { registry }
    }

    /// Authorize and, only after a successful audit write, record a simulated effect.
    #[allow(clippy::too_many_lines)]
    pub fn authorize_and_record(
        &self,
        request: &Request,
        context: &AuthorizationContext<'_>,
        audit: &mut dyn AuditSink,
        actuator: &mut RecordingActuator,
    ) -> Result<KernelOutcome, KernelError> {
        if request.computed_digest() != request.request_digest {
            return Self::finish(
                request,
                None,
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                ReasonCode::RequestDigestMismatch,
                false,
                false,
                audit,
                actuator,
            );
        }
        if context.cancellation.is_cancelled() {
            return Self::finish(
                request,
                None,
                None,
                ApprovalLevel::A0None,
                Decision::Pause,
                ReasonCode::Cancelled,
                true,
                false,
                audit,
                actuator,
            );
        }
        let Some(package) = self
            .registry
            .get(&request.capability_id, &request.capability_version)
        else {
            let reason = if self.registry.contains_id(&request.capability_id) {
                ReasonCode::CapabilityVersionMismatch
            } else {
                ReasonCode::CapabilityUnknown
            };
            return Self::finish(
                request,
                None,
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                reason,
                false,
                false,
                audit,
                actuator,
            );
        };
        if package.digest != request.package_digest {
            return Self::finish(
                request,
                Some(package),
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                ReasonCode::PackageDigestMismatch,
                false,
                false,
                audit,
                actuator,
            );
        }
        if CapabilityRegistry::is_prohibited(package) {
            return Self::finish(
                request,
                Some(package),
                None,
                ApprovalLevel::A5DualControl,
                Decision::Deny,
                ReasonCode::Prohibited,
                false,
                false,
                audit,
                actuator,
            );
        }

        let mut candidates = context
            .grants
            .iter()
            .filter(|grant| {
                grant.capability_id == request.capability_id && grant.recipient == request.principal
            })
            .peekable();
        if candidates.peek().is_none() {
            let reason = if context
                .grants
                .iter()
                .any(|grant| grant.capability_id == request.capability_id)
            {
                ReasonCode::PrincipalMismatch
            } else {
                ReasonCode::NoMatchingGrant
            };
            return Self::finish(
                request,
                Some(package),
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                reason,
                false,
                false,
                audit,
                actuator,
            );
        }
        let now = context.now_ms.now_ms();
        let mut valid = Vec::new();
        let mut failures = Vec::new();
        for grant in candidates {
            if let Some(reason) = grant_failure(grant, package, request, context, now) {
                failures.push((reason_priority(reason), grant.id.as_str(), grant, reason));
            } else {
                valid.push(grant);
            }
        }
        let grant = valid
            .into_iter()
            .min_by(|left, right| left.id.cmp(&right.id));
        let Some(grant) = grant else {
            let (_, _, failed_grant, reason) = failures
                .into_iter()
                .min_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)))
                .expect("candidate grants produced a reason");
            let required = required_approval(
                package,
                failed_grant.confirmation_level,
                context.platform.regime_approval_floor,
                context.platform.safety_approval_floor,
            );
            return Self::finish(
                request,
                Some(package),
                Some(failed_grant),
                required,
                Decision::Deny,
                reason,
                false,
                false,
                audit,
                actuator,
            );
        };
        let required = required_approval(
            package,
            grant.confirmation_level,
            context.platform.regime_approval_floor,
            context.platform.safety_approval_floor,
        );
        if required > ApprovalLevel::A0None {
            let Some(approval) = context.approval else {
                return Self::finish(
                    request,
                    Some(package),
                    Some(grant),
                    required,
                    Decision::ApprovalRequired,
                    ReasonCode::ApprovalMissing,
                    false,
                    false,
                    audit,
                    actuator,
                );
            };
            let reason = if context.used_decision_ids.contains(&approval.decision_id) {
                Some(ReasonCode::ApprovalReplayed)
            } else if now >= approval.expires_at_ms {
                Some(ReasonCode::ApprovalExpired)
            } else if approval.state != ApprovalState::Approved
                || approval.call_id != request.call_id
                || approval.call_digest != request.request_digest
                || approval.package_digest != package.digest
                || approval.grant_id != grant.id
                || approval.level < required
            {
                Some(ReasonCode::ApprovalInsufficient)
            } else if required >= ApprovalLevel::A2Manager
                && approval.approver.id == request.principal.id
            {
                Some(ReasonCode::SelfApproval)
            } else if !approver_roles_satisfy(approval, required, &request.principal) {
                Some(ReasonCode::ApprovalInsufficient)
            } else {
                None
            };
            if let Some(reason) = reason {
                let decision = if matches!(
                    reason,
                    ReasonCode::ApprovalExpired | ReasonCode::ApprovalInsufficient
                ) {
                    Decision::ApprovalRequired
                } else {
                    Decision::Deny
                };
                return Self::finish(
                    request,
                    Some(package),
                    Some(grant),
                    required,
                    decision,
                    reason,
                    false,
                    false,
                    audit,
                    actuator,
                );
            }
        }

        if context.cancellation.is_cancelled() {
            return Self::finish(
                request,
                Some(package),
                Some(grant),
                required,
                Decision::Pause,
                ReasonCode::Cancelled,
                true,
                false,
                audit,
                actuator,
            );
        }
        let postconditions_hold = actuator.postconditions_hold(package);
        let reason = if postconditions_hold {
            ReasonCode::Authorized
        } else {
            ReasonCode::PostconditionFailed
        };
        let decision = if postconditions_hold {
            Decision::Allow
        } else {
            Decision::Deny
        };
        Self::finish(
            request,
            Some(package),
            Some(grant),
            required,
            decision,
            reason,
            false,
            postconditions_hold,
            audit,
            actuator,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        request: &Request,
        package: Option<&CapabilityPackage>,
        grant: Option<&Grant>,
        required: ApprovalLevel,
        decision: Decision,
        reason: ReasonCode,
        cancelled: bool,
        record_effect: bool,
        audit: &mut dyn AuditSink,
        actuator: &mut RecordingActuator,
    ) -> Result<KernelOutcome, KernelError> {
        let package_digest = package.map_or(request.package_digest, |value| value.digest);
        let audit_record = AuditRecord {
            attempt_id: request.attempt_id.clone(),
            request_digest: request.request_digest,
            principal_digest: digest_json(&request.principal),
            scope_digest: digest_json(&request.scope),
            package_digest,
            grant_digest: grant.map(digest_json),
            stage: if record_effect { 10 } else { 5 },
            decision,
            reason,
            parameter_digest: request.parameter_digest,
            approval_required: required,
            cancelled,
            redacted: true,
        };
        audit.record(&audit_record)?;
        let result_digest = if record_effect {
            Some(actuator.record(request, package.expect("effect requires package")))
        } else {
            None
        };
        Ok(KernelOutcome {
            receipt: Receipt {
                attempt_id: request.attempt_id.clone(),
                request_digest: request.request_digest,
                package_digest,
                decision,
                reason,
                required_approval: required,
                result_digest,
                postconditions_satisfied: record_effect,
                redacted: true,
            },
            audit_record,
        })
    }
}

fn grant_failure(
    grant: &Grant,
    package: &CapabilityPackage,
    request: &Request,
    context: &AuthorizationContext<'_>,
    now: u64,
) -> Option<ReasonCode> {
    if grant.scope != request.scope {
        Some(ReasonCode::ScopeMismatch)
    } else if grant.capability_version != request.capability_version {
        Some(ReasonCode::CapabilityVersionMismatch)
    } else if grant.package_digest != request.package_digest {
        Some(ReasonCode::PackageDigestMismatch)
    } else if now < grant.not_before_ms {
        Some(ReasonCode::GrantNotYetValid)
    } else if now >= grant.expires_at_ms {
        Some(ReasonCode::GrantExpired)
    } else if grant.revocation == RevocationState::Suspended {
        Some(ReasonCode::GrantSuspended)
    } else if grant.revocation == RevocationState::Revoked {
        Some(ReasonCode::GrantRevoked)
    } else if grant.revocation_epoch != request.revocation_epoch
        || context.platform.revocation_epoch != request.revocation_epoch
    {
        Some(ReasonCode::RevocationEpochMismatch)
    } else if grant.policy_version != request.policy_version
        || grant.guild_constitution_version != request.guild_constitution_version
        || grant.safety_policy_version != request.safety_policy_version
        || context.platform.policy_version != request.policy_version
        || context.platform.guild_constitution_version != request.guild_constitution_version
        || context.platform.safety_policy_version != request.safety_policy_version
    {
        Some(ReasonCode::PolicyVersionMismatch)
    } else if package.risk > grant.risk_ceiling {
        Some(ReasonCode::NoMatchingGrant)
    } else if !package
        .required_permissions
        .is_subset(&context.platform.permissions)
    {
        Some(ReasonCode::PlatformPermissionMissing)
    } else {
        None
    }
}

const fn reason_priority(reason: ReasonCode) -> u8 {
    match reason {
        ReasonCode::GrantRevoked => 0,
        ReasonCode::GrantSuspended => 1,
        ReasonCode::GrantExpired => 2,
        ReasonCode::GrantNotYetValid => 3,
        ReasonCode::RevocationEpochMismatch => 4,
        ReasonCode::PolicyVersionMismatch => 5,
        ReasonCode::PlatformPermissionMissing => 6,
        ReasonCode::PackageDigestMismatch => 7,
        ReasonCode::CapabilityVersionMismatch => 8,
        ReasonCode::ScopeMismatch => 9,
        _ => 10,
    }
}

fn approver_roles_satisfy(
    approval: &Approval,
    required: ApprovalLevel,
    actor: &crate::Principal,
) -> bool {
    if required == ApprovalLevel::A1Actor {
        return approval.approver.id == actor.id
            && actor.kind == crate::PrincipalKind::HumanSubject;
    }
    if required == ApprovalLevel::A5DualControl {
        return approver_strength(approval.approver.kind) >= ApprovalLevel::A3Admin
            && approval.coapprover.as_ref().is_some_and(|other| {
                approver_strength(other.kind) >= ApprovalLevel::A3Admin
                    && other.id != approval.approver.id
                    && other.id != actor.id
                    && approval.approver.id != actor.id
            });
    }
    approver_strength(approval.approver.kind) >= required
}

const fn approver_strength(kind: crate::PrincipalKind) -> ApprovalLevel {
    match kind {
        crate::PrincipalKind::GuildManager => ApprovalLevel::A2Manager,
        crate::PrincipalKind::GuildAdministrator => ApprovalLevel::A3Admin,
        crate::PrincipalKind::GuildOwner | crate::PrincipalKind::OrganizationOwner => {
            ApprovalLevel::A4Owner
        }
        crate::PrincipalKind::HumanSubject | crate::PrincipalKind::Service => ApprovalLevel::A0None,
    }
}

fn required_approval(
    package: &CapabilityPackage,
    grant: ApprovalLevel,
    regime: ApprovalLevel,
    safety: ApprovalLevel,
) -> ApprovalLevel {
    let risk = match package.risk {
        RiskClass::Informational | RiskClass::Low => ApprovalLevel::A0None,
        RiskClass::Medium => ApprovalLevel::A2Manager,
        RiskClass::High | RiskClass::Prohibited => ApprovalLevel::A4Owner,
    };
    let irreversible = if package.reversibility == Reversibility::Irreversible {
        ApprovalLevel::A4Owner
    } else {
        ApprovalLevel::A0None
    };
    package
        .approval_floor
        .max(grant)
        .max(risk)
        .max(irreversible)
        .max(regime)
        .max(safety)
}

/// Redacted authorization result and exact audit record.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct KernelOutcome {
    /// Caller-visible redacted receipt.
    pub receipt: Receipt,
    /// Persisted redacted audit record.
    pub audit_record: AuditRecord,
}

/// Closed kernel infrastructure failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    /// Audit persistence failed before actuation.
    #[error("audit_failed:{0}")]
    Audit(#[from] AuditError),
}
