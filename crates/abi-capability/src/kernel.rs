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

    /// Evaluate exact preview approval with server TTL; durable consumption is still required.
    #[must_use]
    pub fn evaluate_bound(
        &self,
        request: &Request,
        context: &AuthorizationContext<'_>,
        schema_digest: crate::Digest,
        precondition_digest: crate::Digest,
        approval: Option<&crate::lifecycle::ExactApproval>,
    ) -> AuthorizationEvaluation {
        let bound_context = AuthorizationContext {
            approval: approval.map(|a| &a.approval),
            ..*context
        };
        let mut result = self.evaluate(request, &bound_context);
        if result.decision == Decision::Allow && result.required_approval > ApprovalLevel::A0None {
            let binding = crate::lifecycle::AuthorizationBinding::from_evaluation(
                request,
                &result,
                schema_digest,
                precondition_digest,
            );
            if !approval
                .zip(binding.as_ref())
                .is_some_and(|(a, b)| a.matches(b, context.now_ms.now_ms()))
            {
                result.decision = Decision::ApprovalRequired;
                result.reason = ReasonCode::ApprovalInsufficient;
            }
        }
        result
    }

    /// Evaluate an exact approval against the startup source's effective TTL policy.
    #[must_use]
    pub fn evaluate_source_bound(
        &self,
        request: &Request,
        context: &AuthorizationContext<'_>,
        source: &crate::source::CompiledSource,
        precondition_digest: crate::Digest,
        approval: Option<&crate::lifecycle::ExactApproval>,
    ) -> AuthorizationEvaluation {
        let mut result = self.evaluate_bound(
            request,
            context,
            source.schema_digest(),
            precondition_digest,
            approval,
        );
        if source.package().digest != request.package_digest {
            result.decision = Decision::Deny;
            result.reason = ReasonCode::PackageDigestMismatch;
        } else if result.decision == Decision::Allow
            && approval.is_some_and(|a| {
                a.approval
                    .expires_at_ms
                    .checked_sub(a.issued_at_ms)
                    .is_none_or(|ttl| ttl > source.source().expiry_policy_ms)
            })
        {
            result.decision = Decision::ApprovalRequired;
            result.reason = ReasonCode::ApprovalInsufficient;
        }
        result
    }

    /// Evaluate a declared inverse with the original authority floor and exact origin.
    /// Fresh source prerequisites and durable original-state checks remain required.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_compensation_source_bound(
        &self,
        request: &Request,
        context: &AuthorizationContext<'_>,
        inverse: &crate::source::CompiledSource,
        forward: &crate::source::CompiledSource,
        original: &crate::lifecycle::PreparedExecution,
        precondition_digest: crate::Digest,
        approval: Option<&crate::lifecycle::ExactApproval>,
    ) -> CompensationEvaluation {
        use crate::lifecycle::{AuthorizationBinding, CompensationBinding};
        let declaration = forward.source().compensation.as_ref();
        let linked = declaration.is_some_and(|c| {
            c.capability_id == inverse.package().id && c.version == inverse.package().version
        }) && original.binding().package_digest == forward.package().digest
            && original.binding().schema_digest == forward.schema_digest()
            && original.binding().scope_digest == digest_json(&forward.source().resource_scope)
            && original.binding().scope_digest == digest_json(&request.scope)
            && inverse.source().resource_scope == forward.source().resource_scope
            && inverse.package().digest == request.package_digest
            && inverse.package().risk <= forward.package().risk
            && forward.package().risk != RiskClass::Prohibited;
        let mut platform = context.platform.clone();
        if !declaration.is_some_and(|c| c.preauthorized) {
            platform.safety_approval_floor = platform
                .safety_approval_floor
                .max(original.binding().required_approval);
        }
        let bound_context = AuthorizationContext {
            platform: &platform,
            approval: approval.map(|a| &a.approval),
            ..*context
        };
        let mut result = self.evaluate(request, &bound_context);
        if !linked {
            result.decision = Decision::Deny;
            result.reason = ReasonCode::PackageDigestMismatch;
        }
        let mut binding = AuthorizationBinding::from_evaluation(
            request,
            &result,
            inverse.schema_digest(),
            precondition_digest,
        );
        if let Some(binding) = &mut binding {
            binding.compensation_origin = Some(CompensationBinding::from_prepared(original));
        }
        if result.decision == Decision::Allow
            && (result.required_approval > ApprovalLevel::A0None || approval.is_some())
            && !approval.zip(binding.as_ref()).is_some_and(|(a, b)| {
                a.matches(b, context.now_ms.now_ms())
                    && a.approval
                        .expires_at_ms
                        .checked_sub(a.issued_at_ms)
                        .is_some_and(|ttl| ttl <= inverse.source().expiry_policy_ms)
            })
        {
            result.decision = Decision::ApprovalRequired;
            result.reason = ReasonCode::ApprovalInsufficient;
        }
        CompensationEvaluation {
            evaluation: result,
            binding,
        }
    }

    /// Authorize and, only after a successful audit write, record a simulated effect.
    pub fn authorize_and_record(
        &self,
        request: &Request,
        context: &AuthorizationContext<'_>,
        audit: &mut dyn AuditSink,
        actuator: &mut RecordingActuator,
    ) -> Result<KernelOutcome, KernelError> {
        let mut evaluated = self.evaluate(request, context);
        let package = self
            .registry
            .get(&request.capability_id, &request.capability_version)
            .filter(|p| p.digest == evaluated.package_digest);
        let grant = evaluated
            .grant_digest
            .and_then(|digest| context.grants.iter().find(|g| digest_json(*g) == digest));
        let record_effect = evaluated.decision == Decision::Allow
            && package.is_some_and(|p| actuator.postconditions_hold(p));
        if evaluated.decision == Decision::Allow && !record_effect {
            evaluated.decision = Decision::Deny;
            evaluated.reason = ReasonCode::PostconditionFailed;
        }
        Self::finish(
            request,
            package,
            grant,
            evaluated.required_approval,
            evaluated.decision,
            evaluated.reason,
            evaluated.reason == ReasonCode::Cancelled,
            record_effect,
            audit,
            actuator,
        )
    }

    /// Evaluate all authority checks without audit persistence or effects.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(
        &self,
        request: &Request,
        context: &AuthorizationContext<'_>,
    ) -> AuthorizationEvaluation {
        if request.computed_digest() != request.request_digest {
            return evaluation(
                request,
                None,
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                ReasonCode::RequestDigestMismatch,
            );
        }
        if context.cancellation.is_cancelled() {
            return evaluation(
                request,
                None,
                None,
                ApprovalLevel::A0None,
                Decision::Pause,
                ReasonCode::Cancelled,
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
            return evaluation(
                request,
                None,
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                reason,
            );
        };
        if package.digest != request.package_digest {
            return evaluation(
                request,
                Some(package),
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                ReasonCode::PackageDigestMismatch,
            );
        }
        if CapabilityRegistry::is_prohibited(package) {
            return evaluation(
                request,
                Some(package),
                None,
                ApprovalLevel::A5DualControl,
                Decision::Deny,
                ReasonCode::Prohibited,
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
            return evaluation(
                request,
                Some(package),
                None,
                ApprovalLevel::A0None,
                Decision::Deny,
                reason,
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
            return evaluation(
                request,
                Some(package),
                Some(failed_grant),
                required,
                Decision::Deny,
                reason,
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
                return evaluation(
                    request,
                    Some(package),
                    Some(grant),
                    required,
                    Decision::ApprovalRequired,
                    ReasonCode::ApprovalMissing,
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
                return evaluation(
                    request,
                    Some(package),
                    Some(grant),
                    required,
                    decision,
                    reason,
                );
            }
        }

        if context.cancellation.is_cancelled() {
            return evaluation(
                request,
                Some(package),
                Some(grant),
                required,
                Decision::Pause,
                ReasonCode::Cancelled,
            );
        }
        evaluation(
            request,
            Some(package),
            Some(grant),
            required,
            Decision::Allow,
            ReasonCode::Authorized,
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
    } else if !issuer_is_authorized(grant, package) || package.risk > grant.risk_ceiling {
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

fn issuer_is_authorized(grant: &Grant, package: &CapabilityPackage) -> bool {
    use crate::PrincipalKind::{GuildAdministrator, GuildOwner, OrganizationOwner};
    let permitted = if grant.scope.guild_ref.is_none() {
        grant.issuer.kind == OrganizationOwner
    } else if package.risk <= RiskClass::Low {
        matches!(grant.issuer.kind, GuildOwner | GuildAdministrator)
    } else {
        grant.issuer.kind == GuildOwner
    };
    permitted
        && (package.reversibility != Reversibility::Irreversible || grant.issuer.kind == GuildOwner)
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

/// Content-free pure authorization result. This is not durable admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizationEvaluation {
    /// Closed policy decision.
    pub decision: Decision,
    /// Closed refusal or success reason.
    pub reason: ReasonCode,
    /// Effective approval floor.
    pub required_approval: ApprovalLevel,
    /// Exact package commitment.
    pub package_digest: crate::Digest,
    /// Exact selected grant commitment, including issuer and lifecycle.
    pub grant_digest: Option<crate::Digest>,
}

fn evaluation(
    request: &Request,
    package: Option<&CapabilityPackage>,
    grant: Option<&Grant>,
    required_approval: ApprovalLevel,
    decision: Decision,
    reason: ReasonCode,
) -> AuthorizationEvaluation {
    AuthorizationEvaluation {
        decision,
        reason,
        required_approval,
        package_digest: package.map_or(request.package_digest, |p| p.digest),
        grant_digest: grant.map(digest_json),
    }
}

/// Canonical inverse decision with its exact origin-bound preview, when available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompensationEvaluation {
    /// Pure authority result, never durable admission.
    pub evaluation: AuthorizationEvaluation,
    /// Exact preview for pending or allowed inverse calls.
    pub binding: Option<crate::lifecycle::AuthorizationBinding>,
}
