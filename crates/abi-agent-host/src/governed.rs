//! Recording-only governance over the shared bounded host driver.
#![allow(missing_docs)]
use crate::governed_authorization::{CompensationContext, evaluate};
use crate::{AgentHost, GovernedRunFailure, HostBudget, HostRunReport, ToolExecutionContext};
use abi_agent_runtime::{
    CancellationToken, DenyAllPolicy, EventSink, ModelProvider, ModelRequest, ToolCall,
    ToolRegistry, ToolResult, ToolSpec, ToolStatus,
};
use abi_capability::{
    AuthorizationContext, Clock, Decision, Digest, Grant, Kernel, PlatformFacts, ReasonCode,
    RecordingActuator, Request,
    lifecycle::{
        AuthorizationBinding, DurableLifecycle, ExactApproval, LifecycleError, LifecycleReason,
        OutcomeState, PrepareOutcome, PrepareRequest, PreparedExecution, StageAudit,
        TerminalOutcome,
    },
    source::{
        AdapterBinding, BudgetRequirement, DataClass, ResidencyPolicy, RetentionClass, SourceFacts,
        SourceRegistry,
    },
};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Startup-owned authority template. Generated input cannot choose these fields.
#[derive(Clone)]
pub struct GovernedRegistration {
    pub tool_name: String,
    pub request: Request,
}

/// A fresh trusted observation, acquired outside pure authorization policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GovernedFacts {
    pub platform: PlatformFacts,
    pub grants: Vec<Grant>,
    pub used_decision_ids: BTreeSet<String>,
    pub predicates: BTreeSet<String>,
    pub captured_pre_state: Option<Digest>,
    pub observed_at_ms: u64,
    pub expires_at_ms: u64,
    pub permitted_data_classes: BTreeSet<DataClass>,
    pub residency: ResidencyPolicy,
    pub retention: RetentionClass,
    pub safety_allows: bool,
    pub rate_allows: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Postconditions {
    Verified,
    Failed,
    Unverifiable,
}

/// Caller-owned trusted platform bridge; errors are closed and content-free.
pub trait GovernedFactProvider {
    fn observe(&mut self, request: &Request) -> Result<GovernedFacts, LifecycleError>;
    fn postconditions(
        &mut self,
        request: &Request,
        required: &BTreeSet<String>,
    ) -> Result<Vec<crate::PostconditionObservation>, LifecycleError>;
}

pub struct GovernedOptions<'a> {
    pub clock: &'a dyn Clock,
    pub host_budget: HostBudget,
    pub approval: Option<ExactApproval>,
    pub idempotency_key: Option<Digest>,
    pub requested_deadline_ms: u64,
    pub requested_budgets: BudgetRequirement,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // Fixed-size bounded preview avoids heap allocation.
pub enum GovernedStop {
    Completed,
    DriverStopped(crate::HostStopReason),
    PendingApproval(AuthorizationBinding),
    Refused(LifecycleReason),
    Uncertain,
    Infrastructure(LifecycleError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GovernedRunReport {
    pub driver: HostRunReport,
    pub governance: GovernedStop,
    pub outcomes: Vec<TerminalOutcome>,
    pub executions: Vec<PreparedExecution>,
    pub pending_approval: Option<ApprovalPreview>,
}

/// Closed explicit-compensation submission failure.
#[derive(Debug)]
pub enum GovernedCompensationError {
    InvalidOriginal,
    Lifecycle(LifecycleError),
    Driver(Box<GovernedRunFailure>),
}

/// Structured inspection data derived only from the startup-compiled source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalPreview {
    pub binding: AuthorizationBinding,
    pub tool: String,
    pub call_reference: String,
    pub capability: String,
    pub version: String,
    pub parameter_digest: Digest,
    pub effect: abi_capability::SideEffect,
    pub risk: abi_capability::RiskClass,
    pub reversibility: abi_capability::Reversibility,
    pub postconditions: BTreeSet<String>,
    pub compensation: Option<CompensationPreview>,
    pub rollback: abi_capability::source::RollbackSpec,
    pub expires_at_ms: u64,
    pub uncertainty: PreviewUncertainty,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompensationPreview {
    pub capability: String,
    pub version: String,
    pub required_pre_state: BTreeSet<String>,
    pub preauthorized: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewUncertainty {
    /// Only a simulated recording is possible; live platform effects are unverified.
    RecordingOnly,
}

/// Constructor exposes only a concrete recording actuator, never a generic live executor.
pub struct GovernedHost<'a> {
    provider: &'a dyn ModelProvider,
    sources: &'a SourceRegistry,
    registrations: BTreeMap<String, GovernedRegistration>,
    specs: Vec<ToolSpec>,
}
impl<'a> GovernedHost<'a> {
    pub fn new(
        provider: &'a dyn ModelProvider,
        sources: &'a SourceRegistry,
        registrations: Vec<GovernedRegistration>,
    ) -> Result<Self, LifecycleError> {
        let mut map = BTreeMap::new();
        let mut specs = Vec::new();
        for registration in registrations {
            let request = &registration.request;
            let source = sources
                .get(&request.capability_id, &request.capability_version)
                .ok_or(LifecycleError::InvalidState)?;
            if request.package_digest != source.package().digest
                || request.scope != source.source().resource_scope
                || registration.tool_name.is_empty()
                || registration.tool_name.len() > 128
                || !registration
                    .tool_name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
                || source.source().compensation.as_ref().is_some_and(|c| {
                    c.pre_state.len() > 32 || c.pre_state.iter().any(|p| p.len() > 64)
                })
            {
                return Err(LifecycleError::InvalidState);
            }
            let spec = ToolSpec::new(&registration.tool_name)
                .with_input_schema(source.source().input_schema.to_string());
            specs.push(spec);
            if map
                .insert(registration.tool_name.clone(), registration)
                .is_some()
            {
                return Err(LifecycleError::InvalidState);
            }
        }
        Ok(Self {
            provider,
            sources,
            registrations: map,
            specs,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn run(
        &self,
        request: &ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        options: GovernedOptions<'_>,
        facts: &mut dyn GovernedFactProvider,
        ledger: &mut dyn DurableLifecycle,
        actuator: &mut RecordingActuator,
    ) -> Result<GovernedRunReport, Box<GovernedRunFailure>> {
        self.run_mode(
            request,
            sink,
            cancellation,
            options,
            facts,
            ledger,
            actuator,
            None,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    fn run_mode(
        &self,
        request: &ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        options: GovernedOptions<'_>,
        facts: &mut dyn GovernedFactProvider,
        ledger: &mut dyn DurableLifecycle,
        actuator: &mut RecordingActuator,
        compensation: Option<CompensationContext<'_>>,
    ) -> Result<GovernedRunReport, Box<GovernedRunFailure>> {
        let registry = Specs(self.specs.clone());
        let legacy = AgentHost::new(
            self.provider,
            &registry,
            &DenyAllPolicy,
            &NoAudit,
            &NoExecutor,
        )
        .map_err(|error| Box::new(GovernedRunFailure::from_driver(&error)))?;
        let mut hook = Governance {
            host: self,
            options: &options,
            facts,
            ledger,
            actuator,
            stop: GovernedStop::Completed,
            outcomes: Vec::new(),
            executions: Vec::new(),
            pending_approval: None,
            audit_error: None,
            calls: 0,
            compensation,
        };
        let mut budget = options.host_budget;
        budget.max_duration = budget.max_duration.min(std::time::Duration::from_millis(
            options.requested_deadline_ms,
        ));
        budget.max_events = budget
            .max_events
            .min(u64::from(options.requested_budgets.max_events));
        budget.max_tool_calls = budget
            .max_tool_calls
            .min(u64::from(options.requested_budgets.max_calls));
        budget.max_tool_result_bytes = budget
            .max_tool_result_bytes
            .min(options.requested_budgets.max_output_bytes as usize);
        let driver =
            match legacy.run_with_hook(request, sink, cancellation, budget, &mut Some(&mut hook)) {
                Ok(driver) => driver,
                Err(error) => {
                    let mut failure = GovernedRunFailure::from_driver(&error);
                    failure.audit_error = hook.audit_error;
                    failure.executions = hook.executions;
                    failure.outcomes = hook.outcomes;
                    failure.pending_approval = hook.pending_approval;
                    return Err(Box::new(failure));
                }
            };
        if hook.stop == GovernedStop::Completed && !driver.is_complete() {
            let error = match driver.stop {
                crate::HostStopReason::BudgetExhausted(limit) => {
                    crate::HostError::BudgetExceeded(limit)
                }
                _ => crate::HostError::BudgetExceeded(crate::HostBudgetLimit::Deadline),
            };
            crate::host::CallHook::rejected(&mut hook, &error);
            hook.stop = hook.audit_error.map_or(
                GovernedStop::DriverStopped(driver.stop),
                GovernedStop::Infrastructure,
            );
        }
        Ok(GovernedRunReport {
            driver,
            governance: hook.stop,
            outcomes: hook.outcomes,
            executions: hook.executions,
            pending_approval: hook.pending_approval,
        })
    }
    /// Explicit compensation submission. Only a known failed original intent admits
    /// its declared inverse, with the inverse's own fresh grant/approval and scope.
    #[allow(clippy::too_many_arguments)]
    pub fn run_compensation(
        &self,
        original_tool: &str,
        original: &PreparedExecution,
        request: &ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        options: GovernedOptions<'_>,
        facts: &mut dyn GovernedFactProvider,
        ledger: &mut dyn DurableLifecycle,
        actuator: &mut RecordingActuator,
    ) -> Result<GovernedRunReport, GovernedCompensationError> {
        let refuse = || GovernedCompensationError::InvalidOriginal;
        let registration = self.registrations.get(original_tool).ok_or_else(refuse)?;
        let source = self
            .sources
            .get(
                &registration.request.capability_id,
                &registration.request.capability_version,
            )
            .ok_or_else(refuse)?;
        if original.binding().package_digest != source.package().digest
            || original.binding().scope_digest != commitment(&registration.request.scope)
            || !matches!(
                ledger
                    .lookup(original.execution_id())
                    .map_err(GovernedCompensationError::Lifecycle)?,
                Some(TerminalOutcome {
                    state: OutcomeState::Failed,
                    ..
                })
            )
        {
            return Err(refuse());
        }
        let inverse = source.source().compensation.as_ref().ok_or_else(refuse)?;
        let registrations: Vec<_> = self
            .registrations
            .values()
            .filter(|candidate| {
                candidate.request.capability_id == inverse.capability_id
                    && candidate.request.capability_version == inverse.version
                    && candidate.request.scope == registration.request.scope
            })
            .cloned()
            .collect();
        if registrations.is_empty() {
            return Err(refuse());
        }
        // Never inherit the original approval. Canonical matching requires a new
        // exact inverse preview (or its own A0 grant), and a unique call identity.
        let inverse_host = Self::new(self.provider, self.sources, registrations)
            .map_err(GovernedCompensationError::Lifecycle)?;
        inverse_host
            .run_mode(
                request,
                sink,
                cancellation,
                options,
                facts,
                ledger,
                actuator,
                Some(CompensationContext {
                    forward: source,
                    original,
                }),
            )
            .map_err(GovernedCompensationError::Driver)
    }
}

struct Governance<'a, 'b> {
    host: &'a GovernedHost<'b>,
    options: &'a GovernedOptions<'a>,
    facts: &'a mut dyn GovernedFactProvider,
    ledger: &'a mut dyn DurableLifecycle,
    actuator: &'a mut RecordingActuator,
    stop: GovernedStop,
    outcomes: Vec<TerminalOutcome>,
    executions: Vec<PreparedExecution>,
    pending_approval: Option<ApprovalPreview>,
    audit_error: Option<LifecycleError>,
    calls: u32,
    compensation: Option<CompensationContext<'a>>,
}

impl crate::host::CallHook for Governance<'_, '_> {
    fn rejected(&mut self, error: &crate::HostError) {
        let identity = commitment(&(
            "abi.host.rejected-run.v1",
            self.host
                .registrations
                .iter()
                .map(|(tool, registration)| (tool, &registration.request))
                .collect::<Vec<_>>(),
        ));
        let reason = match error {
            crate::HostError::Provider(_) => LifecycleReason::StorageUnavailable,
            crate::HostError::BudgetExceeded(_) => LifecycleReason::CancelledBeforeDispatch,
            _ => LifecycleReason::Authorization(ReasonCode::RequestDigestMismatch),
        };
        self.audit_error = self
            .stage(identity, 0, Digest::default(), Decision::Deny, reason)
            .err();
    }
    fn call(
        &mut self,
        call: &ToolCall,
        context: ToolExecutionContext<'_>,
        limit: usize,
    ) -> Option<ToolResult> {
        match self.execute(call, context) {
            Ok(true) => Some(ToolResult::with_limit(
                call.id.clone(),
                ToolStatus::Ok,
                "recorded",
                limit,
            )),
            Ok(false) => None,
            Err(error) => {
                if self.stop != GovernedStop::Uncertain {
                    self.stop = GovernedStop::Infrastructure(error);
                }
                None
            }
        }
    }
}

impl Governance<'_, '_> {
    fn stage(
        &mut self,
        id: Digest,
        stage: u8,
        binding: Digest,
        decision: Decision,
        reason: LifecycleReason,
    ) -> Result<(), LifecycleError> {
        self.ledger.append_stage(&StageAudit {
            version: 1,
            execution_id: id,
            stage,
            binding_digest: binding,
            decision,
            reason,
        })
    }
    fn refuse(
        &mut self,
        id: Digest,
        stage: u8,
        reason: LifecycleReason,
    ) -> Result<bool, LifecycleError> {
        self.refuse_bound(id, stage, Digest::default(), reason)
    }
    fn refuse_bound(
        &mut self,
        id: Digest,
        stage: u8,
        binding: Digest,
        reason: LifecycleReason,
    ) -> Result<bool, LifecycleError> {
        self.stage(id, stage, binding, Decision::Deny, reason)?;
        self.stop = GovernedStop::Refused(reason);
        Ok(false)
    }
    fn decline_prepared(&mut self, prepared: &PreparedExecution) -> Result<bool, LifecycleError> {
        let reason = LifecycleReason::CancelledBeforeDispatch;
        let outcome = TerminalOutcome {
            version: 1,
            state: OutcomeState::NotDispatched,
            result_digest: None,
            reason,
        };
        self.ledger.finish(prepared, &outcome)?;
        self.outcomes.push(outcome);
        self.refuse_bound(
            prepared.execution_id(),
            12,
            commitment(prepared.binding()),
            reason,
        )
    }
    fn fresh(&self, facts: &GovernedFacts) -> bool {
        let now = self.options.clock.now_ms();
        facts.observed_at_ms <= now && now < facts.expires_at_ms
    }

    #[allow(clippy::too_many_lines)]
    fn execute(
        &mut self,
        call: &ToolCall,
        context: ToolExecutionContext<'_>,
    ) -> Result<bool, LifecycleError> {
        let id = self.host.registrations.get(&call.name).map_or_else(
            || commitment(&("abi.host.rejected-call.v1", &call.id, &call.name)),
            |registration| {
                commitment(&(
                    "abi.host.execution.v2",
                    &registration.request.attempt_id,
                    &registration.request.principal,
                    &registration.request.scope,
                    &registration.request.capability_id,
                    &registration.request.capability_version,
                    &call.id,
                    &call.name,
                ))
            },
        );
        if self.ledger.refuse_replay(id)? {
            self.stop =
                GovernedStop::Refused(LifecycleReason::Authorization(ReasonCode::ApprovalReplayed));
            return Ok(false);
        }
        let authorized = LifecycleReason::Authorization(ReasonCode::Authorized);
        let invalid = LifecycleReason::Authorization(ReasonCode::RequestDigestMismatch);
        let denied = LifecycleReason::Authorization(ReasonCode::NoMatchingGrant);
        let cancelled = LifecycleReason::CancelledBeforeDispatch;
        if context.should_stop() || self.options.requested_deadline_ms == 0 {
            return self.refuse(id, 0, cancelled);
        }
        if call.id.is_empty()
            || call.id.len() > 128
            || !call
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
            || call.input.len() > self.options.requested_budgets.max_input_bytes as usize
        {
            return self.refuse(id, 0, invalid);
        }
        let input: serde_json::Value = match serde_json::from_str(&call.input) {
            Ok(value) => value,
            Err(_) => return self.refuse(id, 0, invalid),
        };
        self.stage(id, 0, Digest::default(), Decision::Allow, authorized)?;
        let Some(registration) = self.host.registrations.get(&call.name) else {
            return self.refuse(id, 1, denied);
        };
        let template = &registration.request;
        let request = Request::new(
            &template.attempt_id,
            id.to_string().trim_start_matches("sha256:").to_owned(),
            template.principal.clone(),
            template.scope.clone(),
            &template.capability_id,
            &template.capability_version,
            template.package_digest,
            commitment(&input),
            &template.policy_version,
            &template.guild_constitution_version,
            &template.safety_policy_version,
            template.revocation_epoch,
        )
        .map_err(|_| LifecycleError::InvalidState)?;
        self.stage(id, 1, Digest::default(), Decision::Allow, authorized)?;
        let source = self
            .host
            .sources
            .get(&request.capability_id, &request.capability_version)
            .ok_or(LifecycleError::InvalidState)?;
        self.stage(id, 2, Digest::default(), Decision::Allow, authorized)?;
        if !source.validates_input(&input) {
            return self.refuse(id, 3, invalid);
        }
        self.stage(id, 3, Digest::default(), Decision::Allow, authorized)?;
        let facts = match self.facts.observe(&request) {
            Ok(facts) if self.fresh(&facts) => facts,
            _ => return self.refuse(id, 4, LifecycleReason::StorageUnavailable),
        };
        if !source
            .source()
            .data_classes
            .is_subset(&facts.permitted_data_classes)
            || source.source().residency != facts.residency
            || source.source().retention != facts.retention
        {
            return self.refuse(id, 4, denied);
        }
        self.stage(id, 4, Digest::default(), Decision::Allow, authorized)?;
        let registry = self
            .host
            .sources
            .capability_registry()
            .map_err(|_| LifecycleError::InvalidState)?;
        let kernel = Kernel::new(&registry);
        let precondition_digest = commitment(&(
            &facts.predicates,
            facts.captured_pre_state,
            &self.options.requested_budgets,
            self.options.requested_deadline_ms,
            self.options.idempotency_key,
        ));
        let evaluated = evaluate(
            &kernel,
            &request,
            &AuthorizationContext {
                now_ms: self.options.clock,
                cancellation: context.cancellation,
                grants: &facts.grants,
                approval: None,
                used_decision_ids: &facts.used_decision_ids,
                platform: &facts.platform,
            },
            source,
            precondition_digest,
            self.options.approval.as_ref(),
            self.compensation.as_ref(),
        );
        let evaluation = evaluated.evaluation;
        if !matches!(
            evaluation.decision,
            Decision::Allow | Decision::ApprovalRequired
        ) {
            return self.refuse(id, 5, LifecycleReason::Authorization(evaluation.reason));
        }
        let binding = evaluated.binding.ok_or(LifecycleError::InvalidState)?;
        let binding_digest = commitment(&binding);
        self.stage(id, 5, binding_digest, Decision::Allow, authorized)?;
        if !facts.safety_allows {
            return self.refuse(id, 6, denied);
        }
        self.stage(id, 6, binding_digest, Decision::Allow, authorized)?;
        self.calls = self.calls.saturating_add(1);
        if !facts.rate_allows || self.calls > self.options.requested_budgets.max_calls {
            return self.refuse(id, 7, denied);
        }
        let adapter = AdapterBinding {
            adapter_id: "recording".into(),
            api_version: "1".into(),
        };
        let prerequisites = SourceFacts {
            input: &input,
            predicates: &facts.predicates,
            now_ms: self.options.clock.now_ms(),
            requested_deadline_ms: self.options.requested_deadline_ms,
            requested_budgets: &self.options.requested_budgets,
            adapter: &adapter,
            idempotency_key: self.options.idempotency_key,
            captured_pre_state: facts.captured_pre_state,
        };
        if source
            .check_prerequisites(&request, &prerequisites)
            .is_err()
        {
            return self.refuse(id, 7, invalid);
        }
        self.stage(id, 7, binding_digest, Decision::Allow, authorized)?;
        if context.should_stop() {
            return self.refuse(id, 8, cancelled);
        }
        self.stage(id, 8, binding_digest, Decision::Allow, authorized)?;
        let grant_expiry = facts
            .grants
            .iter()
            .find(|grant| Some(commitment(*grant)) == evaluation.grant_digest)
            .map_or(0, |grant| grant.expires_at_ms);
        let remaining_ms = u64::try_from(
            context
                .deadline
                .saturating_duration_since(std::time::Instant::now())
                .as_millis(),
        )
        .unwrap_or(u64::MAX);
        let expires_at_ms = self
            .options
            .clock
            .now_ms()
            .saturating_add(
                source
                    .source()
                    .expiry_policy_ms
                    .min(abi_capability::lifecycle::MAX_APPROVAL_TTL_MS)
                    .min(remaining_ms),
            )
            .min(source.source().expiration_criteria.expires_at_ms)
            .min(grant_expiry);
        let preview = ApprovalPreview {
            binding: binding.clone(),
            tool: call.name.clone(),
            call_reference: request.call_id.clone(),
            capability: source.source().id.clone(),
            version: source.source().version.clone(),
            parameter_digest: request.parameter_digest,
            effect: source.source().side_effect,
            risk: source.source().risk_class,
            reversibility: source.source().reversibility,
            postconditions: source.source().postconditions.clone(),
            compensation: source
                .source()
                .compensation
                .as_ref()
                .map(|c| CompensationPreview {
                    capability: c.capability_id.clone(),
                    version: c.version.clone(),
                    required_pre_state: c.pre_state.clone(),
                    preauthorized: c.preauthorized,
                }),
            rollback: source.source().rollback,
            expires_at_ms,
            uncertainty: PreviewUncertainty::RecordingOnly,
        };
        self.stage(
            id,
            9,
            binding_digest,
            evaluation.decision,
            LifecycleReason::Authorization(evaluation.reason),
        )?;
        if evaluation.decision == Decision::ApprovalRequired {
            self.pending_approval = Some(preview.clone());
            self.stop = GovernedStop::PendingApproval(binding);
            return Ok(false);
        }
        let refreshed = match self.facts.observe(&request) {
            Ok(value) if self.fresh(&value) => value,
            _ => return self.refuse(id, 10, LifecycleReason::StorageUnavailable),
        };
        // Fresh timestamps may advance, but every authority and pre-state fact must agree.
        let mut comparable = refreshed.clone();
        comparable.observed_at_ms = facts.observed_at_ms;
        comparable.expires_at_ms = facts.expires_at_ms;
        let reevaluated = evaluate(
            &kernel,
            &request,
            &AuthorizationContext {
                now_ms: self.options.clock,
                cancellation: context.cancellation,
                grants: &refreshed.grants,
                approval: None,
                used_decision_ids: &refreshed.used_decision_ids,
                platform: &refreshed.platform,
            },
            source,
            precondition_digest,
            self.options.approval.as_ref(),
            self.compensation.as_ref(),
        )
        .evaluation;
        let refreshed_prerequisites = SourceFacts {
            now_ms: self.options.clock.now_ms(),
            predicates: &refreshed.predicates,
            captured_pre_state: refreshed.captured_pre_state,
            ..prerequisites
        };
        if comparable != facts
            || reevaluated != evaluation
            || source
                .check_prerequisites(&request, &refreshed_prerequisites)
                .is_err()
        {
            return self.refuse(id, 10, denied);
        }
        if context.should_stop() {
            return self.refuse(id, 10, cancelled);
        }
        self.stage(id, 10, binding_digest, Decision::Allow, authorized)?;
        if context.should_stop() {
            return self.refuse_bound(id, 11, binding_digest, cancelled);
        }
        // The audit callback itself is a cooperative boundary. Refresh once more
        // after it returns so expiry/revocation cannot cross the intent write.
        let latest = match self.facts.observe(&request) {
            Ok(value) if self.fresh(&value) => value,
            _ => {
                return self.refuse_bound(
                    id,
                    11,
                    binding_digest,
                    LifecycleReason::StorageUnavailable,
                );
            }
        };
        let mut latest_comparable = latest.clone();
        latest_comparable.observed_at_ms = facts.observed_at_ms;
        latest_comparable.expires_at_ms = facts.expires_at_ms;
        let final_evaluation = evaluate(
            &kernel,
            &request,
            &AuthorizationContext {
                now_ms: self.options.clock,
                cancellation: context.cancellation,
                grants: &latest.grants,
                approval: None,
                used_decision_ids: &latest.used_decision_ids,
                platform: &latest.platform,
            },
            source,
            precondition_digest,
            self.options.approval.as_ref(),
            self.compensation.as_ref(),
        )
        .evaluation;
        if latest_comparable != facts
            || final_evaluation != evaluation
            || self.options.clock.now_ms() >= source.source().expiration_criteria.expires_at_ms
        {
            return self.refuse_bound(id, 11, binding_digest, denied);
        }
        if context.should_stop() {
            return self.refuse_bound(id, 11, binding_digest, cancelled);
        }
        let prepare = PrepareRequest {
            call_id: request.call_id.clone(),
            tool_id: call.name.clone(),
            execution_id: id,
            binding,
            approval: self.options.approval.clone(),
            now_ms: self.options.clock.now_ms(),
        };
        let prepared = match self.ledger.prepare(&prepare)? {
            PrepareOutcome::Prepared(prepared) => prepared,
            PrepareOutcome::Pending => {
                self.pending_approval = Some(preview);
                self.stop = GovernedStop::PendingApproval(prepare.binding);
                return Ok(false);
            }
            PrepareOutcome::Refused(reason) => {
                return self.refuse_bound(
                    id,
                    11,
                    binding_digest,
                    LifecycleReason::Authorization(reason),
                );
            }
            PrepareOutcome::Uncertain => {
                self.stop = GovernedStop::Uncertain;
                return Ok(false);
            }
        };
        self.executions.push(prepared.clone());
        self.stop = GovernedStop::Uncertain;
        if context.should_stop() {
            return self.decline_prepared(&prepared);
        }
        // Fallible pre-effect stage audit must succeed before recording.
        if let Err(error) = self.stage(id, 12, binding_digest, Decision::Allow, authorized) {
            self.stop = GovernedStop::Uncertain;
            return Err(error);
        }
        if context.should_stop() {
            return self.decline_prepared(&prepared);
        }
        let digest = self
            .actuator
            .record_prepared(&request, source.package(), &prepared)?;
        let observations = match self.facts.observe(&request) {
            Ok(observed) if self.fresh(&observed) => self
                .facts
                .postconditions(&request, &source.source().postconditions),
            _ => Err(LifecycleError::Unavailable),
        };
        let postconditions =
            observations
                .as_ref()
                .map_or(Postconditions::Unverifiable, |observations| {
                    crate::governed_postconditions::verify(
                        &source.source().postconditions,
                        observations,
                        request.request_digest,
                        self.options.clock.now_ms(),
                    )
                });
        let result_commitment = commitment(&(
            "abi.host.observed-result.v1",
            digest,
            observations.as_ref().ok().filter(|values| {
                values.len() <= 32 && values.iter().all(|value| value.predicate.len() <= 64)
            }),
        ));
        let (state, reason) = if context.should_stop() {
            (OutcomeState::Uncertain, LifecycleReason::OutcomeUnknown)
        } else if !source.validates_output(&serde_json::Value::Bool(true)) {
            (OutcomeState::Failed, LifecycleReason::AdapterFailure)
        } else {
            match postconditions {
                Postconditions::Verified if self.actuator.postconditions_hold(source.package()) => {
                    (OutcomeState::Succeeded, authorized)
                }
                Postconditions::Verified | Postconditions::Failed => {
                    (OutcomeState::Failed, LifecycleReason::PostconditionFailed)
                }
                Postconditions::Unverifiable => (
                    OutcomeState::Unverifiable,
                    LifecycleReason::PostconditionUnverifiable,
                ),
            }
        };
        let (state, reason) = if self.compensation.is_some() {
            match state {
                OutcomeState::Succeeded => (OutcomeState::Compensated, reason),
                OutcomeState::Failed => (
                    OutcomeState::CompensationFailed,
                    LifecycleReason::CompensationFailed,
                ),
                _ => (state, reason),
            }
        } else {
            (state, reason)
        };
        // Once an effect exists, any persistence failure retains uncertain accounting.
        self.stop = GovernedStop::Uncertain;
        self.stage(id, 13, binding_digest, Decision::Allow, reason)?;
        let outcome = TerminalOutcome {
            version: 1,
            state,
            result_digest: Some(result_commitment),
            reason,
        };
        self.ledger.finish(&prepared, &outcome)?;
        self.stage(id, 14, binding_digest, Decision::Allow, reason)?;
        // Compensation requires a separate explicitly submitted governed call; never replay here.
        self.stage(id, 15, binding_digest, Decision::Allow, reason)?;
        self.stage(id, 16, binding_digest, Decision::Allow, reason)?;
        self.outcomes.push(outcome);
        self.stop = if matches!(state, OutcomeState::Succeeded | OutcomeState::Compensated) {
            GovernedStop::Completed
        } else if state == OutcomeState::Uncertain {
            GovernedStop::Uncertain
        } else {
            GovernedStop::Refused(reason)
        };
        Ok(matches!(
            state,
            OutcomeState::Succeeded | OutcomeState::Compensated
        ))
    }
}

fn commitment<T: Serialize>(value: &T) -> Digest {
    let bytes = serde_json::to_vec(value).expect("closed host commitment");
    Digest::from_bytes(Sha256::digest(bytes).into())
}
struct Specs(Vec<ToolSpec>);
impl ToolRegistry for Specs {
    fn specs(&self) -> Vec<ToolSpec> {
        self.0.clone()
    }
    fn spec(&self, name: &str) -> Option<ToolSpec> {
        self.0.iter().find(|s| s.name == name).cloned()
    }
}
struct NoAudit;
impl abi_agent_runtime::AuditSink for NoAudit {
    fn record(&self, _: &abi_agent_runtime::AuditEntry) {}
}
struct NoExecutor;
impl crate::ToolExecutor for NoExecutor {
    fn execute(
        &self,
        _: &ToolCall,
        _: &ToolSpec,
        _: ToolExecutionContext<'_>,
    ) -> Result<crate::ToolOutput, crate::ToolExecutionError> {
        Err(crate::ToolExecutionError::new("governed_hook_required"))
    }
}
