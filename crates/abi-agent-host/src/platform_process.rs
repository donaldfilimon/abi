//! Closed, startup-pinned working-directory process capability.
//!
//! This is a bounded Unix observation adapter, not a shell or an OS sandbox.
//! Durable approval and replay protection remain the governed host's authority.
use crate::{
    GovernedFactProvider, GovernedFacts, GovernedLiveExecutor, ObservationSource,
    PostconditionObservation, Postconditions, ToolExecutionContext, ToolExecutionError, ToolOutput,
};
use abi_agent_runtime::ToolCall;
use abi_capability::{
    ApprovalLevel, Clock, Digest, Request, SideEffect,
    lifecycle::{LifecycleError, PreparedExecution},
    source::{
        AdapterBinding, CancellationContract, CompiledSource, PackageSource, ResidencyPolicy,
    },
};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Arc, Mutex},
};

#[cfg(unix)]
mod unix;
#[cfg(unix)]
use unix as os;
#[cfg(not(unix))]
mod unsupported;
#[cfg(not(unix))]
use unsupported as os;

const EXEC_ACCESS: &str = "process.pinned_exec_access";
const CWD_ACCESS: &str = "process.pinned_cwd_search";
const READY: &str = "process.ready";
const SCOPED: &str = "process.scoped";
const EXITED: &str = "process.exited_zero";
const MATCHES: &str = "process.cwd_matches";

/// Finite execution, aggregate output and cleanup budgets.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ProcessLimits {
    /// Process timeout in milliseconds (1..=30,000).
    pub timeout_ms: u64,
    /// Combined stdout/stderr byte ceiling (1..=65,536).
    pub max_output_bytes: usize,
    /// Cleanup grace in milliseconds (1..=1,000).
    pub cleanup_grace_ms: u64,
}
impl Default for ProcessLimits {
    fn default() -> Self {
        Self {
            timeout_ms: 1000,
            max_output_bytes: 4096,
            cleanup_grace_ms: 250,
        }
    }
}
impl ProcessLimits {
    fn valid(self) -> bool {
        (1..=30_000).contains(&self.timeout_ms)
            && (1..=65_536).contains(&self.max_output_bytes)
            && (1..=1000).contains(&self.cleanup_grace_ms)
    }
}

/// Startup-selected closed process plan. Executable and arguments are not public inputs.
pub struct ProcessPlan {
    pinned: os::Pinned,
    limits: ProcessLimits,
    digest: Digest,
}
/// Sealed process executor; construction occurs only by binding a plan.
pub struct GovernedProcessExecutor {
    shared: Arc<Shared>,
}
/// Opaque handle to this executor's actual observations.
pub struct ProcessObservationHandle {
    shared: Arc<Shared>,
}
/// Decorates existing authority with independent process observations.
pub struct GovernedProcessFacts<F> {
    upstream: F,
    shared: Arc<Shared>,
}
struct Shared {
    plan: ProcessPlan,
    source: PackageSource,
    package: Digest,
    schema: Digest,
    clock: Arc<dyn Clock>,
    state: Mutex<State>,
}
#[derive(Default)]
struct State {
    runner: os::Runner,
    receipt: Option<Receipt>,
}
struct Receipt {
    request: Digest,
    evidence: Digest,
    completed_at: u64,
}

impl ProcessPlan {
    /// Pin a working directory and the reviewed system pwd operation.
    pub fn working_directory(
        path: &Path,
        limits: ProcessLimits,
    ) -> Result<Self, ToolExecutionError> {
        if !limits.valid() {
            return Err(error("process_invalid_limits"));
        }
        let pinned = os::Pinned::new(path)?;
        let digest = commitment(&(
            "abi.process.cwd-plan.v1",
            pinned.identity_digest(),
            limits,
            "pwd",
            "empty-environment",
            [READY, SCOPED],
            [EXITED, MATCHES],
        ));
        Ok(Self {
            pinned,
            limits,
            digest,
        })
    }
    /// Exact versioned resource commitment.
    #[must_use]
    pub const fn resource_digest(&self) -> Digest {
        self.digest
    }
    /// Seal this plan against one exact compiled capability and trusted clock.
    pub fn bind(
        self,
        compiled: &CompiledSource,
        clock: Arc<dyn Clock>,
    ) -> Result<(GovernedProcessExecutor, ProcessObservationHandle), ToolExecutionError> {
        let src = compiled.source();
        let output = output(&self.pinned.cwd);
        if src.id != "local.process.cwd"
            || src.version != "1.0.0"
            || src.resource_scope.resource_digest != self.digest
            || src.adapter_bindings != [adapter()]
            || src.input_schema != json!({"type":"object","additionalProperties":false})
            || src.output_schema
                != json!({"type":"object","properties":{"cwd":{"type":"string"},"exit_code":{"const":0}},"required":["cwd","exit_code"],"additionalProperties":false})
            || src.confirmation_policy != ApprovalLevel::A1Actor
            || src.approval_policy != ApprovalLevel::A1Actor
            || src.side_effect != SideEffect::PlatformRead
            || src.cancellation != CancellationContract::Cooperative
            || src.residency != ResidencyPolicy::LocalOnly
            || !src.required_platform_permissions.contains(EXEC_ACCESS)
            || !src.required_platform_permissions.contains(CWD_ACCESS)
            || !src.preconditions.contains(READY)
            || !src.invariants.contains(SCOPED)
            || src.postconditions != owned_postconditions()
            || !src.receipt_requirements.require_postcondition_evidence
            || !src.receipt_requirements.redacted
            || src.budgets.max_input_bytes < 2
            || output.len() > src.budgets.max_output_bytes as usize
            || self.pinned.cwd.len() + 1 > self.limits.max_output_bytes
            || self.limits.timeout_ms > src.deadline_ms
        {
            return Err(error("process_source_mismatch"));
        }
        self.pinned.verify()?;
        let shared = Arc::new(Shared {
            plan: self,
            source: src.clone(),
            package: compiled.package().digest,
            schema: compiled.schema_digest(),
            clock,
            state: Mutex::new(State::default()),
        });
        Ok((
            GovernedProcessExecutor {
                shared: Arc::clone(&shared),
            },
            ProcessObservationHandle { shared },
        ))
    }
}
impl ProcessObservationHandle {
    /// Retain upstream authority while independently observing OS state.
    pub fn decorate<F: GovernedFactProvider>(self, upstream: F) -> GovernedProcessFacts<F> {
        GovernedProcessFacts {
            upstream,
            shared: self.shared,
        }
    }
}
impl Shared {
    fn matches(&self, request: &Request) -> bool {
        request.computed_digest() == request.request_digest
            && request.package_digest == self.package
            && request.scope == self.source.resource_scope
            && request.capability_id == self.source.id
            && request.capability_version == self.source.version
            && request.parameter_digest == commitment(&json!({}))
    }
}
impl<F: GovernedFactProvider> GovernedFactProvider for GovernedProcessFacts<F> {
    fn observe(&mut self, request: &Request) -> Result<GovernedFacts, LifecycleError> {
        if !self.shared.matches(request) {
            return Err(LifecycleError::InvalidState);
        }
        let mut facts = self.upstream.observe(request)?;
        let now = self.shared.clock.now_ms();
        if facts.observed_at_ms > now
            || now >= facts.expires_at_ms
            || now >= self.shared.source.expiration_criteria.expires_at_ms
        {
            return Err(LifecycleError::Unavailable);
        }
        // Never inherit an upstream assertion about these owned OS facts.
        facts.platform.permissions.remove(EXEC_ACCESS);
        facts.platform.permissions.remove(CWD_ACCESS);
        facts.predicates.remove(READY);
        facts.predicates.remove(SCOPED);
        self.shared
            .plan
            .pinned
            .verify()
            .map_err(|_| LifecycleError::Unavailable)?;
        let observed = self.shared.clock.now_ms();
        if observed < now || observed >= facts.expires_at_ms {
            return Err(LifecycleError::Unavailable);
        }
        facts.platform.permissions.insert(EXEC_ACCESS.into());
        facts.platform.permissions.insert(CWD_ACCESS.into());
        facts.predicates.insert(READY.into());
        facts.predicates.insert(SCOPED.into());
        facts.captured_pre_state = Some(commitment(&(
            "abi.process.pre-state.v1",
            self.shared.plan.digest,
            facts.captured_pre_state,
        )));
        Ok(facts)
    }
    fn postconditions(
        &mut self,
        request: &Request,
        required: &BTreeSet<String>,
    ) -> Result<Vec<PostconditionObservation>, LifecycleError> {
        let mut state = self
            .shared
            .state
            .try_lock()
            .map_err(|_| LifecycleError::Unavailable)?;
        // Every attempt consumes the old receipt, including malformed requests.
        let receipt = state.receipt.take().ok_or(LifecycleError::Unavailable)?;
        let now = self.shared.clock.now_ms();
        let expires = receipt
            .completed_at
            .checked_add(1000)
            .ok_or(LifecycleError::Unavailable)?;
        if !self.shared.matches(request)
            || receipt.request != request.request_digest
            || *required != owned_postconditions()
            || receipt.completed_at > now
            || now >= expires
        {
            return Err(LifecycleError::Unavailable);
        }
        self.shared
            .plan
            .pinned
            .verify()
            .map_err(|_| LifecycleError::Unavailable)?;
        let verified_at = self.shared.clock.now_ms();
        if verified_at < now || verified_at >= expires {
            return Err(LifecycleError::Unavailable);
        }
        Ok(required
            .iter()
            .map(|predicate| PostconditionObservation {
                predicate: predicate.clone(),
                request_digest: receipt.request,
                observed_at_ms: receipt.completed_at,
                expires_at_ms: expires,
                evidence_digest: Some(receipt.evidence),
                source: ObservationSource::TrustedLocalAdapter,
                status: Postconditions::Verified,
            })
            .collect())
    }
}
impl GovernedLiveExecutor for GovernedProcessExecutor {
    fn binding(&self) -> AdapterBinding {
        adapter()
    }
    fn execute(
        &mut self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError> {
        let mut state = self
            .shared
            .state
            .try_lock()
            .map_err(|_| error("process_busy"))?;
        state.receipt = None;
        let id = commitment(&(
            "abi.host.execution.v2",
            &request.attempt_id,
            &request.principal,
            &request.scope,
            &request.capability_id,
            &request.capability_version,
            &call.id,
            &call.name,
        ));
        let binding = prepared.binding();
        if !self.shared.matches(request)
            || source.source() != &self.shared.source
            || source.package().digest != self.shared.package
            || source.schema_digest() != self.shared.schema
            || call.name != "process_cwd"
            || call.id.is_empty()
            || call.id.len() > 128
            || !call
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
            || call.input.len() > self.shared.source.budgets.max_input_bytes as usize
            || serde_json::from_str::<serde_json::Value>(&call.input).ok() != Some(json!({}))
            || prepared.execution_id() != id
            || request.call_id != id.to_string().trim_start_matches("sha256:")
            || binding.request_digest != request.request_digest
            || binding.package_digest != self.shared.package
            || binding.schema_digest != self.shared.schema
            || binding.principal_digest != commitment(&request.principal)
            || binding.scope_digest != commitment(&request.scope)
            || binding.compensation_origin.is_some()
            || binding.required_approval != ApprovalLevel::A1Actor
            || binding.policy_digest
                != commitment(&(
                    "abi.policy.v1",
                    &request.policy_version,
                    &request.guild_constitution_version,
                    &request.safety_policy_version,
                    request.revocation_epoch,
                ))
            || context.should_stop()
        {
            return Err(error("process_binding_refused"));
        }
        self.shared.plan.pinned.verify()?;
        let started = self.shared.clock.now_ms();
        if started >= self.shared.source.expiration_criteria.expires_at_ms || context.should_stop()
        {
            return Err(error("process_stopped"));
        }
        let captured =
            state
                .runner
                .run(&self.shared.plan.pinned, self.shared.plan.limits, context)?;
        let completed_at = self.shared.clock.now_ms();
        if completed_at < started
            || completed_at.checked_add(1000).is_none()
            || context.should_stop()
            || captured != format!("{}\n", self.shared.plan.pinned.cwd).as_bytes()
        {
            return Err(error("process_output_unverified"));
        }
        self.shared.plan.pinned.verify()?;
        let payload = output(&self.shared.plan.pinned.cwd);
        if payload.len() > self.shared.source.budgets.max_output_bytes as usize {
            return Err(error("process_output_limit"));
        }
        let verified_at = self.shared.clock.now_ms();
        if context.should_stop()
            || verified_at < completed_at
            || verified_at >= completed_at + 1000
            || verified_at >= self.shared.source.expiration_criteria.expires_at_ms
        {
            return Err(error("process_final_observation_stopped"));
        }
        state.receipt = Some(Receipt {
            request: request.request_digest,
            completed_at,
            evidence: commitment(&(
                "abi.process.receipt.v1",
                request.request_digest,
                prepared.execution_id(),
                self.shared.plan.digest,
                &captured,
                "reaped-zero-eof-group-cleanup",
                completed_at,
            )),
        });
        Ok(ToolOutput::ok(payload))
    }
}
fn adapter() -> AdapterBinding {
    AdapterBinding {
        adapter_id: "abi.pinned-process".into(),
        api_version: "1".into(),
    }
}
fn output(cwd: &str) -> String {
    json!({"cwd":cwd,"exit_code":0}).to_string()
}
fn owned_postconditions() -> BTreeSet<String> {
    BTreeSet::from([EXITED.into(), MATCHES.into()])
}
fn error(code: &'static str) -> ToolExecutionError {
    ToolExecutionError::new(code)
}
fn commitment<T: Serialize>(value: &T) -> Digest {
    Digest::from_bytes(
        Sha256::digest(serde_json::to_vec(value).expect("closed process commitment")).into(),
    )
}

#[cfg(all(test, unix))]
#[path = "platform_process/final_verification_tests.rs"]
mod final_verification_tests;
