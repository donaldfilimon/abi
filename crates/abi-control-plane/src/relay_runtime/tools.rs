//! Encrypted proposals and locally approved, single-dispatch tool execution.
use super::DurableRelayRuntime;
use crate::{Dispatch, DispatchPermit, Error, JobBinding, Result, SignedDeliveryPolicy};
use abi_agent_host::{
    GovernedFactProvider, GovernedFacts, GovernedHost, GovernedLiveExecutor, GovernedOptions,
    GovernedRegistration, GovernedRunReport, GovernedStop, HostBudget, PostconditionObservation,
    local_approval::{LocalApprovalPreview, LocalApprovalStatus, LocalApprovalStore},
};
use abi_agent_runtime::{
    CancellationToken, CollectingSink, ModelEvent, ModelProvider, ModelRequest, RunContext,
    RuntimeError, ToolCall,
};
use abi_capability::{
    ApprovalLevel, Clock, Digest, Request, lifecycle::LifecycleError, source::SourceRegistry,
};
use abi_worker::{
    Admission, JobControlStore, JobResult, JobStatus, RecordedExecution, ResultChunk,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicBool, Ordering},
};

/// Encrypted tool proposal. It cannot choose authority, adapters or executables.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolJob {
    /// Must be abi.governed-tool.v1.
    pub protocol: String,
    /// Startup-registered tool name.
    pub tool: String,
    /// Startup-selected tool-set revision.
    pub revision: String,
    /// Exact parameters presented for local approval.
    pub parameters: serde_json::Value,
}
impl ToolJob {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.protocol != "abi.governed-tool.v1"
            || self.tool.is_empty()
            || self.tool.len() > 128
            || self.revision.is_empty()
            || self.revision.len() > 128
            || serde_json::to_vec(&self.parameters)
                .map_err(|_| Error::Invalid)?
                .len()
                > 8192
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
}
/// Startup-owned local authority and adapter dependencies; never deserialized from a job.
pub struct GovernedToolContext<'a> {
    /// Trusted local user authenticated by the approval owner.
    pub user_id: &'a str,
    /// Exact human-visible resource description, selected locally.
    pub resource: &'a str,
    /// Expected effects selected by the registered adapter.
    pub expected_effects: &'a [String],
    /// Compiled, tenant-scoped capability sources.
    pub sources: &'a SourceRegistry,
    /// Startup registration; job can only select this tool.
    pub registration: GovernedRegistration,
    /// Current grants, platform permission and pre/postcondition observations.
    pub facts: &'a mut dyn GovernedFactProvider,
    /// Abbey-owned local durable approval and lifecycle authority.
    pub approvals: &'a mut dyn LocalApprovalStore,
    /// Startup-selected adapter.
    pub executor: &'a mut dyn GovernedLiveExecutor,
    /// Trusted monotonic wall-clock observation.
    pub clock: &'a dyn Clock,
    /// Fresh outbound authority inspection, invoked again immediately before effects.
    pub refresh: &'a mut dyn FnMut() -> Result<DispatchPermit>,
}
/// Pending never authorizes dispatch; unresolved never authorizes repetition.
#[derive(Debug)]
pub enum GovernedToolOutcome {
    /// Exact preview is available at the local Abbey authority.
    Pending(LocalApprovalStatus),
    /// A previously or newly durably recorded result.
    Recorded(JobResult),
    /// Effect or interrupted admission has no safely repeatable outcome.
    Unresolved,
}
struct Proposal {
    call: ToolCall,
    emitted: AtomicBool,
}
impl ModelProvider for Proposal {
    fn id(&self) -> &'static str {
        "abi-governed-tool-proposal"
    }
    fn run(
        &self,
        _: &ModelRequest,
        run: &mut RunContext<'_>,
    ) -> std::result::Result<(), RuntimeError> {
        if !self.emitted.swap(true, Ordering::SeqCst) {
            run.emit(&ModelEvent::ToolCall(self.call.clone()));
        }
        Ok(())
    }
}
struct FreshFacts<'a> {
    inner: &'a mut dyn GovernedFactProvider,
    refresh: &'a mut dyn FnMut() -> Result<DispatchPermit>,
    binding: &'a JobBinding,
    submission_digest: String,
    key: [u8; 32],
    clock: &'a dyn Clock,
}
impl GovernedFactProvider for FreshFacts<'_> {
    fn observe(&mut self, request: &Request) -> std::result::Result<GovernedFacts, LifecycleError> {
        let permit = (self.refresh)().map_err(|_| LifecycleError::Unavailable)?;
        permit
            .verify(
                &self.key,
                i64::try_from(self.clock.now_ms() / 1000)
                    .map_err(|_| LifecycleError::InvalidState)?,
            )
            .map_err(|_| LifecycleError::Unavailable)?;
        if permit.binding != *self.binding || permit.submission_digest != self.submission_digest {
            return Err(LifecycleError::InvalidState);
        }
        self.inner.observe(request)
    }
    fn postconditions(
        &mut self,
        request: &Request,
        required: &BTreeSet<String>,
    ) -> std::result::Result<Vec<PostconditionObservation>, LifecycleError> {
        self.inner.postconditions(request, required)
    }
}
fn commitment<T: Serialize>(value: &T) -> Digest {
    Digest::from_bytes(Sha256::digest(serde_json::to_vec(value).expect("closed commitment")).into())
}
impl<S: JobControlStore> DurableRelayRuntime<S> {
    /// Inspect/create an exact local approval, then dispatch only after its explicit decision.
    /// Recovery and duplicates return durable state and never call the adapter again.
    pub fn execute_governed(
        &mut self,
        dispatch: &Dispatch,
        policy: &SignedDeliveryPolicy,
        context: &mut GovernedToolContext<'_>,
        cancel: &CancellationToken,
    ) -> Result<GovernedToolOutcome> {
        if self.unresolved_live || cancel.is_cancelled() {
            return Err(Error::Denied);
        }
        let b = &dispatch.submission.binding;
        let envelope = self.runtime.envelope_for_dispatch(dispatch)?;
        if self.runtime.capability != "tool.execute"
            || envelope.request.capability_revision != self.runtime.revision
        {
            return Err(Error::Denied);
        }
        match self.runtime.recorded_dispatch(dispatch)? {
            RecordedExecution::Completed(result) => {
                self.finish_local(dispatch, time(context.clock)?)?;
                return Ok(GovernedToolOutcome::Recorded(result));
            }
            RecordedExecution::Missing => {}
            _ => return Ok(GovernedToolOutcome::Unresolved),
        }
        let (submission_digest, job, registration) =
            self.validate_tool(dispatch, policy, context)?;
        let source = context
            .sources
            .get(
                &registration.request.capability_id,
                &registration.request.capability_version,
            )
            .ok_or(Error::Denied)?;
        self.outbox.prepare(dispatch, policy)?;
        let mut facts = FreshFacts {
            inner: context.facts,
            refresh: context.refresh,
            binding: b,
            submission_digest,
            key: self.runtime.authority_key,
            clock: context.clock,
        };
        let preview = run(
            &job,
            b,
            context.sources,
            registration.clone(),
            &mut facts,
            context.approvals,
            context.executor,
            context.clock,
            None,
            cancel,
        )?;
        let Some(preview) = preview.pending_approval else {
            return Err(Error::Denied);
        };
        let created = context
            .approvals
            .read_local_approval(&preview.request.call_id)
            .map_err(|_| Error::Denied)?
            .map_or(context.clock.now_ms(), |p| p.created_at_ms);
        let local = local_preview(
            &job,
            b,
            preview,
            &mut facts,
            source,
            context.user_id,
            context.resource,
            context.expected_effects,
            created,
        )?;
        let (status, approval) = local_decision(context.approvals, &local, context.clock)?;
        let Some(approval) = approval else {
            return Ok(GovernedToolOutcome::Pending(status));
        };
        let Admission::Leased(lease) = self
            .runtime
            .manager
            .submit(&envelope, time_ms(context.clock)?)
            .map_err(|_| Error::Denied)?
        else {
            return Ok(GovernedToolOutcome::Unresolved);
        };
        self.unresolved_live = true;
        let report = run(
            &job,
            b,
            context.sources,
            registration,
            &mut facts,
            context.approvals,
            context.executor,
            context.clock,
            Some(approval),
            cancel,
        )?;
        self.finish_tool(dispatch, &report, &lease, context.clock)
    }
    fn validate_tool(
        &self,
        dispatch: &Dispatch,
        policy: &SignedDeliveryPolicy,
        context: &GovernedToolContext<'_>,
    ) -> Result<(String, ToolJob, GovernedRegistration)> {
        let b = &dispatch.submission.binding;
        if let Some(e) = self.outbox.entry(&b.job_id)?
            && (e.cancelled
                || e.abandoned
                || e.expired
                || e.manifest.is_some()
                || !e.packets.is_empty())
        {
            return Err(Error::Denied);
        }
        let submission_digest =
            crate::authority::hash(&crate::enrollment::json(&dispatch.submission)?);
        policy.verify(&self.runtime.authority_key, b, &submission_digest)?;
        self.runtime.certificate.verify(
            &self.runtime.authority_key,
            &b.audience,
            time(context.clock)?,
        )?;
        if context.clock.now_ms()
            >= u64::try_from(b.expires_at)
                .map_err(|_| Error::Denied)?
                .saturating_mul(1000)
        {
            return Err(Error::Denied);
        }
        let input = self
            .runtime
            .encryption
            .open(&b.aad("request")?, &dispatch.submission.payload)?;
        let job: ToolJob = serde_json::from_slice(&input).map_err(|_| Error::Denied)?;
        let mut registration = context.registration.clone();
        if registration.tool_name != job.tool
            || registration.request.scope.tenant_id != b.tenant_id
            || registration.request.principal.id != b.account_id
        {
            return Err(Error::Denied);
        }
        registration.request.attempt_id.clone_from(&b.job_id);
        let source = context
            .sources
            .get(
                &registration.request.capability_id,
                &registration.request.capability_version,
            )
            .ok_or(Error::Denied)?;
        if source.source().approval_policy != ApprovalLevel::A1Actor {
            return Err(Error::Denied);
        }
        Ok((submission_digest, job, registration))
    }
    fn finish_tool(
        &mut self,
        dispatch: &Dispatch,
        report: &GovernedRunReport,
        lease: &abi_worker::JobLease,
        clock: &dyn Clock,
    ) -> Result<GovernedToolOutcome> {
        if report.governance == GovernedStop::Uncertain {
            return Ok(GovernedToolOutcome::Unresolved);
        }
        let results = report
            .driver
            .tool_results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "call_id": r.call_id, "status": r.status.as_str(), "payload": r.payload.text()
                })
            })
            .collect::<Vec<_>>();
        let bytes = serde_json::to_vec(
            &serde_json::json!({"outcomes": report.outcomes, "results": results}),
        )
        .map_err(|_| Error::Invalid)?;
        let status = if report.governance == GovernedStop::Completed && report.outcomes.len() == 1 {
            JobStatus::Succeeded
        } else {
            JobStatus::Failed
        };
        let result = self
            .runtime
            .manager
            .complete(
                lease,
                status,
                vec![ResultChunk { sequence: 0, bytes }],
                time_ms(clock)?,
            )
            .map_err(|_| Error::Denied)?;
        self.finish_local(dispatch, time(clock)?)?;
        self.unresolved_live = false;
        Ok(GovernedToolOutcome::Recorded(result))
    }
}
fn local_decision(
    store: &dyn LocalApprovalStore,
    preview: &LocalApprovalPreview,
    clock: &dyn Clock,
) -> Result<(
    LocalApprovalStatus,
    Option<abi_capability::lifecycle::ExactApproval>,
)> {
    let status = store
        .request_local_approval(preview)
        .map_err(|_| Error::Denied)?;
    let approval = store
        .exact_local_approval(&preview.request.call_id, preview.digest(), clock.now_ms())
        .map_err(|_| Error::Denied)?;
    Ok((status, approval))
}
#[allow(clippy::too_many_arguments)]
fn local_preview(
    job: &ToolJob,
    b: &JobBinding,
    preview: abi_agent_host::ApprovalPreview,
    facts: &mut FreshFacts<'_>,
    source: &abi_capability::source::CompiledSource,
    user_id: &str,
    resource: &str,
    expected_effects: &[String],
    created: u64,
) -> Result<LocalApprovalPreview> {
    let observed = facts.observe(&preview.request).map_err(|_| Error::Denied)?;
    let grant = observed
        .grants
        .iter()
        .find(|g| commitment(*g) == preview.binding.grant_digest)
        .ok_or(Error::Denied)?;
    let local = LocalApprovalPreview {
        version: 1,
        tenant_id: b.tenant_id.clone(),
        account_id: b.account_id.clone(),
        node_id: b.node_id.clone(),
        user_id: user_id.into(),
        request: preview.request,
        binding: preview.binding,
        grant_id: grant.id.clone(),
        tool_id: job.tool.clone(),
        operation: source.source().id.clone(),
        resource: resource.into(),
        parameters: job.parameters.clone(),
        expected_effects: expected_effects.to_vec(),
        created_at_ms: created,
        expires_at_ms: created
            .saturating_add(900_000.min(source.source().expiry_policy_ms))
            .min(
                u64::try_from(b.expires_at)
                    .map_err(|_| Error::Denied)?
                    .saturating_mul(1000),
            )
            .min(source.source().expiration_criteria.expires_at_ms)
            .min(grant.expires_at_ms),
    };
    Ok(local)
}
#[allow(clippy::too_many_arguments)]
fn run(
    job: &ToolJob,
    binding: &JobBinding,
    sources: &SourceRegistry,
    registration: GovernedRegistration,
    facts: &mut dyn GovernedFactProvider,
    ledger: &mut dyn LocalApprovalStore,
    executor: &mut dyn GovernedLiveExecutor,
    clock: &dyn Clock,
    approval: Option<abi_capability::lifecycle::ExactApproval>,
    cancel: &CancellationToken,
) -> Result<GovernedRunReport> {
    let provider = Proposal {
        call: ToolCall::new(
            &binding.job_id,
            &job.tool,
            serde_json::to_string(&job.parameters).map_err(|_| Error::Invalid)?,
        ),
        emitted: AtomicBool::new(false),
    };
    let source = sources
        .get(
            &registration.request.capability_id,
            &registration.request.capability_version,
        )
        .ok_or(Error::Denied)?;
    let deadline = u64::try_from(binding.expires_at)
        .map_err(|_| Error::Denied)?
        .saturating_mul(1000)
        .saturating_sub(clock.now_ms())
        .min(source.source().deadline_ms);
    let host =
        GovernedHost::new(&provider, sources, vec![registration]).map_err(|_| Error::Denied)?;
    let budget = HostBudget {
        max_tool_calls: 1,
        max_tool_rounds: 1,
        max_provider_runs: 2,
        max_tool_result_bytes: binding
            .max_result_bytes
            .saturating_sub(4096)
            .min(16_384)
            .saturating_sub(4096)
            / 6,
        max_duration: std::time::Duration::from_millis(deadline),
        ..HostBudget::default()
    };
    host.run_live(
        &ModelRequest::new("governed-tool-v1").with_max_output_tokens(128),
        &mut CollectingSink::new(),
        cancel,
        GovernedOptions {
            clock,
            host_budget: budget,
            approval,
            idempotency_key: Some(commitment(binding)),
            requested_deadline_ms: source.source().deadline_ms,
            requested_budgets: source.source().budgets.clone(),
        },
        facts,
        ledger,
        executor,
    )
    .map_err(|_| Error::Denied)
}
fn time(clock: &dyn Clock) -> Result<i64> {
    i64::try_from(clock.now_ms() / 1000).map_err(|_| Error::Denied)
}
fn time_ms(clock: &dyn Clock) -> Result<i64> {
    i64::try_from(clock.now_ms()).map_err(|_| Error::Denied)
}
