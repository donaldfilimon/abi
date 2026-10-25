//! Explicit startup-owned live adapter and final authority checks.
use super::{
    AdapterBinding, AuthorizationContext, Decision, Digest, Governance, GovernedFacts, Kernel,
    LifecycleError, PreparedExecution, PreviewUncertainty, RecordingActuator, Request, SourceFacts,
    ToolCall, ToolExecutionContext, authority_expiry, commitment, evaluate,
};
use crate::{ToolExecutionError, ToolOutput};
use abi_capability::source::CompiledSource;

/// Trusted startup implementation, bound to one declared adapter identity.
/// It receives the canonical compiled source and committed execution intent;
/// implementations must never retry a call after an ambiguous outcome.
pub trait GovernedLiveExecutor {
    /// Exact adapter identity declared in the compiled source.
    fn binding(&self) -> AdapterBinding;
    /// Perform one validated call. Every error is conservatively uncertain.
    fn execute(
        &mut self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError>;
}

pub(super) enum Dispatch<'a> {
    Recording(&'a mut RecordingActuator),
    Live(&'a mut dyn GovernedLiveExecutor),
}
impl Dispatch<'_> {
    pub(super) fn reborrow(&mut self) -> Dispatch<'_> {
        match self {
            Self::Recording(a) => Dispatch::Recording(a),
            Self::Live(a) => Dispatch::Live(*a),
        }
    }

    pub(super) fn is_live(&self) -> bool {
        matches!(self, Self::Live(_))
    }
    pub(super) fn binding(&self) -> AdapterBinding {
        match self {
            Self::Recording(_) => AdapterBinding {
                adapter_id: "recording".into(),
                api_version: "1".into(),
            },
            Self::Live(executor) => executor.binding(),
        }
    }
    pub(super) fn uncertainty(&self) -> PreviewUncertainty {
        if self.is_live() {
            PreviewUncertainty::LiveAdapter
        } else {
            PreviewUncertainty::RecordingOnly
        }
    }
    pub(super) fn execute(
        &mut self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
        context: ToolExecutionContext<'_>,
    ) -> Result<(Digest, Option<ToolOutput>, bool), LifecycleError> {
        match self {
            Self::Recording(actuator) => Ok((
                actuator.record_prepared(request, source.package(), prepared)?,
                None,
                actuator.postconditions_hold(source.package()),
            )),
            Self::Live(executor) => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    executor.execute(call, request, source, prepared, context)
                }));
                let output = result
                    .map_err(|_| LifecycleError::Unavailable)?
                    .map_err(|_| LifecycleError::Unavailable)?;
                Ok((
                    commitment(&("abi.host.live-result.v1", output.payload())),
                    Some(output),
                    true,
                ))
            }
        }
    }
}

impl Governance<'_, '_> {
    /// Durable prepare consumes an exact approval. Ignore only that expected
    /// consumption, retaining every other authority change and replay marker.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn live_authority_valid(
        &mut self,
        request: &Request,
        source: &CompiledSource,
        admitted: &GovernedFacts,
        prepared: &PreparedExecution,
        kernel: &Kernel<'_>,
        input: &serde_json::Value,
        adapter: &AdapterBinding,
        context: ToolExecutionContext<'_>,
    ) -> bool {
        let Ok(mut current) = self.facts.observe(request) else {
            return false;
        };
        if context.should_stop() || !self.fresh(&current) || self.dispatch.binding() != *adapter {
            return false;
        }
        if let Some(approval) = self.options.approval.as_ref()
            && approval.matches(prepared.binding(), self.options.clock.now_ms())
            && !admitted
                .used_decision_ids
                .contains(&approval.approval.decision_id)
        {
            current
                .used_decision_ids
                .remove(&approval.approval.decision_id);
        }
        let evaluation = evaluate(
            kernel,
            request,
            &AuthorizationContext {
                now_ms: self.options.clock,
                cancellation: context.cancellation,
                grants: &current.grants,
                approval: None,
                used_decision_ids: &current.used_decision_ids,
                platform: &current.platform,
            },
            source,
            prepared.binding().precondition_digest,
            self.options.approval.as_ref(),
            self.compensation.as_ref(),
        );
        let prerequisites = SourceFacts {
            input,
            predicates: &current.predicates,
            now_ms: self.options.clock.now_ms(),
            requested_deadline_ms: self.options.requested_deadline_ms,
            requested_budgets: &self.options.requested_budgets,
            adapter,
            idempotency_key: self.options.idempotency_key,
            captured_pre_state: current.captured_pre_state,
        };
        let valid = evaluation.evaluation.decision == Decision::Allow
            && evaluation.binding.as_ref() == Some(prepared.binding())
            && source.check_prerequisites(request, &prerequisites).is_ok()
            && authority_expiry::valid_at(
                self.options.clock.now_ms(),
                &current,
                source,
                prepared,
                self.options.approval.as_ref(),
            );
        current.observed_at_ms = admitted.observed_at_ms;
        current.expires_at_ms = admitted.expires_at_ms;
        valid && current == *admitted
    }
}
