//! Normal and compensation admission use the same canonical kernel.
use abi_capability::{
    AuthorizationContext, CompensationEvaluation, Digest, Kernel, Request,
    lifecycle::{AuthorizationBinding, ExactApproval, PreparedExecution},
    source::CompiledSource,
};

pub(crate) struct CompensationContext<'a> {
    pub(crate) forward: &'a CompiledSource,
    pub(crate) original: &'a PreparedExecution,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate(
    kernel: &Kernel<'_>,
    request: &Request,
    context: &AuthorizationContext<'_>,
    source: &CompiledSource,
    precondition: Digest,
    approval: Option<&ExactApproval>,
    compensation: Option<&CompensationContext<'_>>,
) -> CompensationEvaluation {
    if let Some(compensation) = compensation {
        kernel.evaluate_compensation_source_bound(
            request,
            context,
            source,
            compensation.forward,
            compensation.original,
            precondition,
            approval,
        )
    } else {
        let evaluation =
            kernel.evaluate_source_bound(request, context, source, precondition, approval);
        let binding = AuthorizationBinding::from_evaluation(
            request,
            &evaluation,
            source.schema_digest(),
            precondition,
        );
        CompensationEvaluation {
            evaluation,
            binding,
        }
    }
}
