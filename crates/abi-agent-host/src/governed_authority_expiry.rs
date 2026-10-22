//! Time-only recheck of already admitted authority, not a second admission.
use super::{GovernedFacts, commitment};
use abi_capability::{
    lifecycle::{ExactApproval, PreparedExecution},
    source::CompiledSource,
};

pub(super) fn valid_at(
    now_ms: u64,
    latest: &GovernedFacts,
    source: &CompiledSource,
    prepared: &PreparedExecution,
    approval: Option<&ExactApproval>,
) -> bool {
    latest.observed_at_ms <= now_ms
        && now_ms < latest.expires_at_ms
        && now_ms < source.source().expiration_criteria.expires_at_ms
        && latest
            .grants
            .iter()
            .find(|grant| commitment(*grant) == prepared.binding().grant_digest)
            .is_some_and(|grant| grant.not_before_ms <= now_ms && now_ms < grant.expires_at_ms)
        // The immutable approved envelope remains bound to this same intent.
        // Its durable consumption is expected; never rerun replay admission here.
        && approval.is_none_or(|approval| approval.matches(prepared.binding(), now_ms))
}
