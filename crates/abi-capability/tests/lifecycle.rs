//! Internal lifecycle state and bounded transaction-input regressions.
use abi_capability::lifecycle::{
    AuthorizationBinding, LifecycleReason, OutcomeState, PrepareRequest, TerminalOutcome,
};
use abi_capability::{ApprovalLevel, Digest};
fn binding() -> AuthorizationBinding {
    let d = Digest::from_bytes([1; 32]);
    AuthorizationBinding {
        request_digest: d,
        principal_digest: d,
        scope_digest: d,
        package_digest: d,
        grant_digest: d,
        policy_digest: d,
        schema_digest: d,
        precondition_digest: d,
        required_approval: ApprovalLevel::A0None,
    }
}
#[test]
fn a0_needs_no_fabricated_approval_but_higher_floors_do() {
    let mut request = PrepareRequest {
        call_id: "call".into(),
        tool_id: "local.observe".into(),
        execution_id: Digest::from_bytes([2; 32]),
        binding: binding(),
        approval: None,
        now_ms: 100,
    };
    assert!(request.is_valid());
    request.binding.required_approval = ApprovalLevel::A1Actor;
    assert!(!request.is_valid());
    request.binding.required_approval = ApprovalLevel::A0None;
    request.call_id = "canary secret\n".into();
    assert!(!request.is_valid());
    request.call_id = "x".repeat(129);
    assert!(!request.is_valid());
}
#[test]
fn uncertainty_and_unverifiable_are_explicit_versioned_redacted_states() {
    for (state, reason) in [
        (OutcomeState::Uncertain, LifecycleReason::Interrupted),
        (OutcomeState::Uncertain, LifecycleReason::OutcomeUnknown),
        (
            OutcomeState::Unverifiable,
            LifecycleReason::PostconditionUnverifiable,
        ),
    ] {
        let outcome = TerminalOutcome {
            version: 1,
            state,
            result_digest: None,
            reason,
        };
        let bytes = serde_json::to_vec(&outcome).unwrap();
        assert_eq!(
            serde_json::from_slice::<TerminalOutcome>(&bytes).unwrap(),
            outcome
        );
        assert!(!String::from_utf8(bytes).unwrap().contains("succeeded"));
    }
    assert!(serde_json::from_str::<TerminalOutcome>(r#"{"version":1,"state":"uncertain","result_digest":null,"reason":"outcome_unknown","raw":"secret"}"#).is_err());
}
