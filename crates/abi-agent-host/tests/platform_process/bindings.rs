//! Changed requests, source, preparation and tool data never create OS receipts.
use super::*;
fn change_prepared(prepared: &PreparedExecution, field: &str) -> PreparedExecution {
    let mut binding = prepared.binding().clone();
    let zero = Digest::default();
    match field {
        "prepared_request" => binding.request_digest = zero,
        "prepared_package" => binding.package_digest = zero,
        "prepared_schema" => binding.schema_digest = zero,
        "prepared_principal" => binding.principal_digest = zero,
        "prepared_scope" => binding.scope_digest = zero,
        "prepared_policy" => binding.policy_digest = zero,
        "prepared_approval" => binding.required_approval = ApprovalLevel::A0None,
        "prepared_origin" => {
            binding.compensation_origin =
                Some(abi_capability::lifecycle::CompensationBinding::from_prepared(prepared));
        }
        "execution" => return PreparedExecution::after_commit(zero, binding),
        _ => unreachable!(),
    }
    PreparedExecution::after_commit(prepared.execution_id(), binding)
}
#[test]
fn execute_refuses_each_changed_adapter_owned_binding_without_a_receipt() {
    let scratch = Scratch::new();
    let plan = ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).unwrap();
    let src = source(plan.resource_digest());
    let registry = SourceRegistry::compile([src.clone()]).unwrap();
    let compiled = registry.get("local.process.cwd", "1.0.0").unwrap();
    let (mut executor, handle) = plan.bind(compiled, Arc::new(FixedClock::new(100))).unwrap();
    let mut facts = handle.decorate(Upstream { expires: 500 });
    for mutation in [
        "scope",
        "principal",
        "package",
        "parameter",
        "request_digest",
        "source",
        "tool",
        "input",
        "call",
        "prepared_request",
        "prepared_package",
        "prepared_schema",
        "prepared_principal",
        "prepared_scope",
        "prepared_policy",
        "prepared_approval",
        "prepared_origin",
        "execution",
        "deadline",
    ] {
        let (mut request, mut prepared, mut call) = execution(compiled);
        let mut changed_source = src.clone();
        match mutation {
            "scope" => request.scope.tenant_id = "other".into(),
            "principal" => request.principal.id = "other".into(),
            "package" => request.package_digest = Digest::default(),
            "parameter" => request.parameter_digest = hash(&json!({"changed":true})),
            "request_digest" => request.request_digest = Digest::default(),
            "source" => changed_source.deadline_ms += 1,
            "tool" => call.name = "other".into(),
            "input" => call.input = "{\"changed\":true}".into(),
            "call" => call.id = "other".into(),
            "deadline" => {}
            field => prepared = change_prepared(&prepared, field),
        }
        if mutation != "request_digest" {
            request.request_digest = request.computed_digest();
        }
        let alternate = SourceRegistry::compile([changed_source]).unwrap();
        let token = CancellationToken::new();
        let deadline = if mutation == "deadline" {
            Instant::now()
        } else {
            Instant::now() + Duration::from_secs(1)
        };
        assert!(
            executor
                .execute(
                    &call,
                    &request,
                    alternate.get("local.process.cwd", "1.0.0").unwrap(),
                    &prepared,
                    ToolExecutionContext {
                        cancellation: &token,
                        deadline
                    }
                )
                .is_err(),
            "{mutation}"
        );
        assert!(
            facts.postconditions(&request, &required()).is_err(),
            "{mutation} must publish no receipt"
        );
    }
}
