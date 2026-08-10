//! Public contract tests for explicit local-model loading and execution.

mod support;

use abi_agent_runtime::{
    BudgetLimit, CancellationToken, CollectingSink, ModelRequest, RunBudget, RuntimeError,
    StopReason, ToolSpec, run_provider,
};
use abi_compute::Backend;
use abi_model_runtime::{
    DevicePreference, ExecutionPath, LoadConfig, LocalModelProvider, ModelRuntimeError,
};
use support::{MODEL_ID, PRINCIPAL, fixture, fixture_with_architecture};

#[test]
fn cpu_load_report_distinguishes_initialization_from_execution() {
    let fixture = fixture();
    let provider = fixture.load_cpu();
    let report = provider.load_report();

    assert_eq!(report.model_id, MODEL_ID);
    assert_eq!(report.revision, support::REVISION);
    assert_eq!(report.requested_device, DevicePreference::Cpu);
    assert_eq!(report.initialized_path, ExecutionPath::Cpu);
    assert_eq!(report.capability.backend(), Backend::CpuScalar);
    assert!(report.capability.initialized());
    assert!(!report.capability.executed());
    assert!(!report.capability.runtime_verified());
    assert_eq!(report.tensor_count, 1);
    assert_eq!(report.parameter_count, 25);
    assert!(report.artifact_bytes > 0);
}

#[test]
fn cpu_provider_executes_the_generated_scratch_model_deterministically() {
    let fixture = fixture();
    let provider = fixture.load_cpu();
    let request = ModelRequest::new(MODEL_ID).with_user("hello");
    let mut sink = CollectingSink::new();
    let report = run_provider(
        &provider,
        &request,
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect("fixture inference");

    assert_eq!(report.stop, StopReason::Completed);
    assert_eq!(report.text(), "world again");
    assert_eq!(report.usage.input_tokens, 1);
    assert_eq!(report.usage.output_tokens, 2);
    assert_eq!(
        sink.kinds(),
        vec!["started", "text_delta", "text_delta", "finished"]
    );

    let evidence = provider
        .last_inference_report()
        .expect("report state")
        .expect("successful inference report");
    assert_eq!(evidence.executed_path, ExecutionPath::Cpu);
    assert!(evidence.capability.executed());
    assert!(evidence.capability.runtime_verified());
    assert_eq!(evidence.generated_tokens, 2);
    assert_eq!(evidence.native_operations, 9);
    assert!(!evidence.fallback_used);
    assert!(!evidence.mixed_execution);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_provider_reports_real_initialized_and_executed_evidence() {
    let fixture = fixture();
    let provider = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        PRINCIPAL,
        LoadConfig::new(DevicePreference::Metal),
    )
    .expect("Metal fixture loads on the exercised Mac");
    assert_eq!(
        provider.load_report().initialized_path,
        ExecutionPath::Metal
    );
    assert!(!provider.load_report().capability.executed());

    let mut sink = CollectingSink::new();
    let report = run_provider(
        &provider,
        &ModelRequest::new(MODEL_ID).with_user("hello"),
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect("Metal inference");
    assert_eq!(report.text(), "world again");
    let evidence = provider
        .last_inference_report()
        .expect("report state")
        .expect("Metal report");
    assert_eq!(evidence.executed_path, ExecutionPath::Metal);
    assert_eq!(evidence.capability.backend(), Backend::GpuMetal);
    assert!(evidence.capability.executed());
    assert!(evidence.capability.runtime_verified());
    assert!(!evidence.fallback_used);
    assert!(!evidence.mixed_execution);
}

#[test]
fn zero_generation_does_not_claim_native_execution() {
    let fixture = fixture();
    let provider = fixture.load_cpu();
    let request = ModelRequest::new(MODEL_ID)
        .with_user("hello")
        .with_max_output_tokens(0);
    let mut sink = CollectingSink::new();
    let report = run_provider(
        &provider,
        &request,
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect("zero-token inference");
    assert_eq!(report.text(), "");

    let evidence = provider
        .last_inference_report()
        .expect("report state")
        .expect("inference report");
    assert_eq!(evidence.native_operations, 0);
    assert!(!evidence.capability.executed());
    assert!(!evidence.capability.runtime_verified());
}

#[test]
fn exact_model_selection_forbids_substitution() {
    let fixture = fixture();
    let provider = fixture.load_cpu();
    let request = ModelRequest::new("another-model").with_user("hello");
    let mut sink = CollectingSink::new();
    let error = run_provider(
        &provider,
        &request,
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect_err("model substitution must fail");
    assert!(matches!(error, RuntimeError::Provider { .. }));
    assert!(error.to_string().contains("substitution is forbidden"));
    assert_eq!(sink.kinds(), vec!["finished"]);
    assert!(
        provider
            .last_inference_report()
            .expect("report state")
            .is_none()
    );
}

#[test]
fn empty_requests_and_tools_are_explicitly_unsupported() {
    let fixture = fixture();
    let provider = fixture.load_cpu();
    let mut sink = CollectingSink::new();
    let empty = run_provider(
        &provider,
        &ModelRequest::new(MODEL_ID),
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect_err("empty conversation must fail");
    assert!(empty.to_string().contains("empty conversation"));

    let mut sink = CollectingSink::new();
    let tools = ModelRequest::new(MODEL_ID)
        .with_user("hello")
        .with_tool(ToolSpec::new("fixture.read"));
    let error = run_provider(
        &provider,
        &tools,
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect_err("fixture model has no tool architecture");
    assert!(error.to_string().contains("does not support offered tools"));
}

#[test]
fn license_acceptance_is_bound_to_the_exact_principal() {
    let fixture = fixture();
    let error = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        "different@example.invalid",
        LoadConfig::new(DevicePreference::Cpu),
    )
    .expect_err("another principal did not accept the license");
    assert!(matches!(error, ModelRuntimeError::Model(_)));
}

#[test]
fn every_artifact_is_hash_verified_before_loading() {
    let fixture = fixture();
    std::fs::write(fixture.artifact_path("tokenizer.json"), b"tampered").expect("tamper tokenizer");
    let error = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        PRINCIPAL,
        LoadConfig::new(DevicePreference::Cpu),
    )
    .expect_err("hash mismatch must fail before parsing");
    assert!(matches!(error, ModelRuntimeError::Model(_)));
    assert!(error.to_string().contains("tokenizer.json"));
}

#[test]
fn load_policy_bounds_declared_model_bytes() {
    let fixture = fixture();
    let error = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        PRINCIPAL,
        LoadConfig::new(DevicePreference::Cpu).with_max_model_bytes(1),
    )
    .expect_err("model exceeds explicit load bound");
    assert!(matches!(error, ModelRuntimeError::ModelTooLarge { .. }));
}

#[test]
fn unsupported_architecture_is_not_silently_interpreted() {
    let fixture = fixture_with_architecture("gemma4");
    let error = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        PRINCIPAL,
        LoadConfig::new(DevicePreference::Cpu),
    )
    .expect_err("Gemma is not implemented by the fixture loader");
    assert!(matches!(
        error,
        ModelRuntimeError::UnsupportedArchitecture { .. }
    ));
}

#[test]
fn context_and_generation_limits_are_enforced() {
    let fixture = fixture();
    let provider = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        PRINCIPAL,
        LoadConfig::new(DevicePreference::Cpu).with_max_generated_tokens(1),
    )
    .expect("bounded fixture loads");
    let mut sink = CollectingSink::new();
    let report = run_provider(
        &provider,
        &ModelRequest::new(MODEL_ID)
            .with_user("hello")
            .with_max_output_tokens(99),
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect("generation is provider-bounded");
    assert_eq!(report.text(), "world");

    let mut sink = CollectingSink::new();
    let oversized = ModelRequest::new(MODEL_ID)
        .with_user("hello hello hello hello hello hello hello hello hello");
    let error = run_provider(
        &provider,
        &oversized,
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited(),
    )
    .expect_err("prompt exceeds manifest context");
    assert!(error.to_string().contains("context limit"));
}

#[test]
fn cancellation_and_runtime_budget_stop_at_event_boundaries() {
    let fixture = fixture();
    let provider = fixture.load_cpu();
    let request = ModelRequest::new(MODEL_ID).with_user("hello");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut sink = CollectingSink::new();
    let cancelled = run_provider(
        &provider,
        &request,
        &mut sink,
        &cancel,
        RunBudget::unlimited(),
    )
    .expect("cooperative cancellation is a normal stop");
    assert_eq!(cancelled.stop, StopReason::Cancelled);
    assert_eq!(sink.kinds(), vec!["finished"]);

    let mut sink = CollectingSink::new();
    let bounded = run_provider(
        &provider,
        &request,
        &mut sink,
        &CancellationToken::new(),
        RunBudget::unlimited().with_max_events(1),
    )
    .expect("event budget is a normal stop");
    assert_eq!(
        bounded.stop,
        StopReason::BudgetExhausted(BudgetLimit::Events)
    );
    assert_eq!(bounded.text(), "");
    let evidence = provider
        .last_inference_report()
        .expect("report state")
        .expect("bounded execution report");
    assert!(evidence.native_operations > 0);
    assert!(evidence.capability.executed());
}

#[cfg(not(feature = "metal"))]
#[test]
fn metal_request_without_feature_fails_without_cpu_fallback() {
    let fixture = fixture();
    let error = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        PRINCIPAL,
        LoadConfig::new(DevicePreference::Metal),
    )
    .expect_err("uncompiled Metal is unavailable");
    assert!(matches!(error, ModelRuntimeError::DeviceUnavailable { .. }));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_request_without_feature_fails_without_cpu_fallback() {
    let fixture = fixture();
    let error = LocalModelProvider::load(
        &fixture.registry,
        &fixture.ledger,
        &fixture.storage,
        MODEL_ID,
        PRINCIPAL,
        LoadConfig::new(DevicePreference::Cuda),
    )
    .expect_err("uncompiled CUDA is unavailable");
    assert!(matches!(error, ModelRuntimeError::DeviceUnavailable { .. }));
}
