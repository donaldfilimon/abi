//! External contracts for bounded host orchestration.

use abi_agent_host::{
    AgentHost, HostBudget, HostBudgetLimit, HostError, HostStopReason, ToolExecutionContext,
    ToolExecutionError, ToolExecutor, ToolOutput,
};
use abi_agent_runtime::{
    CancellationToken, CollectingSink, DenyAllPolicy, EffectScopedPolicy, ExecutionPolicy, Flow,
    MemoryAuditSink, ModelEvent, ModelProvider, ModelRequest, PolicyDecision, Role, RunContext,
    RuntimeError, StaticToolRegistry, StopReason, ToolCall, ToolRegistry, ToolResult, ToolSpec,
    ToolStatus,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

const READ_SCHEMA: &str = r#"{
  "type":"object",
  "properties":{"path":{"type":"string"}},
  "required":["path"],
  "additionalProperties":false
}"#;

struct SequenceProvider {
    turns: Mutex<VecDeque<Vec<ModelEvent>>>,
    requests: Mutex<Vec<ModelRequest>>,
    output_tokens_per_turn: u64,
}

impl SequenceProvider {
    fn new(turns: impl IntoIterator<Item = Vec<ModelEvent>>) -> Self {
        Self {
            turns: Mutex::new(turns.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            output_tokens_per_turn: 0,
        }
    }

    fn with_output_tokens_per_turn(mut self, tokens: u64) -> Self {
        self.output_tokens_per_turn = tokens;
        self
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().expect("requests lock").clone()
    }
}

impl ModelProvider for SequenceProvider {
    fn id(&self) -> &'static str {
        "sequence"
    }

    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        self.requests
            .lock()
            .expect("requests lock")
            .push(request.clone());
        if run.emit(&ModelEvent::started(request.model())) == Flow::Stop {
            return Ok(());
        }
        let events = self
            .turns
            .lock()
            .expect("turns lock")
            .pop_front()
            .ok_or(RuntimeError::ScriptExhausted { scripted: 0 })?;
        for event in events {
            if run.emit(&event) == Flow::Stop {
                return Ok(());
            }
        }
        run.record_output_tokens(self.output_tokens_per_turn);
        Ok(())
    }
}

#[derive(Default)]
struct FixtureExecutor {
    calls: AtomicUsize,
    outputs: Mutex<VecDeque<Result<ToolOutput, ToolExecutionError>>>,
}

impl FixtureExecutor {
    fn returning(
        outputs: impl IntoIterator<Item = Result<ToolOutput, ToolExecutionError>>,
    ) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            outputs: Mutex::new(outputs.into_iter().collect()),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ToolExecutor for FixtureExecutor {
    fn execute(
        &self,
        _call: &ToolCall,
        _spec: &ToolSpec,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if context.should_stop() {
            return Err(ToolExecutionError::new("execution boundary stopped"));
        }
        self.outputs
            .lock()
            .expect("outputs lock")
            .pop_front()
            .unwrap_or_else(|| Ok(ToolOutput::ok("ok")))
    }
}

struct AllowAll;

impl ExecutionPolicy for AllowAll {
    fn name(&self) -> &'static str {
        "allow-all-fixture"
    }

    fn authorize(&self, _call: &ToolCall, _spec: Option<&ToolSpec>) -> PolicyDecision {
        PolicyDecision::Allow
    }
}

fn tool_registry() -> StaticToolRegistry {
    StaticToolRegistry::new().with_spec(ToolSpec::new("read").with_input_schema(READ_SCHEMA))
}

fn call(id: &str) -> ModelEvent {
    ModelEvent::ToolCall(ToolCall::new(id, "read", r#"{"path":"notes.md"}"#))
}

fn request() -> ModelRequest {
    ModelRequest::new("fixture-model")
        .with_user("inspect")
        .with_tool(ToolSpec::new("model-invented-tool"))
}

fn assemble_host<'a>(
    provider: &'a dyn ModelProvider,
    registry: &dyn ToolRegistry,
    policy: &'a dyn ExecutionPolicy,
    audit: &'a MemoryAuditSink,
    executor: &'a dyn ToolExecutor,
) -> AgentHost<'a> {
    AgentHost::new(provider, registry, policy, audit, executor).expect("host assembles")
}

#[test]
fn ordered_call_execution_result_and_continuation_complete_offline() {
    let provider = SequenceProvider::new([vec![call("c1")], vec![ModelEvent::text("complete")]]);
    let executor = FixtureExecutor::returning([Ok(ToolOutput::ok("contents"))]);
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let mut sink = CollectingSink::new();
    let report = host
        .run(
            &request(),
            &mut sink,
            &CancellationToken::new(),
            HostBudget::bounded(),
        )
        .expect("host run completes");

    assert_eq!(
        sink.kinds(),
        [
            "started",
            "tool_call",
            "tool_result",
            "started",
            "text_delta",
            "finished"
        ]
    );
    assert!(report.is_complete());
    assert_eq!(report.text(), "complete");
    assert_eq!(report.tool_calls, 1);
    assert_eq!(report.provider_runs, 2);
    assert_eq!(report.tool_results[0].status, ToolStatus::Ok);
    assert_eq!(executor.calls(), 1);
    assert_eq!(audit.len(), 1);

    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tools(), registry.specs());
    assert_eq!(requests[0].tools().len(), 1);
    let tool_message = requests[1].messages().last().expect("tool continuation");
    assert_eq!(tool_message.role, Role::Tool);
    assert!(tool_message.content.contains(r#""call_id":"c1""#));
    assert!(tool_message.content.contains(r#""payload":"contents""#));
}

#[test]
fn denied_and_confirmation_calls_are_audited_and_never_executed() {
    for policy in [
        &DenyAllPolicy as &dyn ExecutionPolicy,
        &EffectScopedPolicy as &dyn ExecutionPolicy,
    ] {
        let provider = SequenceProvider::new([vec![call("c1")], vec![ModelEvent::text("done")]]);
        let executor = FixtureExecutor::default();
        let audit = MemoryAuditSink::new();
        let mutating = StaticToolRegistry::new().with_spec(
            ToolSpec::new("read")
                .with_input_schema(READ_SCHEMA)
                .with_effect(abi_agent_runtime::ToolEffect::Mutating),
        );
        let host = assemble_host(&provider, &mutating, policy, &audit, &executor);
        let mut sink = CollectingSink::new();
        let report = host
            .run(
                &request(),
                &mut sink,
                &CancellationToken::new(),
                HostBudget::bounded(),
            )
            .expect("denial continues safely");
        assert_eq!(report.tool_results[0].status, ToolStatus::Denied);
        assert_eq!(executor.calls(), 0);
        assert_eq!(audit.len(), 1);
    }
}

#[test]
fn executor_failures_become_bounded_error_results_and_continue() {
    let provider = SequenceProvider::new([vec![call("c1")], vec![ModelEvent::text("recovered")]]);
    let executor = FixtureExecutor::returning([Err(ToolExecutionError::new("fixture failed"))]);
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let mut sink = CollectingSink::new();
    let report = host
        .run(
            &request(),
            &mut sink,
            &CancellationToken::new(),
            HostBudget::bounded(),
        )
        .expect("executor error is model-visible");
    assert_eq!(report.tool_results[0].status, ToolStatus::Error);
    assert_eq!(report.tool_results[0].payload.text(), "fixture failed");
    assert_eq!(report.text(), "recovered");
}

#[test]
fn unknown_malformed_and_schema_invalid_calls_fail_before_policy_or_execution() {
    let cases = [
        (ToolCall::new("c1", "missing", "{}"), "unknown"),
        (ToolCall::new("c2", "read", "{"), "malformed"),
        (ToolCall::new("c3", "read", r#"{"path":7}"#), "schema"),
    ];
    for (bad_call, expected) in cases {
        let provider = SequenceProvider::new([vec![ModelEvent::ToolCall(bad_call)]]);
        let executor = FixtureExecutor::default();
        let audit = MemoryAuditSink::new();
        let registry = tool_registry();
        let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
        let mut sink = CollectingSink::new();
        let error = host
            .run(
                &request(),
                &mut sink,
                &CancellationToken::new(),
                HostBudget::bounded(),
            )
            .expect_err("invalid call fails closed");
        match expected {
            "unknown" => assert!(matches!(error, HostError::UnknownTool { .. })),
            "malformed" => assert!(matches!(error, HostError::MalformedToolInput { .. })),
            "schema" => assert!(matches!(error, HostError::ToolSchemaViolation { .. })),
            _ => unreachable!(),
        }
        assert_eq!(executor.calls(), 0);
        assert_eq!(audit.len(), 0);
        assert_eq!(sink.events().last().map(ModelEvent::kind), Some("finished"));
        assert_eq!(
            sink.events().last(),
            Some(&ModelEvent::Finished {
                stop: StopReason::Failed
            })
        );
    }
}

#[test]
fn duplicate_call_ids_fail_before_the_second_execution() {
    let provider = SequenceProvider::new([vec![call("same")], vec![call("same")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let error = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded(),
        )
        .expect_err("duplicate fails");
    assert!(matches!(error, HostError::DuplicateCallId { .. }));
    assert_eq!(executor.calls(), 1);
    assert_eq!(audit.len(), 1);
}

#[test]
fn provider_fabricated_results_and_post_terminal_events_fail_closed() {
    let fabricated =
        SequenceProvider::new([vec![ModelEvent::ToolResult(ToolResult::ok("c1", "fake"))]]);
    let post_terminal = SequenceProvider::new([vec![
        ModelEvent::Finished {
            stop: StopReason::Completed,
        },
        ModelEvent::text("after"),
    ]]);
    for (provider, expected) in [
        (&fabricated as &dyn ModelProvider, "result"),
        (&post_terminal as &dyn ModelProvider, "terminal"),
    ] {
        let executor = FixtureExecutor::default();
        let audit = MemoryAuditSink::new();
        let registry = tool_registry();
        let host = assemble_host(provider, &registry, &AllowAll, &audit, &executor);
        let error = host
            .run(
                &request(),
                &mut CollectingSink::new(),
                &CancellationToken::new(),
                HostBudget::bounded(),
            )
            .expect_err("invalid provider stream fails");
        if expected == "result" {
            assert!(matches!(error, HostError::ProviderToolResult { .. }));
        } else {
            assert!(matches!(error, HostError::PostTerminalEvent { .. }));
        }
        assert_eq!(executor.calls(), 0);
    }
}

#[test]
fn oversized_raw_tool_output_is_rejected_not_truncated() {
    let provider = SequenceProvider::new([vec![call("c1")]]);
    let executor = FixtureExecutor::returning([Ok(ToolOutput::ok("four"))]);
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let error = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_tool_result_bytes(3),
        )
        .expect_err("oversized result fails");
    assert_eq!(
        error,
        HostError::ToolResultTooLarge {
            call_id: "c1".to_owned(),
            bytes: 4,
            limit: 3,
        }
    );
}

#[test]
fn tool_call_round_provider_run_and_event_budgets_precede_execution() {
    let budgets = [
        (
            HostBudget::bounded().with_max_tool_calls(0),
            HostBudgetLimit::ToolCalls,
        ),
        (
            HostBudget::bounded().with_max_tool_rounds(0),
            HostBudgetLimit::ToolRounds,
        ),
    ];
    for (budget, expected_limit) in budgets {
        let provider = SequenceProvider::new([vec![call("c1")]]);
        let executor = FixtureExecutor::default();
        let audit = MemoryAuditSink::new();
        let registry = tool_registry();
        let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
        let error = host
            .run(
                &request(),
                &mut CollectingSink::new(),
                &CancellationToken::new(),
                budget,
            )
            .expect_err("budget fails closed");
        assert_eq!(error, HostError::BudgetExceeded(expected_limit));
        assert_eq!(executor.calls(), 0);
    }

    let provider = SequenceProvider::new([vec![call("c1")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let report = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_events(1),
        )
        .expect("provider event budget returns a terminal report");
    assert_eq!(
        report.stop,
        HostStopReason::BudgetExhausted(HostBudgetLimit::Events)
    );
    assert_eq!(executor.calls(), 0);

    let provider = SequenceProvider::new([vec![call("c1")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let error = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_events(2),
        )
        .expect_err("call execution reserves its terminal result event");
    assert_eq!(error, HostError::BudgetExceeded(HostBudgetLimit::Events));
    assert_eq!(executor.calls(), 0);

    let provider = SequenceProvider::new([vec![call("c1")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let error = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_provider_runs(0),
        )
        .expect_err("provider budget fails before provider");
    assert_eq!(
        error,
        HostError::BudgetExceeded(HostBudgetLimit::ProviderRuns)
    );
    assert_eq!(executor.calls(), 0);
    assert_eq!(provider.requests(), Vec::<ModelRequest>::new());
}

#[test]
fn oversized_provider_events_are_rejected_before_the_turn_buffer_clones_them() {
    let provider = SequenceProvider::new([vec![ModelEvent::text("large")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let error = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_event_bytes(4),
        )
        .expect_err("oversized event fails closed");
    assert_eq!(
        error,
        HostError::BudgetExceeded(HostBudgetLimit::EventBytes)
    );
    assert_eq!(executor.calls(), 0);
}

#[test]
fn output_byte_and_token_budgets_return_precise_terminal_reports() {
    let provider = SequenceProvider::new([vec![ModelEvent::text("abc")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let report = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_output_bytes(2),
        )
        .expect("byte budget is a terminal report");
    assert_eq!(
        report.stop,
        HostStopReason::BudgetExhausted(HostBudgetLimit::OutputBytes)
    );
    assert_eq!(report.text(), "");

    let provider =
        SequenceProvider::new([vec![ModelEvent::text("a")]]).with_output_tokens_per_turn(2);
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let report = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_output_tokens(2),
        )
        .expect("token budget is a terminal report");
    assert_eq!(
        report.stop,
        HostStopReason::BudgetExhausted(HostBudgetLimit::OutputTokens)
    );
}

#[test]
fn pre_cancelled_and_expired_runs_admit_no_provider_or_tool_work() {
    let provider = Arc::new(SequenceProvider::new([vec![call("c1")]]));
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(provider.as_ref(), &registry, &AllowAll, &audit, &executor);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let report = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &cancellation,
            HostBudget::bounded(),
        )
        .expect("cancellation is a terminal report");
    assert_eq!(report.stop, HostStopReason::Cancelled);
    assert_eq!(provider.requests(), Vec::<ModelRequest>::new());
    assert_eq!(executor.calls(), 0);

    let error = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_duration(Duration::ZERO),
        )
        .expect_err("expired deadline fails before provider");
    assert_eq!(error, HostError::BudgetExceeded(HostBudgetLimit::Deadline));
    assert_eq!(provider.requests(), Vec::<ModelRequest>::new());
}
