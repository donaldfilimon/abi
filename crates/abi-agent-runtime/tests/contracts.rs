//! External-consumer contract tests.
//!
//! These drive `abi-agent-runtime` the way a host crate would: through the
//! public API only, with no access to crate internals.

use abi_agent_runtime::{
    AuditSink, CancellationToken, CapturedText, CollectingSink, DenyAllPolicy, EchoProvider,
    EffectScopedPolicy, EventSink, ExecutionPolicy, Flow, MAX_CAPTURED_BYTES, MemoryAuditSink,
    ModelEvent, ModelProvider, ModelRequest, NullAuditSink, NullSink, PolicyDecision, RunBudget,
    RunContext, RunReport, RuntimeError, ScriptedProvider, StaticToolRegistry, StopReason,
    ToolCall, ToolEffect, ToolRegistry, ToolResult, ToolSpec, ToolStatus, Usage,
    authorize_and_audit, run_provider,
};

/// A provider that emits `count` text events and stops when told to.
struct CountingProvider {
    count: usize,
}

impl ModelProvider for CountingProvider {
    fn id(&self) -> &'static str {
        "counting"
    }

    fn run(&self, _request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        for index in 0..self.count {
            if run.emit(&ModelEvent::text(index.to_string())) == Flow::Stop {
                return Ok(());
            }
        }
        Ok(())
    }
}

/// A sink that cancels the run after it has seen `after` events.
struct CancellingSink {
    inner: CollectingSink,
    token: CancellationToken,
    after: usize,
}

impl EventSink for CancellingSink {
    fn emit(&mut self, event: &ModelEvent) {
        self.inner.emit(event);
        if self.inner.len() >= self.after {
            self.token.cancel();
        }
    }
}

#[test]
fn boxed_provider_is_object_safe_and_runnable() {
    let providers: Vec<Box<dyn ModelProvider>> = vec![
        Box::new(EchoProvider::new()),
        Box::new(ScriptedProvider::new("scripted").with_text("scripted reply")),
    ];

    let request = ModelRequest::new("test-model").with_user("hello world");
    let mut outputs = Vec::new();
    for provider in &providers {
        let mut sink = NullSink;
        let cancel = CancellationToken::new();
        let report = run_provider(
            provider.as_ref(),
            &request,
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect("deterministic provider runs");
        outputs.push(report.output.into_text());
    }
    assert_eq!(outputs, vec!["hello world", "scripted reply"]);
}

#[test]
fn cancellation_stops_the_stream_at_the_next_event_boundary() {
    let token = CancellationToken::new();
    let mut sink = CancellingSink {
        inner: CollectingSink::new(),
        token: token.clone(),
        after: 3,
    };

    let report = run_provider(
        &CountingProvider { count: 50 },
        &ModelRequest::new("m"),
        &mut sink,
        &token,
        RunBudget::unlimited(),
    )
    .expect("provider honours the stop signal");

    // Exactly three provider events were delivered, then the terminal event.
    assert_eq!(report.events, 3);
    assert_eq!(report.stop, StopReason::Cancelled);
    assert!(!report.is_complete());
    assert_eq!(sink.inner.len(), 4);
    assert_eq!(
        sink.inner.kinds(),
        vec!["text_delta", "text_delta", "text_delta", "finished"]
    );
    assert_eq!(sink.inner.text(), "012");
    assert_eq!(report.text(), "012");
}

#[test]
fn a_run_cancelled_before_it_starts_delivers_only_the_terminal_event() {
    let token = CancellationToken::new();
    token.cancel();
    let mut sink = CollectingSink::new();

    let report = run_provider(
        &CountingProvider { count: 5 },
        &ModelRequest::new("m"),
        &mut sink,
        &token,
        RunBudget::unlimited(),
    )
    .expect("provider honours the stop signal");

    assert_eq!(report.events, 0);
    assert_eq!(report.stop, StopReason::Cancelled);
    assert_eq!(sink.kinds(), vec!["finished"]);
}

#[test]
fn event_budget_stops_the_stream_and_names_the_limit() {
    let token = CancellationToken::new();
    let mut sink = CollectingSink::new();

    let report = run_provider(
        &CountingProvider { count: 50 },
        &ModelRequest::new("m"),
        &mut sink,
        &token,
        RunBudget::unlimited().with_max_events(4),
    )
    .expect("provider honours the stop signal");

    assert_eq!(report.events, 4);
    assert_eq!(report.stop.as_str(), "budget_exhausted");
    assert_eq!(sink.len(), 5);
}

#[test]
fn capture_is_bounded_and_truncation_is_disclosed() {
    // The documented cap is a byte count, not a character count.
    assert_eq!(MAX_CAPTURED_BYTES, 64 * 1024);

    let oversized = "é".repeat(MAX_CAPTURED_BYTES);
    let captured = CapturedText::capture(&oversized);
    assert!(captured.text().len() <= MAX_CAPTURED_BYTES);
    assert!(captured.truncated());
    assert!(captured.dropped_bytes() > 0);
    // Truncation landed on a character boundary: the retained text is still
    // valid UTF-8 made only of whole characters.
    assert!(captured.text().chars().all(|c| c == 'é'));
    assert_eq!(captured.text().len() % 2, 0);

    // The whole event still reached the sink; only the retained copy is capped.
    let mut sink = CollectingSink::new();
    let token = CancellationToken::new();
    let provider = ScriptedProvider::new("scripted").with_text(oversized.clone());
    let report = run_provider(
        &provider,
        &ModelRequest::new("m"),
        &mut sink,
        &token,
        RunBudget::unlimited(),
    )
    .expect("scripted provider runs");
    assert!(report.output.truncated());
    assert!(report.text().len() <= MAX_CAPTURED_BYTES);
    assert_eq!(sink.text().len(), oversized.len());
}

#[test]
fn usage_is_counts_and_time_only() {
    let mut sink = CollectingSink::new();
    let token = CancellationToken::new();
    let report: RunReport = run_provider(
        &EchoProvider::new(),
        &ModelRequest::new("m")
            .with_system("be brief")
            .with_user("one two three"),
        &mut sink,
        &token,
        RunBudget::unlimited(),
    )
    .expect("echo provider runs");

    let usage: Usage = report.usage;
    // Two system words plus three user words, by the echo provider's
    // documented whitespace rule.
    assert_eq!(usage.input_tokens, 5);
    assert_eq!(usage.output_tokens, 3);
    assert_eq!(usage.total_tokens(), 8);
    assert_eq!(usage.merged(usage).total_tokens(), 16);
    assert_eq!(Usage::zero().total_tokens(), 0);
}

#[test]
fn tool_calls_arrive_as_structured_events() {
    let call = ToolCall::new("call-1", "read_file", "{\"path\":\"README.md\"}");
    let provider = ScriptedProvider::new("scripted")
        .with_text("looking that up: ")
        .with_tool_call(call.clone())
        .with_tool_result(ToolResult::ok("call-1", "file body"))
        .with_text("done");

    let mut sink = CollectingSink::new();
    let token = CancellationToken::new();
    let report = run_provider(
        &provider,
        &ModelRequest::new("m").with_tool(ToolSpec::new("read_file")),
        &mut sink,
        &token,
        RunBudget::unlimited(),
    )
    .expect("scripted provider runs");

    assert_eq!(report.text(), "looking that up: done");
    let observed: Vec<ToolCall> = sink
        .events()
        .iter()
        .filter_map(|event| match event {
            ModelEvent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(observed, vec![call]);

    let results: Vec<&ModelEvent> = sink
        .events()
        .iter()
        .filter(|event| event.kind() == "tool_result")
        .collect();
    assert_eq!(results.len(), 1);
}

#[test]
fn registry_and_policy_decide_without_executing_anything() {
    let registry: Box<dyn ToolRegistry> = Box::new(
        StaticToolRegistry::new()
            .with_spec(ToolSpec::new("read_file").with_description("read a file"))
            .with_spec(ToolSpec::new("write_file").with_effect(ToolEffect::Mutating))
            .with_spec(ToolSpec::new("rm_rf").with_effect(ToolEffect::Destructive)),
    );
    let policy: Box<dyn ExecutionPolicy> = Box::new(EffectScopedPolicy);
    let audit = MemoryAuditSink::new();

    let cases = [
        ("read_file", "allow"),
        ("write_file", "require_confirmation"),
        ("rm_rf", "deny"),
        ("undeclared_tool", "deny"),
    ];
    for (index, (tool, expected)) in cases.iter().enumerate() {
        let call = ToolCall::new(format!("c{index}"), *tool, "{}");
        let spec = registry.spec(tool);
        let decision = authorize_and_audit(policy.as_ref(), &audit, &call, spec.as_ref());
        assert_eq!(decision.as_str(), *expected, "tool {tool}");
    }

    let entries = audit.entries();
    assert_eq!(entries.len(), 4);
    assert!(entries.iter().all(|entry| entry.policy == "effect-scoped"));
    assert_eq!(entries[3].tool, "undeclared_tool");

    // The safe default refuses everything, including read-only tools.
    let deny_all: Box<dyn ExecutionPolicy> = Box::new(DenyAllPolicy);
    let call = ToolCall::new("c9", "read_file", "{}");
    let spec = registry.spec("read_file");
    assert!(
        !deny_all.authorize(&call, spec.as_ref()).is_allowed(),
        "the default policy must not allow tool use"
    );

    // A null audit sink is a valid trait object and simply discards.
    let null: Box<dyn AuditSink> = Box::new(NullAuditSink);
    let decision = authorize_and_audit(deny_all.as_ref(), null.as_ref(), &call, spec.as_ref());
    assert!(matches!(decision, PolicyDecision::Deny { .. }));
}

#[test]
fn denied_tool_results_carry_the_reason_and_bounded_payload() {
    let result = ToolResult::denied("c1", "policy refused");
    assert_eq!(result.status, ToolStatus::Denied);
    assert_eq!(result.payload.text(), "policy refused");
    assert!(!result.payload.truncated());

    let huge = ToolResult::with_limit(
        "c2",
        ToolStatus::Ok,
        &"x".repeat(MAX_CAPTURED_BYTES + 10),
        MAX_CAPTURED_BYTES,
    );
    assert!(huge.payload.truncated());
    assert_eq!(huge.payload.dropped_bytes(), 10);
}

#[test]
fn an_empty_request_is_an_error_not_a_fabricated_reply() {
    let mut sink = CollectingSink::new();
    let token = CancellationToken::new();
    let error = run_provider(
        &EchoProvider::new(),
        &ModelRequest::new("m"),
        &mut sink,
        &token,
        RunBudget::unlimited(),
    )
    .expect_err("there is nothing to echo");
    assert_eq!(error, RuntimeError::EmptyRequest);
    assert_eq!(error.to_string(), "request carried no messages");
    // The stream is still closed rather than left hanging.
    assert_eq!(sink.kinds(), vec!["finished"]);
}
