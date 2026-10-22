//! Completed provider usage must gate buffered tool admission.

use super::*;

struct CheckpointedProvider(SequenceProvider);

impl ModelProvider for CheckpointedProvider {
    fn id(&self) -> &'static str {
        "checkpointed-sequence"
    }

    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        self.0.run(request, run)?;
        let _ = run.checkpoint();
        Ok(())
    }
}

fn assert_token_admission(
    provider: &dyn ModelProvider,
    ceiling: u64,
    expected_calls: usize,
) -> HostRunReport {
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(provider, &registry, &AllowAll, &audit, &executor);
    let report = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_output_tokens(ceiling),
        )
        .expect("token exhaustion is a terminal report");
    assert_eq!(
        report.stop,
        HostStopReason::BudgetExhausted(HostBudgetLimit::OutputTokens)
    );
    assert_eq!(executor.calls(), expected_calls, "no post-budget execution");
    assert_eq!(report.tool_calls, expected_calls as u64);
    assert_eq!(
        audit.entries().len(),
        expected_calls,
        "no post-budget authorization"
    );
    report
}

#[test]
fn final_usage_at_token_ceiling_prevents_buffered_tool_execution() {
    let provider = SequenceProvider::new([vec![call("c1")], vec![]]).with_output_tokens_per_turn(2);
    assert_token_admission(&provider, 2, 0);
}

#[test]
fn provider_token_stop_prevents_buffered_tool_execution() {
    let provider = CheckpointedProvider(
        SequenceProvider::new([vec![call("c1")], vec![]]).with_output_tokens_per_turn(2),
    );
    assert_token_admission(&provider, 2, 0);
}

#[test]
fn below_token_ceiling_still_executes_then_stops_on_cumulative_usage() {
    let provider = SequenceProvider::new([
        vec![call("c1")],
        vec![ModelEvent::text("retained"), call("c2")],
    ])
    .with_output_tokens_per_turn(2);
    let report = assert_token_admission(&provider, 3, 1);
    assert_eq!(report.text(), "retained", "prior text remains delivered");
}

#[test]
fn provider_event_stop_prevents_buffered_tool_authorization() {
    let provider = CheckpointedProvider(SequenceProvider::new([vec![call("c1")]]));
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let report = host
        .run(
            &request(),
            &mut CollectingSink::new(),
            &CancellationToken::new(),
            HostBudget::bounded().with_max_events(2),
        )
        .expect("provider event exhaustion is a terminal report");
    assert_eq!(
        report.stop,
        HostStopReason::BudgetExhausted(HostBudgetLimit::Events)
    );
    assert_eq!(executor.calls(), 0);
    assert_eq!(audit.entries(), [] as [AuditEntry; 0]);
}
