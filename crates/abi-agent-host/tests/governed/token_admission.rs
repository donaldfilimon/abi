//! Exhausted provider usage cannot admit a governed buffered call.
use super::*;

struct TokenProvider {
    runs: AtomicUsize,
    tokens: u64,
    checkpoint: bool,
}

impl ModelProvider for TokenProvider {
    fn id(&self) -> &'static str {
        "governed-token-admission"
    }

    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        if self.runs.fetch_add(1, Ordering::SeqCst) == 0 {
            assert_eq!(
                run.emit(&ModelEvent::ToolCall(ToolCall::new(
                    "call", "observe", "{}"
                ))),
                Flow::Continue
            );
            run.record_output_tokens(self.tokens);
            if self.checkpoint {
                assert_eq!(run.checkpoint(), Flow::Stop);
            }
        }
        Ok(())
    }
}

fn fixture(tokens: u64, checkpoint: bool) -> (GovernedRunReport, Ledger, RecordingActuator, usize) {
    let provider = TokenProvider {
        runs: AtomicUsize::new(0),
        tokens,
        checkpoint,
    };
    let mut ledger = Ledger::default();
    let (result, actuator) = run_provider_fixture(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        None,
        &provider,
        HostBudget::default().with_max_output_tokens(4),
    );
    (
        result.unwrap(),
        ledger,
        actuator,
        provider.runs.load(Ordering::SeqCst),
    )
}

fn exhausted(checkpoint: bool) {
    let (report, ledger, actuator, runs) = fixture(4, checkpoint);
    assert_eq!(
        report.driver.stop,
        HostStopReason::BudgetExhausted(HostBudgetLimit::OutputTokens)
    );
    assert_eq!(report.driver.usage.output_tokens, 4);
    assert_eq!(runs, 1);
    assert!(ledger.prepares.is_empty(), "no durable admission attempt");
    assert_eq!(ledger.intents, 0);
    assert_eq!(ledger.execution_ids, [] as [Digest; 0]);
    assert!(ledger.outcomes_by_id.is_empty());
    assert_eq!(actuator.records(), []);
    assert_eq!(report.executions, [] as [PreparedExecution; 0]);
    assert_eq!(report.outcomes, [] as [TerminalOutcome; 0]);
    assert!(
        ledger
            .stages
            .iter()
            .all(|stage| stage.decision != Decision::Allow),
        "no authorization or admission stage may pass after token exhaustion"
    );
    assert!(!ledger.stages.is_empty(), "driver stop remains audited");
}

#[test]
fn final_usage_at_ceiling_refuses_governed_buffered_call() {
    exhausted(false);
}

#[test]
fn explicit_checkpoint_token_stop_refuses_governed_buffered_call() {
    exhausted(true);
}

#[test]
fn below_ceiling_retains_governed_admission_and_recording() {
    let (report, ledger, actuator, runs) = fixture(3, false);
    assert_eq!(report.governance, GovernedStop::Completed);
    assert_eq!(report.driver.stop, HostStopReason::Completed);
    assert_eq!(report.driver.usage.output_tokens, 3);
    assert_eq!(runs, 2);
    assert_eq!(ledger.prepares.len(), 1);
    assert_eq!(ledger.intents, 1);
    assert_eq!(actuator.records().len(), 1);
    assert_eq!(report.executions.len(), 1);
    assert_eq!(report.outcomes.len(), 1);
    assert_eq!(report.outcomes[0].state, OutcomeState::Succeeded);
}
