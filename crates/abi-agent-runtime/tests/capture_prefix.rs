//! Retained runtime text must be a prefix, independently of stream chunks.

use abi_agent_runtime::{
    CancellationToken, CollectingSink, Flow, ModelEvent, RunBudget, RunContext, StopReason,
};

#[test]
fn truncated_capture_preserves_prefix_without_changing_delivered_events() {
    let cancellation = CancellationToken::new();
    let mut sink = CollectingSink::new();
    let mut run =
        RunContext::with_capture_limit(&mut sink, &cancellation, RunBudget::unlimited(), 2);
    assert_eq!(run.emit(&ModelEvent::text("aé")), Flow::Continue);
    assert_eq!(run.emit(&ModelEvent::text("b")), Flow::Continue);
    let report = run.finish();
    assert_eq!(report.stop, StopReason::Completed);
    assert_eq!(report.text(), "a");
    assert_eq!(report.output.dropped_bytes(), 3);
    assert_eq!(report.events, 2);
    assert_eq!(
        sink.events(),
        &[
            ModelEvent::text("aé"),
            ModelEvent::text("b"),
            ModelEvent::Finished {
                stop: StopReason::Completed
            },
        ]
    );
}
