//! Deadline expiry during an observed callback refuses later admission.
use super::*;
use std::thread;

const WINDOW: Duration = Duration::from_millis(100);
const CALLBACK_WORK: Duration = Duration::from_millis(200);

struct DeadlineSink {
    inner: CollectingSink,
    kind: &'static str,
    reached: bool,
}
impl EventSink for DeadlineSink {
    fn emit(&mut self, event: &ModelEvent) {
        self.inner.emit(event);
        if event.kind() == self.kind {
            self.reached = true;
            thread::sleep(CALLBACK_WORK);
        }
    }
}
fn deadline_error(result: Result<HostRunReport, HostError>) {
    assert_eq!(
        result.unwrap_err(),
        HostError::BudgetExceeded(HostBudgetLimit::Deadline)
    );
}
#[test]
fn deadline_callback_started_admits_no_later_buffered_text() {
    let provider = SequenceProvider::new([vec![ModelEvent::text("after-expiry")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let mut sink = DeadlineSink {
        inner: CollectingSink::new(),
        kind: "started",
        reached: false,
    };
    let result = host.run(
        &request(),
        &mut sink,
        &CancellationToken::new(),
        HostBudget::bounded().with_max_duration(WINDOW),
    );
    assert!(
        sink.reached,
        "actual host callback must run for attributable deadline evidence"
    );
    deadline_error(result);
    assert_eq!(sink.inner.kinds(), ["started", "finished"]);
    assert_eq!(executor.calls(), 0);
    assert!(audit.is_empty());
}
#[test]
fn deadline_callback_tool_call_admits_no_authorization_or_executor() {
    let provider = SequenceProvider::new([vec![call("c1")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let mut sink = DeadlineSink {
        inner: CollectingSink::new(),
        kind: "tool_call",
        reached: false,
    };
    let result = host.run(
        &request(),
        &mut sink,
        &CancellationToken::new(),
        HostBudget::bounded().with_max_duration(WINDOW),
    );
    assert!(
        sink.reached,
        "actual call callback must run before the window is exhausted"
    );
    deadline_error(result);
    assert_eq!(
        executor.calls(),
        0,
        "post-deadline tool admission must be refused by the host"
    );
    assert!(
        audit.is_empty(),
        "authorization cannot begin after the sink consumed the deadline"
    );
    assert_eq!(sink.inner.kinds(), ["started", "tool_call", "finished"]);
}
struct DeadlineAuthorization {
    in_policy: bool,
    reached: AtomicUsize,
    audit: MemoryAuditSink,
}
impl ExecutionPolicy for DeadlineAuthorization {
    fn name(&self) -> &'static str {
        "deadline-fixture"
    }
    fn authorize(&self, _call: &ToolCall, _spec: Option<&ToolSpec>) -> PolicyDecision {
        if self.in_policy {
            self.reached.fetch_add(1, Ordering::SeqCst);
            thread::sleep(CALLBACK_WORK);
        }
        PolicyDecision::Allow
    }
}
impl AuditSink for DeadlineAuthorization {
    fn record(&self, entry: &AuditEntry) {
        self.audit.record(entry);
        if !self.in_policy {
            self.reached.fetch_add(1, Ordering::SeqCst);
            thread::sleep(CALLBACK_WORK);
        }
    }
}
#[test]
fn deadline_callback_policy_or_audit_admits_no_executor() {
    for in_policy in [true, false] {
        let provider = SequenceProvider::new([vec![call("c1")]]);
        let executor = FixtureExecutor::default();
        let registry = tool_registry();
        let auth = DeadlineAuthorization {
            in_policy,
            reached: AtomicUsize::new(0),
            audit: MemoryAuditSink::new(),
        };
        let host = AgentHost::new(&provider, &registry, &auth, &auth, &executor).unwrap();
        let mut sink = CollectingSink::new();
        let result = host.run(
            &request(),
            &mut sink,
            &CancellationToken::new(),
            HostBudget::bounded().with_max_duration(WINDOW),
        );
        assert_eq!(
            auth.reached.load(Ordering::SeqCst),
            1,
            "deadline callback must be exercised"
        );
        deadline_error(result);
        assert_eq!(
            executor.calls(),
            0,
            "authorization/audit may not admit post-deadline execution"
        );
        assert_eq!(
            auth.audit.len(),
            1,
            "the already observed decision remains auditable"
        );
        assert_eq!(sink.kinds(), ["started", "tool_call", "finished"]);
    }
}

#[test]
fn deadline_callback_valid_control_retains_normal_tool_continuation() {
    let provider = SequenceProvider::new([vec![call("c1")], vec![ModelEvent::text("complete")]]);
    let executor = FixtureExecutor::default();
    let audit = MemoryAuditSink::new();
    let registry = tool_registry();
    let host = assemble_host(&provider, &registry, &AllowAll, &audit, &executor);
    let mut sink = DeadlineSink {
        inner: CollectingSink::new(),
        kind: "never",
        reached: false,
    };
    let result = host
        .run(
            &request(),
            &mut sink,
            &CancellationToken::new(),
            HostBudget::bounded().with_max_duration(Duration::from_secs(10)),
        )
        .unwrap();
    assert_eq!(result.stop, HostStopReason::Completed);
    assert_eq!(result.text(), "complete");
    assert_eq!(executor.calls(), 1);
    assert_eq!(audit.len(), 1);
    assert_eq!(provider.requests().len(), 2);
    assert!(!sink.reached);
}
