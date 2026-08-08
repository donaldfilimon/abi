use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
    mpsc,
};

fn noop() -> TaskFn {
    Box::new(|| Ok(()))
}

fn failing(message: &str) -> TaskFn {
    let message = message.to_string();
    Box::new(move || Err(message))
}

#[test]
fn a_new_scheduler_is_empty() {
    let scheduler = Scheduler::new();
    assert_eq!(scheduler.stats(), Stats::default());
}

#[test]
fn ids_start_at_one_and_increase() {
    let scheduler = Scheduler::new();
    assert_eq!(scheduler.submit("a", TaskPriority::Normal, noop()), 1);
    assert_eq!(scheduler.submit("b", TaskPriority::Normal, noop()), 2);
    assert_eq!(scheduler.submit("c", TaskPriority::Normal, noop()), 3);
}

#[test]
fn submit_then_run_reaches_completed() {
    let scheduler = Scheduler::new();
    let id = scheduler.submit("work", TaskPriority::Normal, noop());
    assert_eq!(scheduler.pending_count(), 1);

    assert_eq!(scheduler.run_next().expect("runs"), Some(id));

    let info = scheduler.task(id).expect("task exists");
    assert_eq!(info.status, TaskStatus::Completed);
    assert!(info.started_at > 0);
    assert!(info.completed_at > 0);
    assert!(info.error_msg.is_none());
    assert_eq!(scheduler.pending_count(), 0);
    assert_eq!(scheduler.completed_count(), 1);
}

#[test]
fn run_next_on_an_empty_scheduler_returns_none() {
    let scheduler = Scheduler::new();
    assert_eq!(scheduler.run_next().expect("no error"), None);
}

#[test]
fn concurrent_callers_never_execute_more_than_one_task() {
    let scheduler = Arc::new(Scheduler::new());
    let second_started = Arc::new(AtomicBool::new(false));
    let (first_started_tx, first_started_rx) = mpsc::sync_channel(0);
    let (release_first_tx, release_first_rx) = mpsc::sync_channel(0);

    scheduler.submit(
        "first",
        TaskPriority::Normal,
        Box::new(move || {
            first_started_tx.send(()).expect("signal first task");
            release_first_rx.recv().expect("release first task");
            Ok(())
        }),
    );
    let second_started_in_task = Arc::clone(&second_started);
    scheduler.submit(
        "second",
        TaskPriority::Normal,
        Box::new(move || {
            second_started_in_task.store(true, AtomicOrdering::SeqCst);
            Ok(())
        }),
    );

    let first_scheduler = Arc::clone(&scheduler);
    let first_runner =
        std::thread::spawn(move || first_scheduler.run_next().expect("first runner"));
    first_started_rx.recv().expect("first task started");

    let second_scheduler = Arc::clone(&scheduler);
    let (second_ready_tx, second_ready_rx) = mpsc::sync_channel(0);
    let second_runner = std::thread::spawn(move || {
        second_ready_tx.send(()).expect("second caller ready");
        second_scheduler.run_next().expect("second runner")
    });
    second_ready_rx.recv().expect("second caller ready");
    assert!(matches!(
        scheduler.execution.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    assert!(!second_started.load(AtomicOrdering::SeqCst));
    assert_eq!(scheduler.stats().running, 1);
    assert_eq!(scheduler.stats().pending, 1);

    release_first_tx.send(()).expect("release first task");
    assert_eq!(first_runner.join().expect("join first"), Some(1));
    assert_eq!(second_runner.join().expect("join second"), Some(2));
    assert!(second_started.load(AtomicOrdering::SeqCst));
    assert_eq!(scheduler.stats().running, 0);
}

#[test]
fn higher_priority_runs_first() {
    let scheduler = Scheduler::new();
    let order = Arc::new(Mutex::new(Vec::new()));

    for (name, priority) in [
        ("low", TaskPriority::Low),
        ("critical", TaskPriority::Critical),
        ("normal", TaskPriority::Normal),
        ("high", TaskPriority::High),
    ] {
        let order = Arc::clone(&order);
        scheduler.submit(
            name,
            priority,
            Box::new(move || {
                order.lock().unwrap().push(name);
                Ok(())
            }),
        );
    }

    scheduler.run_all().expect("all succeed");
    assert_eq!(
        *order.lock().unwrap(),
        ["critical", "high", "normal", "low"]
    );
}

#[test]
fn equal_priority_runs_in_submission_order() {
    // Zig's heap gave no tie-break, so equal-priority order was unspecified.
    let scheduler = Scheduler::new();
    let order = Arc::new(Mutex::new(Vec::new()));
    for name in ["first", "second", "third", "fourth"] {
        let order = Arc::clone(&order);
        scheduler.submit(
            name,
            TaskPriority::Normal,
            Box::new(move || {
                order.lock().unwrap().push(name);
                Ok(())
            }),
        );
    }
    scheduler.run_all().expect("all succeed");
    assert_eq!(
        *order.lock().unwrap(),
        ["first", "second", "third", "fourth"]
    );
}

#[test]
fn a_failing_task_records_its_message() {
    let scheduler = Scheduler::new();
    let id = scheduler.submit("bad", TaskPriority::Normal, failing("disk on fire"));

    let err = scheduler.run_next().expect_err("should fail");
    assert_eq!(
        err,
        SchedulerError::TaskFailed {
            id,
            message: "disk on fire".to_string()
        }
    );

    let info = scheduler.task(id).expect("task exists");
    assert_eq!(info.status, TaskStatus::Failed);
    assert_eq!(info.error_msg.as_deref(), Some("disk on fire"));
    assert_eq!(scheduler.failed_count(), 1);
}

#[test]
fn run_all_drains_past_a_failure_and_reports_the_first() {
    let scheduler = Scheduler::new();
    let ran = Arc::new(AtomicUsize::new(0));

    // All equal priority, so they run in submission order: ok, bad, bad2, ok2.
    for spec in ["ok", "bad", "bad2", "ok2"] {
        let ran = Arc::clone(&ran);
        let fails = spec.starts_with("bad");
        scheduler.submit(
            spec,
            TaskPriority::Normal,
            Box::new(move || {
                ran.fetch_add(1, AtomicOrdering::SeqCst);
                if fails {
                    Err(format!("{spec} failed"))
                } else {
                    Ok(())
                }
            }),
        );
    }

    let err = scheduler.run_all().expect_err("first failure surfaces");
    // Every task ran — a failure must not strand work queued behind it.
    assert_eq!(ran.load(AtomicOrdering::SeqCst), 4);
    // The FIRST failure is the one reported.
    match err {
        SchedulerError::TaskFailed { message, .. } => assert_eq!(message, "bad failed"),
        other => panic!("unexpected error: {other:?}"),
    }

    let stats = scheduler.stats();
    assert_eq!(stats.completed, 2);
    assert_eq!(stats.failed, 2);
    assert_eq!(stats.pending, 0);
}

#[test]
fn cancel_prevents_execution() {
    let scheduler = Scheduler::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let ran_clone = Arc::clone(&ran);
    let id = scheduler.submit(
        "cancelled",
        TaskPriority::Normal,
        Box::new(move || {
            ran_clone.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(())
        }),
    );

    scheduler.cancel(id).expect("cancel a pending task");
    assert_eq!(scheduler.run_next().expect("nothing runnable"), None);
    assert_eq!(ran.load(AtomicOrdering::SeqCst), 0);

    let info = scheduler.task(id).expect("task exists");
    assert_eq!(info.status, TaskStatus::Cancelled);
}

#[test]
fn cancel_keeps_the_counters_consistent() {
    // Zig cancelled by mutating the list while leaving a stale heap copy;
    // pending was tracked separately and could disagree. Here pending is
    // derived, so it cannot.
    let scheduler = Scheduler::new();
    let keep = scheduler.submit("keep", TaskPriority::Normal, noop());
    let drop = scheduler.submit("drop", TaskPriority::High, noop());

    scheduler.cancel(drop).expect("cancel");
    let stats = scheduler.stats();
    assert_eq!(stats.pending, 1);
    assert_eq!(stats.cancelled, 1);
    assert_eq!(stats.total_tasks, 2);

    // The cancelled high-priority task must not be picked next.
    assert_eq!(scheduler.run_next().expect("runs"), Some(keep));
}

#[test]
fn peek_skips_a_cancelled_higher_priority_task() {
    // The exact bug Zig's `peek` needed a workaround for.
    let scheduler = Scheduler::new();
    let normal = scheduler.submit("normal", TaskPriority::Normal, noop());
    let critical = scheduler.submit("critical", TaskPriority::Critical, noop());

    assert_eq!(scheduler.peek().map(|t| t.id), Some(critical));
    scheduler.cancel(critical).expect("cancel");
    assert_eq!(
        scheduler.peek().map(|t| t.id),
        Some(normal),
        "peek must not report a cancelled task as next"
    );
}

#[test]
fn peek_does_not_consume() {
    let scheduler = Scheduler::new();
    let id = scheduler.submit("t", TaskPriority::Normal, noop());
    assert_eq!(scheduler.peek().map(|t| t.id), Some(id));
    assert_eq!(scheduler.peek().map(|t| t.id), Some(id));
    assert_eq!(scheduler.pending_count(), 1);
}

#[test]
fn peek_on_an_empty_scheduler_is_none() {
    assert!(Scheduler::new().peek().is_none());
}

#[test]
fn cancelling_a_finished_task_is_an_invalid_state() {
    let scheduler = Scheduler::new();
    let id = scheduler.submit("t", TaskPriority::Normal, noop());
    scheduler.run_next().expect("runs");
    assert_eq!(
        scheduler.cancel(id).expect_err("cannot cancel"),
        SchedulerError::InvalidState {
            id,
            status: TaskStatus::Completed
        }
    );
}

#[test]
fn cancelling_twice_fails_the_second_time() {
    let scheduler = Scheduler::new();
    let id = scheduler.submit("t", TaskPriority::Normal, noop());
    scheduler.cancel(id).expect("first cancel");
    assert!(scheduler.cancel(id).is_err());
    assert_eq!(scheduler.cancelled_count(), 1, "must not double-count");
}

#[test]
fn cancelling_an_unknown_id_reports_not_found() {
    let scheduler = Scheduler::new();
    assert_eq!(
        scheduler.cancel(999).expect_err("no such task"),
        SchedulerError::NotFound { id: 999 }
    );
}

#[test]
fn tasks_are_listed_in_submission_order() {
    let scheduler = Scheduler::new();
    scheduler.submit("first", TaskPriority::Low, noop());
    scheduler.submit("second", TaskPriority::Critical, noop());
    scheduler.submit("third", TaskPriority::Normal, noop());

    let names: Vec<String> = scheduler.tasks().into_iter().map(|t| t.name).collect();
    assert_eq!(names, ["first", "second", "third"]);
}

#[test]
fn a_task_may_submit_more_work_without_deadlocking() {
    // The lock must be released across execution.
    let scheduler = Arc::new(Scheduler::new());
    let inner = Arc::clone(&scheduler);
    scheduler.submit(
        "outer",
        TaskPriority::Normal,
        Box::new(move || {
            inner.submit("inner", TaskPriority::Normal, Box::new(|| Ok(())));
            Ok(())
        }),
    );
    scheduler.run_all().expect("no deadlock, no failure");
    assert_eq!(scheduler.stats().completed, 2);
}

#[test]
fn stats_render_matches_the_frozen_output_format() {
    // Byte-compared against tests/golden/mcp-scheduler-calls.jsonl and
    // tests/golden/scheduler-status.txt.
    let scheduler = Scheduler::new();
    assert_eq!(
        scheduler.stats().render(),
        "running=0 pending=0 completed=0 failed=0 cancelled=0 total_tasks=0"
    );

    scheduler.submit("t", TaskPriority::Normal, noop());
    scheduler.run_next().expect("runs");
    assert_eq!(
        scheduler.stats().render(),
        "running=0 pending=0 completed=1 failed=0 cancelled=0 total_tasks=1"
    );
}

#[test]
fn total_tasks_is_the_sum_of_the_other_counters() {
    let scheduler = Scheduler::new();
    scheduler.submit("done", TaskPriority::Normal, noop());
    scheduler.submit("fails", TaskPriority::Normal, failing("x"));
    let cancelled = scheduler.submit("cancelled", TaskPriority::Normal, noop());
    scheduler.submit("pending", TaskPriority::Normal, noop());
    scheduler.cancel(cancelled).expect("cancel");

    scheduler.run_next().expect("first ok");
    let _ = scheduler.run_next();

    let stats = scheduler.stats();
    assert_eq!(
        stats.total_tasks,
        stats.running + stats.pending + stats.completed + stats.failed + stats.cancelled
    );
    assert_eq!(stats.total_tasks, 4);
}

#[test]
fn a_scheduler_can_be_shared_across_threads() {
    let scheduler = Arc::new(Scheduler::new());
    let handles: Vec<_> = (0..4)
        .map(|n| {
            let scheduler = Arc::clone(&scheduler);
            std::thread::spawn(move || {
                for i in 0..25 {
                    scheduler.submit(
                        format!("t{n}-{i}"),
                        TaskPriority::Normal,
                        Box::new(|| Ok(())),
                    );
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("submitter panicked");
    }
    assert_eq!(scheduler.pending_count(), 100);
    scheduler.run_all().expect("all succeed");
    assert_eq!(scheduler.stats().completed, 100);
}

#[test]
fn memory_tracker_attachment_is_observable() {
    let scheduler = Scheduler::new();
    assert!(scheduler.memory_tracker().is_none());

    let tracker = Arc::new(MemoryTracker::new());
    let scheduler = scheduler.with_memory_tracker(Arc::clone(&tracker));
    assert!(scheduler.memory_tracker().is_some());
}
