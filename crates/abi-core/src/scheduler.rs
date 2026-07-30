//! A one-shot, priority-ordered task scheduler.
//!
//! Ported from `src/core/scheduler.zig`.
//!
//! ## Concurrency model
//!
//! **Tasks run synchronously on the caller's thread.** There is no worker pool
//! and no async runtime. `abi scheduler status` reports `mode=one-shot`, and the
//! Zig implementation matched that: `runNext` popped the highest-priority task
//! and invoked it inline. Nothing ever incremented `running_count` from another
//! thread — the observed `running=0` in every status output is not a coincidence,
//! it is the model.
//!
//! Recording this explicitly because `scheduler_stats` and `scheduler_info` are
//! two of the twelve frozen MCP tools, and introducing a runtime here would
//! change their output shape. A `Mutex` still guards the queue so a `Scheduler`
//! can be shared across threads, but concurrent *execution* is out of scope.
//!
//! ## A bug this port fixes
//!
//! Zig kept tasks in two containers — an owning `ArrayList` and a
//! `PriorityQueue` holding non-owning copies — which it then had to keep in sync
//! by hand. `cancel` mutated the list but left a stale copy in the heap, so
//! `runNext` had to purge lazily and `peek` had to bypass the heap entirely and
//! rescan the list (its comment says as much). A single `BinaryHeap` of pending
//! entries plus a map of records removes the duplication and the class of bug.

use crate::memory::MemoryTracker;
use abi_foundation::time;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex};

use crate::task::{TaskFn, TaskInfo, TaskPriority, TaskStatus};

/// Why a scheduler operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// No task with that id exists.
    NotFound {
        /// The id that was looked up.
        id: u64,
    },
    /// The task exists but is not in a state permitting the operation.
    InvalidState {
        /// The id that was operated on.
        id: u64,
        /// The state it was actually in.
        status: TaskStatus,
    },
    /// A task's closure returned an error.
    TaskFailed {
        /// The failing task's id.
        id: u64,
        /// The message it returned.
        message: String,
    },
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { id } => write!(f, "no task with id {id}"),
            Self::InvalidState { id, status } => {
                write!(f, "task {id} is {status}, not pending")
            }
            Self::TaskFailed { id, message } => write!(f, "task {id} failed: {message}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

/// Aggregate task counters.
///
/// Field names and their rendering are a contract: `abi scheduler status` and the
/// `scheduler_stats` MCP tool both emit
/// `running=N pending=N completed=N failed=N cancelled=N total_tasks=N`, in that
/// order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Tasks currently executing. Always 0 or 1 under the one-shot model.
    pub running: usize,
    /// Tasks queued but not started.
    pub pending: usize,
    /// Tasks that finished successfully.
    pub completed: usize,
    /// Tasks whose closure returned an error.
    pub failed: usize,
    /// Tasks cancelled before starting.
    pub cancelled: usize,
    /// The sum of every other counter.
    pub total_tasks: usize,
}

impl Stats {
    /// Render in the exact form the CLI and MCP tool emit.
    ///
    /// Callers append their own ` source=<tag>`, matching the Zig layout where
    /// the CLI emits `source=cli-scheduler-status` and MCP `source=mcp-server`.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "running={} pending={} completed={} failed={} cancelled={} total_tasks={}",
            self.running,
            self.pending,
            self.completed,
            self.failed,
            self.cancelled,
            self.total_tasks
        )
    }
}

/// A queued task awaiting execution.
struct Queued {
    id: u64,
    priority: TaskPriority,
    work: TaskFn,
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.id == other.id
    }
}

impl Eq for Queued {}

impl Ord for Queued {
    /// Highest priority first; ties broken by submission order (lowest id first).
    ///
    /// `BinaryHeap` is a max-heap, so the id comparison is reversed to make a
    /// *smaller* id compare greater and therefore pop sooner. Without this,
    /// equal-priority tasks would run in an unspecified order.
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Scheduler state behind one lock.
struct Inner {
    queue: BinaryHeap<Queued>,
    records: HashMap<u64, TaskInfo>,
    order: Vec<u64>,
    next_id: u64,
    running: usize,
    completed: usize,
    failed: usize,
    cancelled: usize,
}

impl Inner {
    fn stats(&self) -> Stats {
        let pending = self
            .records
            .values()
            .filter(|info| info.status == TaskStatus::Pending)
            .count();
        Stats {
            running: self.running,
            pending,
            completed: self.completed,
            failed: self.failed,
            cancelled: self.cancelled,
            total_tasks: pending + self.running + self.completed + self.failed + self.cancelled,
        }
    }
}

/// A one-shot, priority-ordered task scheduler.
pub struct Scheduler {
    inner: Mutex<Inner>,
    memory_tracker: Option<Arc<MemoryTracker>>,
}

impl Scheduler {
    /// An empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                queue: BinaryHeap::new(),
                records: HashMap::new(),
                order: Vec::new(),
                next_id: 1,
                running: 0,
                completed: 0,
                failed: 0,
                cancelled: 0,
            }),
            memory_tracker: None,
        }
    }

    /// Attach a memory tracker, so `abi scheduler status` reports
    /// `memory_tracker=attached` and its usage figures.
    #[must_use]
    pub fn with_memory_tracker(mut self, tracker: Arc<MemoryTracker>) -> Self {
        self.memory_tracker = Some(tracker);
        self
    }

    /// The attached memory tracker, if any.
    #[must_use]
    pub fn memory_tracker(&self) -> Option<&Arc<MemoryTracker>> {
        self.memory_tracker.as_ref()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Queue `work` and return its id.
    ///
    /// Ids start at 1 and increase monotonically, matching Zig's
    /// `next_id.init(1)`.
    pub fn submit(&self, name: impl Into<String>, priority: TaskPriority, work: TaskFn) -> u64 {
        let mut inner = self.lock();
        let id = inner.next_id;
        inner.next_id += 1;

        inner.records.insert(
            id,
            TaskInfo {
                id,
                name: name.into(),
                priority,
                status: TaskStatus::Pending,
                created_at: time::unix_ms(),
                started_at: 0,
                completed_at: 0,
                error_msg: None,
            },
        );
        inner.order.push(id);
        inner.queue.push(Queued { id, priority, work });
        id
    }

    /// Cancel a pending task.
    ///
    /// Fails with [`SchedulerError::InvalidState`] if it has already started or
    /// finished — cancellation is only meaningful before execution, since
    /// execution is synchronous.
    pub fn cancel(&self, id: u64) -> Result<(), SchedulerError> {
        let mut inner = self.lock();
        let Some(info) = inner.records.get_mut(&id) else {
            return Err(SchedulerError::NotFound { id });
        };
        if info.status != TaskStatus::Pending {
            return Err(SchedulerError::InvalidState {
                id,
                status: info.status,
            });
        }
        info.status = TaskStatus::Cancelled;
        info.completed_at = time::unix_ms();
        inner.cancelled += 1;
        // The queue entry stays; `run_next` skips any entry whose record is no
        // longer Pending. Unlike Zig this cannot desynchronize the counters,
        // because `pending` is derived from the records rather than tracked
        // separately.
        Ok(())
    }

    /// Run the highest-priority pending task.
    ///
    /// Returns the id that ran, or `None` when nothing is pending.
    pub fn run_next(&self) -> Result<Option<u64>, SchedulerError> {
        let (id, work) = {
            let mut inner = self.lock();
            let mut picked = None;
            // Skip entries whose record is no longer pending (cancelled).
            while let Some(entry) = inner.queue.pop() {
                let still_pending = inner
                    .records
                    .get(&entry.id)
                    .is_some_and(|info| info.status == TaskStatus::Pending);
                if still_pending {
                    picked = Some(entry);
                    break;
                }
            }
            let Some(entry) = picked else {
                return Ok(None);
            };
            if let Some(info) = inner.records.get_mut(&entry.id) {
                info.status = TaskStatus::Running;
                info.started_at = time::unix_ms();
            }
            inner.running += 1;
            (entry.id, entry.work)
        };

        // Run with the lock released, so a task may re-enter the scheduler
        // (submit, or read stats) without deadlocking.
        let outcome = work();

        let mut inner = self.lock();
        inner.running -= 1;
        let finished_at = time::unix_ms();
        match outcome {
            Ok(()) => {
                if let Some(info) = inner.records.get_mut(&id) {
                    info.status = TaskStatus::Completed;
                    info.completed_at = finished_at;
                }
                inner.completed += 1;
                Ok(Some(id))
            }
            Err(message) => {
                if let Some(info) = inner.records.get_mut(&id) {
                    info.status = TaskStatus::Failed;
                    info.completed_at = finished_at;
                    info.error_msg = Some(message.clone());
                }
                inner.failed += 1;
                Err(SchedulerError::TaskFailed { id, message })
            }
        }
    }

    /// Drain the queue, running every pending task.
    ///
    /// A failing task never aborts the batch: each failure is recorded on its own
    /// task and the drain continues, so work queued behind a failure is not
    /// stranded. The *first* error is returned once the queue is empty. This
    /// matches the Zig `runAll` contract, which documented the same behaviour.
    pub fn run_all(&self) -> Result<(), SchedulerError> {
        let mut first_error = None;
        loop {
            match self.run_next() {
                Ok(None) => break,
                Ok(Some(_)) => {}
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// A snapshot of the task with `id`.
    #[must_use]
    pub fn task(&self, id: u64) -> Option<TaskInfo> {
        self.lock().records.get(&id).cloned()
    }

    /// Every task, in submission order.
    #[must_use]
    pub fn tasks(&self) -> Vec<TaskInfo> {
        let inner = self.lock();
        inner
            .order
            .iter()
            .filter_map(|id| inner.records.get(id).cloned())
            .collect()
    }

    /// The highest-priority pending task, without running it.
    #[must_use]
    pub fn peek(&self) -> Option<TaskInfo> {
        let inner = self.lock();
        inner
            .queue
            .iter()
            .filter_map(|entry| inner.records.get(&entry.id))
            .filter(|info| info.status == TaskStatus::Pending)
            .max_by(|a, b| a.priority.cmp(&b.priority).then_with(|| b.id.cmp(&a.id)))
            .cloned()
    }

    /// Whether nothing is pending.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stats().pending == 0
    }

    /// Aggregate counters.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.lock().stats()
    }

    /// Number of pending tasks.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.stats().pending
    }

    /// Number of completed tasks.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.lock().completed
    }

    /// Number of failed tasks.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.lock().failed
    }

    /// Number of cancelled tasks.
    #[must_use]
    pub fn cancelled_count(&self) -> usize {
        self.lock().cancelled
    }

    /// Total tasks tracked across every status.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.stats().total_tasks
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Scheduler {
    /// Reports the counters rather than the queue contents.
    ///
    /// Cannot be derived: [`TaskFn`] is a boxed closure and has no `Debug`.
    /// `finish_non_exhaustive` marks the omission explicitly — the queued
    /// closures are genuinely unprintable, not forgotten.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("stats", &self.stats())
            .field("memory_tracker", &self.memory_tracker.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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
        assert!(scheduler.is_empty());
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
}
