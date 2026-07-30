//! Task identity, priority, and lifecycle state.
//!
//! Ported from `src/core/task.zig`. The Zig `Task` carried a raw
//! `*const fn (?*anyopaque) anyerror!void` plus an untyped `ctx` pointer, which
//! is how a language without closures passes work around. Rust uses a boxed
//! closure, so the `ctx` field disappears — captured state lives in the closure.

use std::fmt;

/// Where a task is in its lifecycle.
///
/// The names are load-bearing: `abi scheduler status` and the `scheduler_stats`
/// MCP tool print one counter per variant, and those key names are a contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskStatus {
    /// Queued, not yet started.
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Cancelled before it started.
    Cancelled,
}

impl TaskStatus {
    /// The lowercase counter name used in status output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether this is an end state.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Scheduling priority. Higher runs first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum TaskPriority {
    /// Lowest.
    Low = 0,
    /// The default.
    #[default]
    Normal = 1,
    /// Above normal.
    High = 2,
    /// Highest.
    Critical = 3,
}

impl TaskPriority {
    /// The lowercase name used in status output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    /// Parse a priority by name, case-insensitively.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "normal" => Some(Self::Normal),
            "high" => Some(Self::High),
            "critical" => Some(Self::Critical),
            _ => None,
        }
    }
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The work a task performs.
///
/// `FnOnce` because a task runs at most once, and `Send` so a scheduler can be
/// moved across threads even though execution itself is synchronous.
pub type TaskFn = Box<dyn FnOnce() -> Result<(), String> + Send>;

/// An observable snapshot of a task.
///
/// Deliberately excludes the closure, so it is `Clone` and can outlive the
/// scheduler. Zig's `getTask` returned a `Task` whose `name`/`error_msg` slices
/// were still borrowed from the scheduler — a footgun its own doc comment had to
/// warn about. Here the strings are owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskInfo {
    /// Scheduler-assigned identifier, unique and starting at 1.
    pub id: u64,
    /// Caller-supplied label.
    pub name: String,
    /// Scheduling priority.
    pub priority: TaskPriority,
    /// Current lifecycle state.
    pub status: TaskStatus,
    /// Unix milliseconds at submission.
    pub created_at: i64,
    /// Unix milliseconds when execution began, or 0 if it has not.
    pub started_at: i64,
    /// Unix milliseconds when execution ended, or 0 if it has not.
    pub completed_at: i64,
    /// Failure message, present only when `status` is [`TaskStatus::Failed`].
    pub error_msg: Option<String>,
}

impl TaskInfo {
    /// Milliseconds spent running, if the task has both started and finished.
    #[must_use]
    pub fn duration_ms(&self) -> Option<i64> {
        if self.started_at == 0 || self.completed_at == 0 {
            return None;
        }
        Some(self.completed_at - self.started_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_names_match_the_status_output_contract() {
        // These strings appear in `abi scheduler status` and scheduler_stats.
        assert_eq!(TaskStatus::Pending.name(), "pending");
        assert_eq!(TaskStatus::Running.name(), "running");
        assert_eq!(TaskStatus::Completed.name(), "completed");
        assert_eq!(TaskStatus::Failed.name(), "failed");
        assert_eq!(TaskStatus::Cancelled.name(), "cancelled");
    }

    #[test]
    fn terminal_states_are_exactly_the_finished_ones() {
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::Running.is_terminal());
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(TaskStatus::Cancelled.is_terminal());
    }

    #[test]
    fn priority_orders_low_to_critical() {
        assert!(TaskPriority::Low < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::Critical);
    }

    #[test]
    fn priority_discriminants_match_the_zig_values() {
        assert_eq!(TaskPriority::Low as u8, 0);
        assert_eq!(TaskPriority::Normal as u8, 1);
        assert_eq!(TaskPriority::High as u8, 2);
        assert_eq!(TaskPriority::Critical as u8, 3);
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(TaskPriority::default(), TaskPriority::Normal);
    }

    #[test]
    fn priority_parses_by_name() {
        assert_eq!(TaskPriority::parse("low"), Some(TaskPriority::Low));
        assert_eq!(
            TaskPriority::parse("CRITICAL"),
            Some(TaskPriority::Critical)
        );
        assert_eq!(TaskPriority::parse("urgent"), None);
    }

    #[test]
    fn duration_needs_both_timestamps() {
        let mut info = TaskInfo {
            id: 1,
            name: "t".to_string(),
            priority: TaskPriority::Normal,
            status: TaskStatus::Pending,
            created_at: 100,
            started_at: 0,
            completed_at: 0,
            error_msg: None,
        };
        assert_eq!(info.duration_ms(), None);

        info.started_at = 110;
        assert_eq!(info.duration_ms(), None);

        info.completed_at = 160;
        assert_eq!(info.duration_ms(), Some(50));
    }
}
