//! One-shot scheduler CLI adapter.

use std::sync::Arc;

use abi_core::{MemoryTracker, Scheduler, TaskPriority};

use crate::app::Outcome;

const USAGE: &str = "usage: abi scheduler status";

fn usage() -> Outcome {
    Outcome::stderr(format!("error: {USAGE}\n"), 2)
}

fn status() -> Outcome {
    let tracker = Arc::new(MemoryTracker::new());
    let scheduler = Scheduler::new().with_memory_tracker(Arc::clone(&tracker));
    scheduler.submit(
        "scheduler-status-probe",
        TaskPriority::Low,
        Box::new(|| Ok(())),
    );
    if let Err(detail) = scheduler.run_all() {
        return Outcome::stderr(format!("scheduler status failed: {detail}\n"), 1);
    }

    let stats = scheduler.stats();
    let memory = tracker.snapshot();
    let report = format!(
        "scheduler status\nsource=cli-scheduler-status\nmode=one-shot\n{}\nmemory_tracker=attached peak={}B current={}B records={}\n{}",
        stats.render(),
        memory.peak_usage,
        memory.current_usage,
        memory.record_count,
        abi_telemetry::write_text()
    );
    Outcome::stderr(report, 0)
}

pub(crate) fn run(args: &[String]) -> Outcome {
    match args {
        [command] if command == "status" => status(),
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn status_matches_the_frozen_one_shot_shape() {
        abi_telemetry::reset();
        let outcome = run(&strings(&["status"]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert_eq!(
            outcome.stderr,
            include_str!("../../../tests/golden/scheduler-status.txt")
        );
        abi_telemetry::reset();
    }

    #[test]
    fn invalid_invocations_are_usage_errors() {
        assert_eq!(run(&[]), usage());
        assert_eq!(run(&strings(&["unknown"])), usage());
    }
}
