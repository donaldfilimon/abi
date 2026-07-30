//! Bounded, process-wide telemetry counters.
//!
//! This is the Rust successor to `src/features/telemetry/mod.zig`. Counter
//! insertion order is retained because the CLI Prometheus exposition is a
//! captured compatibility surface.

pub mod metrics;

pub use metrics::{GaugeSnapshot, Metrics};

use std::{
    fmt::Write as _,
    sync::{Mutex, OnceLock},
};

/// Maximum distinct counter names retained process-wide.
pub const SLOT_CAPACITY: usize = 128;
/// Maximum bytes retained in one counter name.
pub const NAME_CAPACITY: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Counter {
    name: String,
    value: u64,
}

#[derive(Debug, Default)]
struct Table {
    counters: Vec<Counter>,
    total: u64,
    dropped: u64,
}

impl Table {
    fn increment(&mut self, name: &str, delta: u64) {
        if name.is_empty() || delta == 0 {
            return;
        }
        if name.len() > NAME_CAPACITY {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        if let Some(counter) = self
            .counters
            .iter_mut()
            .find(|counter| counter.name == name)
        {
            counter.value = counter.value.saturating_add(delta);
            self.total = self.total.saturating_add(delta);
            return;
        }
        if self.counters.len() == SLOT_CAPACITY {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        self.counters.push(Counter {
            name: name.to_owned(),
            value: delta,
        });
        self.total = self.total.saturating_add(delta);
    }
}

fn table() -> &'static Mutex<Table> {
    static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(Table::default()))
}

fn lock() -> std::sync::MutexGuard<'static, Table> {
    table()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Aggregate table health.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Summary {
    /// Sum of all accepted deltas.
    pub total: u64,
    /// Number of retained counter names.
    pub distinct: usize,
    /// Number of rejected overlong/full-table events.
    pub dropped: u64,
}

/// Owned point-in-time counter value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterSnapshot {
    /// Original counter name.
    pub name: String,
    /// Accumulated value.
    pub value: u64,
}

/// Telemetry is always enabled in the nightly Rust rewrite.
#[must_use]
pub const fn is_enabled() -> bool {
    true
}

/// Record one named event.
pub fn record(name: &str) {
    increment(name, 1);
}

/// Add `delta` to a named counter.
pub fn increment(name: &str, delta: u64) {
    lock().increment(name, delta);
}

/// Current value of `name`, or zero when absent.
#[must_use]
pub fn counter_value(name: &str) -> u64 {
    lock()
        .counters
        .iter()
        .find(|counter| counter.name == name)
        .map_or(0, |counter| counter.value)
}

/// Sum of all retained counter deltas.
#[must_use]
pub fn total_events() -> u64 {
    lock().total
}

/// Number of distinct retained names.
#[must_use]
pub fn distinct_counters() -> usize {
    lock().counters.len()
}

/// Number of rejected events.
#[must_use]
pub fn dropped_events() -> u64 {
    lock().dropped
}

/// Read aggregate table health atomically.
#[must_use]
pub fn summary() -> Summary {
    let table = lock();
    Summary {
        total: table.total,
        distinct: table.counters.len(),
        dropped: table.dropped,
    }
}

/// Copy retained counters in first-seen order.
#[must_use]
pub fn snapshot() -> Vec<CounterSnapshot> {
    lock()
        .counters
        .iter()
        .map(|counter| CounterSnapshot {
            name: counter.name.clone(),
            value: counter.value,
        })
        .collect()
}

fn sanitized_name(name: &str) -> String {
    let mut output = String::with_capacity(name.len() + 1);
    if name.is_empty() || name.as_bytes()[0].is_ascii_digit() {
        output.push('_');
    }
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':') {
            output.push(char::from(byte));
        } else {
            output.push('_');
        }
    }
    output
}

/// Render Prometheus text exposition in the frozen Zig order.
#[must_use]
pub fn write_text() -> String {
    let table = lock();
    let mut output = String::from("# abi telemetry (prometheus text exposition)\n");
    writeln!(
        output,
        "# TYPE abi_telemetry_events_total counter\nabi_telemetry_events_total {}",
        table.total
    )
    .expect("writing to a String cannot fail");
    writeln!(
        output,
        "# TYPE abi_telemetry_distinct_counters gauge\nabi_telemetry_distinct_counters {}",
        table.counters.len()
    )
    .expect("writing to a String cannot fail");
    writeln!(
        output,
        "# TYPE abi_telemetry_dropped_events_total counter\nabi_telemetry_dropped_events_total {}",
        table.dropped
    )
    .expect("writing to a String cannot fail");
    for counter in &table.counters {
        writeln!(
            output,
            "{} {}",
            sanitized_name(&counter.name),
            counter.value
        )
        .expect("writing to a String cannot fail");
    }
    output
}

/// Clear every counter. Primarily used for process/test initialization.
pub fn reset() {
    *lock() = Table::default();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn counters_accumulate_and_render_in_insertion_order() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        reset();
        record("scheduler.tasks.submitted");
        increment("scheduler.tasks.completed", 2);
        assert_eq!(counter_value("scheduler.tasks.submitted"), 1);
        assert_eq!(
            summary(),
            Summary {
                total: 3,
                distinct: 2,
                dropped: 0
            }
        );
        let text = write_text();
        assert!(text.contains("abi_telemetry_events_total 3"));
        assert!(
            text.find("scheduler_tasks_submitted 1") < text.find("scheduler_tasks_completed 2")
        );
        reset();
    }

    #[test]
    fn invalid_and_excess_names_are_observable() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        reset();
        record(&"x".repeat(NAME_CAPACITY + 1));
        for index in 0..SLOT_CAPACITY {
            record(&format!("counter_{index}"));
        }
        record("one_too_many");
        assert_eq!(dropped_events(), 2);
        assert_eq!(distinct_counters(), SLOT_CAPACITY);
        reset();
    }

    #[test]
    fn prometheus_names_are_sanitized_without_colliding_by_truncation() {
        assert_eq!(sanitized_name("3d.render.count"), "_3d_render_count");
        assert_eq!(sanitized_name("plugin:loaded"), "plugin:loaded");
        assert_eq!(sanitized_name("é"), "__");
    }
}
