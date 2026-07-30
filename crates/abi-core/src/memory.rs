//! Allocation accounting.
//!
//! Ported from `src/core/memory.zig`. This is *bookkeeping*, not an allocator:
//! callers report allocations and frees, and it maintains totals plus a record of
//! live allocations that `abi scheduler status` and `wdbx_stats` read back.
//!
//! The Zig version keyed records by raw `usize` address and had to resolve
//! address reuse by "newest match wins". Rust callers have no raw pointers to
//! hand over, so records are keyed by an opaque [`AllocationId`] issued at track
//! time. That removes the reuse ambiguity entirely — a freed id is never
//! reissued.
//!
//! Interior mutability plus a `Mutex` means a tracker can be shared as
//! `Arc<MemoryTracker>` between the scheduler and WDBX, which is how both the Zig
//! call sites used it.

use abi_foundation::time;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// A handle identifying one tracked allocation.
///
/// Opaque and never reused, so a stale id cannot silently refer to a different
/// allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId(u64);

/// A live tracked allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationRecord {
    /// The identifier returned when this allocation was tracked.
    pub id: AllocationId,
    /// Size in bytes.
    pub size: usize,
    /// Unix milliseconds at which it was tracked.
    pub timestamp: i64,
    /// Caller-supplied label, for attributing usage to a subsystem.
    pub tag: String,
}

/// A consistent read of every counter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Cumulative bytes ever tracked as allocated. Never decreases.
    pub total_allocated: usize,
    /// Cumulative bytes ever tracked as freed. Never decreases.
    pub total_freed: usize,
    /// The high-water mark of `current_usage`.
    pub peak_usage: usize,
    /// Bytes currently tracked as live.
    pub current_usage: usize,
    /// Number of live allocation records.
    pub record_count: usize,
}

impl MemorySnapshot {
    /// Bytes allocated but not freed.
    ///
    /// Saturating, so a caller that over-reports frees yields 0 rather than
    /// wrapping to an enormous number.
    #[must_use]
    pub const fn leaked_bytes(&self) -> usize {
        self.total_allocated.saturating_sub(self.total_freed)
    }
}

#[derive(Debug, Default)]
struct Counters {
    records: HashMap<AllocationId, AllocationRecord>,
    total_allocated: usize,
    total_freed: usize,
    peak_usage: usize,
    current_usage: usize,
}

impl Counters {
    fn add(&mut self, size: usize) {
        self.total_allocated = self.total_allocated.saturating_add(size);
        self.current_usage = self.current_usage.saturating_add(size);
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    fn remove(&mut self, size: usize) {
        self.total_freed = self.total_freed.saturating_add(size);
        // Saturating rather than wrapping: a double-free or an untracked free
        // would otherwise send current_usage to usize::MAX and make every
        // subsequent reading nonsense.
        self.current_usage = self.current_usage.saturating_sub(size);
    }

    fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            total_allocated: self.total_allocated,
            total_freed: self.total_freed,
            peak_usage: self.peak_usage,
            current_usage: self.current_usage,
            record_count: self.records.len(),
        }
    }
}

/// Tracks allocation totals and live records.
#[derive(Debug, Default)]
pub struct MemoryTracker {
    inner: Mutex<Counters>,
    next_id: AtomicU64,
}

impl MemoryTracker {
    /// A zeroed tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Counters::default()),
            next_id: AtomicU64::new(1),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Counters> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Record an allocation of `size` bytes attributed to `tag`.
    ///
    /// The returned id must be passed to [`MemoryTracker::track_free`] or
    /// [`MemoryTracker::track_resize`].
    pub fn track_alloc(&self, size: usize, tag: impl Into<String>) -> AllocationId {
        let id = AllocationId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let record = AllocationRecord {
            id,
            size,
            timestamp: time::unix_ms(),
            tag: tag.into(),
        };
        let mut inner = self.lock();
        inner.add(size);
        inner.records.insert(id, record);
        id
    }

    /// Record an allocation without keeping a per-allocation record.
    ///
    /// For hot paths where the tag allocation is not worth it — Zig's
    /// `trackAllocNoTag`. Totals move; `record_count` does not, so such an
    /// allocation cannot be freed by id. Use [`MemoryTracker::track_free_untagged`].
    pub fn track_alloc_untagged(&self, size: usize) {
        self.lock().add(size);
    }

    /// Record a free of an untagged allocation.
    pub fn track_free_untagged(&self, size: usize) {
        self.lock().remove(size);
    }

    /// Record the free of a tracked allocation.
    ///
    /// Returns the size that was freed, or `None` if `id` is unknown — which
    /// means either a double free or a free of an untagged allocation.
    pub fn track_free(&self, id: AllocationId) -> Option<usize> {
        let mut inner = self.lock();
        let record = inner.records.remove(&id)?;
        inner.remove(record.size);
        Some(record.size)
    }

    /// Record a reallocation, adjusting usage by the delta.
    ///
    /// Returns `false` if `id` is unknown, in which case nothing is recorded.
    pub fn track_resize(&self, id: AllocationId, new_size: usize) -> bool {
        let mut inner = self.lock();
        let Some(record) = inner.records.get_mut(&id) else {
            return false;
        };
        let old_size = record.size;
        record.size = new_size;
        if new_size >= old_size {
            inner.add(new_size - old_size);
        } else {
            inner.remove(old_size - new_size);
        }
        true
    }

    /// A consistent read of every counter.
    #[must_use]
    pub fn snapshot(&self) -> MemorySnapshot {
        self.lock().snapshot()
    }

    /// Live allocation records, ordered by id (so, by track time).
    #[must_use]
    pub fn records(&self) -> Vec<AllocationRecord> {
        let inner = self.lock();
        let mut records: Vec<_> = inner.records.values().cloned().collect();
        records.sort_unstable_by_key(|record| record.id);
        records
    }

    /// The number of live allocation records.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.lock().records.len()
    }

    /// Bytes currently tracked as live.
    #[must_use]
    pub fn current_usage(&self) -> usize {
        self.lock().current_usage
    }

    /// The high-water mark of live bytes.
    #[must_use]
    pub fn peak_usage(&self) -> usize {
        self.lock().peak_usage
    }

    /// Cumulative bytes ever tracked as allocated.
    ///
    /// Unlike [`MemoryTracker::current_usage`], this still reflects transient
    /// allocations that were freed within one operation, so it can prove a
    /// balanced hot-path allocation was actually tracked.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.lock().total_allocated
    }

    /// Cumulative bytes ever tracked as freed.
    #[must_use]
    pub fn total_freed(&self) -> usize {
        self.lock().total_freed
    }

    /// Bytes allocated but not freed.
    #[must_use]
    pub fn leaked_bytes(&self) -> usize {
        self.snapshot().leaked_bytes()
    }

    /// Drop every record and zero every counter.
    pub fn reset(&self) {
        *self.lock() = Counters::default();
    }
}

/// Render a byte count the way the status output does: `0B`, `1.5KB`, `2.0MB`.
///
/// `abi scheduler status` prints `peak=0B current=0B`, so the `B` suffix with no
/// space is part of the observed format.
#[must_use]
pub fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    #[expect(
        clippy::cast_precision_loss,
        reason = "display only; a byte count large enough to lose precision is already unreadable"
    )]
    let value = bytes as f64;
    if value >= GB {
        format!("{:.1}GB", value / GB)
    } else if value >= MB {
        format!("{:.1}MB", value / MB)
    } else if value >= KB {
        format!("{:.1}KB", value / KB)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_new_tracker_is_zeroed() {
        let tracker = MemoryTracker::new();
        assert_eq!(tracker.snapshot(), MemorySnapshot::default());
    }

    #[test]
    fn tracking_an_allocation_moves_totals_and_records() {
        let tracker = MemoryTracker::new();
        let id = tracker.track_alloc(1024, "wdbx.index");

        let snap = tracker.snapshot();
        assert_eq!(snap.total_allocated, 1024);
        assert_eq!(snap.current_usage, 1024);
        assert_eq!(snap.peak_usage, 1024);
        assert_eq!(snap.total_freed, 0);
        assert_eq!(snap.record_count, 1);

        let records = tracker.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, id);
        assert_eq!(records[0].size, 1024);
        assert_eq!(records[0].tag, "wdbx.index");
        assert!(records[0].timestamp > 0);
    }

    #[test]
    fn freeing_reduces_current_but_not_total() {
        let tracker = MemoryTracker::new();
        let id = tracker.track_alloc(512, "scratch");
        assert_eq!(tracker.track_free(id), Some(512));

        let snap = tracker.snapshot();
        assert_eq!(snap.total_allocated, 512, "cumulative must not decrease");
        assert_eq!(snap.total_freed, 512);
        assert_eq!(snap.current_usage, 0);
        assert_eq!(snap.peak_usage, 512, "peak must persist past the free");
        assert_eq!(snap.record_count, 0);
        assert_eq!(snap.leaked_bytes(), 0);
    }

    #[test]
    fn peak_records_the_high_water_mark() {
        let tracker = MemoryTracker::new();
        let a = tracker.track_alloc(100, "a");
        let b = tracker.track_alloc(200, "b");
        assert_eq!(tracker.peak_usage(), 300);

        tracker.track_free(b);
        tracker.track_free(a);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 300);
    }

    #[test]
    fn a_double_free_is_reported_and_does_not_double_count() {
        let tracker = MemoryTracker::new();
        let id = tracker.track_alloc(128, "once");
        assert_eq!(tracker.track_free(id), Some(128));
        // The id was consumed; freeing again must be detectable.
        assert_eq!(tracker.track_free(id), None);
        assert_eq!(tracker.total_freed(), 128, "must not double-count the free");
    }

    #[test]
    fn ids_are_never_reused() {
        // Zig keyed records by raw address, so a reused address needed a
        // "newest match wins" rule. Opaque ids remove the ambiguity.
        let tracker = MemoryTracker::new();
        let first = tracker.track_alloc(64, "a");
        tracker.track_free(first);
        let second = tracker.track_alloc(64, "b");
        assert_ne!(first, second);
        assert_eq!(tracker.track_free(first), None);
        assert_eq!(tracker.track_free(second), Some(64));
    }

    #[test]
    fn resize_upward_adds_the_delta() {
        let tracker = MemoryTracker::new();
        let id = tracker.track_alloc(100, "grow");
        assert!(tracker.track_resize(id, 250));

        let snap = tracker.snapshot();
        assert_eq!(snap.current_usage, 250);
        assert_eq!(snap.total_allocated, 250);
        assert_eq!(snap.total_freed, 0);
        assert_eq!(snap.record_count, 1);
    }

    #[test]
    fn resize_downward_frees_the_delta() {
        let tracker = MemoryTracker::new();
        let id = tracker.track_alloc(300, "shrink");
        assert!(tracker.track_resize(id, 100));

        let snap = tracker.snapshot();
        assert_eq!(snap.current_usage, 100);
        assert_eq!(snap.total_allocated, 300);
        assert_eq!(snap.total_freed, 200);
    }

    #[test]
    fn resizing_an_unknown_id_records_nothing() {
        let tracker = MemoryTracker::new();
        let id = tracker.track_alloc(10, "a");
        tracker.track_free(id);
        assert!(!tracker.track_resize(id, 999));
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn resize_then_free_uses_the_new_size() {
        let tracker = MemoryTracker::new();
        let id = tracker.track_alloc(100, "r");
        tracker.track_resize(id, 400);
        assert_eq!(tracker.track_free(id), Some(400));
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn untagged_tracking_moves_totals_without_a_record() {
        let tracker = MemoryTracker::new();
        tracker.track_alloc_untagged(4096);
        let snap = tracker.snapshot();
        assert_eq!(snap.current_usage, 4096);
        assert_eq!(snap.total_allocated, 4096);
        assert_eq!(snap.record_count, 0, "untagged keeps no record");

        tracker.track_free_untagged(4096);
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn over_freeing_saturates_instead_of_wrapping() {
        // Wrapping would put current_usage at usize::MAX and poison every
        // later reading.
        let tracker = MemoryTracker::new();
        tracker.track_alloc_untagged(100);
        tracker.track_free_untagged(500);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.leaked_bytes(), 0);
    }

    #[test]
    fn leaked_bytes_reports_the_unfreed_remainder() {
        let tracker = MemoryTracker::new();
        tracker.track_alloc(1000, "kept");
        let freed = tracker.track_alloc(500, "released");
        tracker.track_free(freed);
        assert_eq!(tracker.leaked_bytes(), 1000);
    }

    #[test]
    fn reset_zeroes_everything() {
        let tracker = MemoryTracker::new();
        tracker.track_alloc(100, "a");
        tracker.reset();
        assert_eq!(tracker.snapshot(), MemorySnapshot::default());
    }

    #[test]
    fn records_are_ordered_by_track_time() {
        let tracker = MemoryTracker::new();
        for tag in ["first", "second", "third"] {
            tracker.track_alloc(1, tag);
        }
        let tags: Vec<String> = tracker.records().into_iter().map(|r| r.tag).collect();
        assert_eq!(tags, ["first", "second", "third"]);
    }

    #[test]
    fn a_tracker_is_shareable_across_threads() {
        let tracker = Arc::new(MemoryTracker::new());
        let handles: Vec<_> = (0..8)
            .map(|n| {
                let tracker = Arc::clone(&tracker);
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        tracker.track_alloc(16, format!("worker{n}"));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("worker panicked");
        }
        let snap = tracker.snapshot();
        assert_eq!(snap.record_count, 400);
        assert_eq!(snap.total_allocated, 6400);
        assert_eq!(snap.current_usage, 6400);
    }

    #[test]
    fn format_bytes_matches_the_status_output_format() {
        // `abi scheduler status` prints `peak=0B current=0B`.
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(1024), "1.0KB");
        assert_eq!(format_bytes(1536), "1.5KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0GB");
    }
}
