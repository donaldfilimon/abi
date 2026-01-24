//! Event-Based Synchronization Primitive
//!
//! Provides non-blocking synchronization for GPU operations,
//! replacing polling-based dirty state checks with wait/signal semantics.

const std = @import("std");

/// Event-based synchronization primitive for GPU operations.
/// Replaces polling-based dirty state checks with wait/signal semantics.
pub const SyncEvent = struct {
    /// Completion state - true when event has been signaled.
    completed: std.atomic.Value(bool),

    /// Initialize a new sync event in incomplete state.
    pub fn init() SyncEvent {
        return .{
            .completed = std.atomic.Value(bool).init(false),
        };
    }

    /// Cleanup (no-op for this implementation).
    pub fn deinit(self: *SyncEvent) void {
        _ = self;
    }

    /// Record event completion (called by GPU callback or completion handler).
    /// This signals that the associated operation has completed.
    pub fn record(self: *SyncEvent) void {
        self.completed.store(true, .release);
    }

    /// Check if event has completed without blocking.
    /// Returns true if the event has been signaled via record().
    pub fn isComplete(self: *const SyncEvent) bool {
        return self.completed.load(.acquire);
    }

    /// Block until event completes using spin-wait.
    /// For production use with real GPU operations, this would use
    /// OS-level synchronization primitives or GPU event queries.
    pub fn wait(self: *SyncEvent) void {
        while (!self.completed.load(.acquire)) {
            // Spin-wait with pause hint for better power efficiency
            std.atomic.spinLoopHint();
        }
    }

    /// Block until event completes or timeout expires.
    /// Returns true if event completed, false if timeout.
    pub fn waitTimeout(self: *SyncEvent, timeout_ns: u64) bool {
        var timer = std.time.Timer.start() catch return self.isComplete();

        while (!self.completed.load(.acquire)) {
            if (timer.read() >= timeout_ns) {
                return false;
            }
            std.atomic.spinLoopHint();
        }
        return true;
    }

    /// Reset event to incomplete state for reuse.
    pub fn reset(self: *SyncEvent) void {
        self.completed.store(false, .release);
    }
};

// Tests
test "SyncEvent records and queries completion" {
    var event = SyncEvent.init();
    defer event.deinit();

    // Event should not be complete initially
    try std.testing.expect(!event.isComplete());

    // Record event (simulates GPU operation completion)
    event.record();

    // After recording, event should be complete
    try std.testing.expect(event.isComplete());
}

test "SyncEvent wait returns immediately when complete" {
    var event = SyncEvent.init();
    defer event.deinit();

    // Record completion first
    event.record();

    // Wait should return immediately
    event.wait();
    try std.testing.expect(event.isComplete());
}

test "SyncEvent reset clears completion state" {
    var event = SyncEvent.init();
    defer event.deinit();

    event.record();
    try std.testing.expect(event.isComplete());

    event.reset();
    try std.testing.expect(!event.isComplete());
}

test "SyncEvent waitTimeout returns false on timeout" {
    var event = SyncEvent.init();
    defer event.deinit();

    // Should timeout since event is never signaled
    const completed = event.waitTimeout(1_000_000); // 1ms timeout
    try std.testing.expect(!completed);
}

test "SyncEvent waitTimeout returns true when signaled" {
    var event = SyncEvent.init();
    defer event.deinit();

    event.record();

    const completed = event.waitTimeout(1_000_000_000); // 1s timeout
    try std.testing.expect(completed);
}
