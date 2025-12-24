//! Concurrent Data Structures
//!
//! This module provides lock-free and thread-safe data structures
//! for high-performance concurrent access patterns.

const std = @import("std");
const Atomic = std.atomic.Value;

pub const LockFreeError = error{
    OutOfMemory,
    CapacityExceeded,
    InvalidOperation,
    Timeout,
};

pub const LockFreeStats = struct {
    operations: u64 = 0,
    successful_operations: u64 = 0,
    failed_operations: u64 = 0,
    average_latency_ns: u64 = 0,
    max_latency_ns: u64 = 0,
    min_latency_ns: u64 = std.math.maxInt(u64),
    
    pub fn recordOperation(self: *LockFreeStats, success: bool, latency_ns: u64) void {
        self.operations += 1;
        if (success) {
            self.successful_operations += 1;
        } else {
            self.failed_operations += 1;
        }
        self.max_latency_ns = @max(self.max_latency_ns, latency_ns);
        self.min_latency_ns = @min(self.min_latency_ns, latency_ns);
        if (self.operations == 1) {
            self.average_latency_ns = latency_ns;
        } else {
            self.average_latency_ns = (self.average_latency_ns + latency_ns) / 2;
        }
    }
    
    pub fn successRate(self: *const LockFreeStats) f32 {
        if (self.operations == 0) return 0.0;
        return @as(f32, @floatFromInt(self.successful_operations)) / @as(f32, @floatFromInt(self.operations));
    }
};

test "LockFreeStats" {
    var stats = LockFreeStats{};
    stats.recordOperation(true, 100);
    try std.testing.expect(stats.operations == 1);
    try std.testing.expect(stats.successful_operations == 1);
}
