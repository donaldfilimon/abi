const std = @import("std");

/// Latency histogram with 7 log-scale buckets.
pub const LatencyHistogram = struct {
    // Buckets: <1ms, <5ms, <10ms, <50ms, <100ms, <500ms, >=500ms
    buckets: [7]u64 = [_]u64{0} ** 7,
    total_ns: u128 = 0,
    count: u64 = 0,

    pub fn record(self: *LatencyHistogram, latency_ns: u64) void {
        const ms = latency_ns / std.time.ns_per_ms;
        const bucket_idx: usize = if (ms < 1) 0 else if (ms < 5) 1 else if (ms < 10) 2 else if (ms < 50) 3 else if (ms < 100) 4 else if (ms < 500) 5 else 6;
        self.buckets[bucket_idx] += 1;
        self.total_ns += latency_ns;
        self.count += 1;
    }

    pub fn avgMs(self: *const LatencyHistogram) u64 {
        if (self.count == 0) return 0;
        return @intCast(self.total_ns / self.count / std.time.ns_per_ms);
    }
};
