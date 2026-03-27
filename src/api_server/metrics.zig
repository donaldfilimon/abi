//! Prometheus-compatible Metrics
//!
//! Atomic counters for request tracking, token generation, latency, and
//! active connections. Exports in Prometheus text format.

const std = @import("std");

pub const Metrics = struct {
    requests_total: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_success: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    requests_failed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tokens_generated: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    latency_sum_us: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    active_connections: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    pub fn recordRequest(self: *Metrics, success: bool) void {
        _ = self.requests_total.fetchAdd(1, .monotonic);
        if (success) {
            _ = self.requests_success.fetchAdd(1, .monotonic);
        } else {
            _ = self.requests_failed.fetchAdd(1, .monotonic);
        }
    }

    pub fn recordTokens(self: *Metrics, count: u64) void {
        _ = self.tokens_generated.fetchAdd(count, .monotonic);
    }

    pub fn recordLatency(self: *Metrics, microseconds: u64) void {
        _ = self.latency_sum_us.fetchAdd(microseconds, .monotonic);
    }

    pub fn connectionOpened(self: *Metrics) void {
        _ = self.active_connections.fetchAdd(1, .monotonic);
    }

    pub fn connectionClosed(self: *Metrics) void {
        _ = self.active_connections.fetchSub(1, .monotonic);
    }

    /// Render metrics in Prometheus exposition format.
    pub fn toPrometheus(self: *const Metrics, buffer: []u8) []const u8 {
        const total = self.requests_total.load(.monotonic);
        const success = self.requests_success.load(.monotonic);
        const failed = self.requests_failed.load(.monotonic);
        const tokens = self.tokens_generated.load(.monotonic);
        const latency = self.latency_sum_us.load(.monotonic);
        const active = self.active_connections.load(.monotonic);

        const avg_latency: f64 = if (total > 0)
            @as(f64, @floatFromInt(latency)) / @as(f64, @floatFromInt(total))
        else
            0;

        const written = std.fmt.bufPrint(buffer,
            \\# HELP abi_requests_total Total number of requests.
            \\# TYPE abi_requests_total counter
            \\abi_requests_total {d}
            \\# HELP abi_requests_success Successful requests.
            \\# TYPE abi_requests_success counter
            \\abi_requests_success {d}
            \\# HELP abi_requests_failed Failed requests.
            \\# TYPE abi_requests_failed counter
            \\abi_requests_failed {d}
            \\# HELP abi_tokens_generated Total tokens generated.
            \\# TYPE abi_tokens_generated counter
            \\abi_tokens_generated {d}
            \\# HELP abi_latency_avg_us Average request latency in microseconds.
            \\# TYPE abi_latency_avg_us gauge
            \\abi_latency_avg_us {d:.2}
            \\# HELP abi_active_connections Current active connections.
            \\# TYPE abi_active_connections gauge
            \\abi_active_connections {d}
            \\
        , .{ total, success, failed, tokens, avg_latency, active }) catch return buffer[0..0];

        return written;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "metrics record and export" {
    var m = Metrics{};

    m.recordRequest(true);
    m.recordRequest(true);
    m.recordRequest(false);
    m.recordTokens(100);
    m.recordLatency(5000);
    m.connectionOpened();

    try std.testing.expectEqual(@as(u64, 3), m.requests_total.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2), m.requests_success.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 1), m.requests_failed.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 100), m.tokens_generated.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 1), m.active_connections.load(.monotonic));

    var buf: [2048]u8 = undefined;
    const output = m.toPrometheus(&buf);
    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "abi_requests_total") != null);
}
