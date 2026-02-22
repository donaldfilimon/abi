//! Observability Middleware
//!
//! Records per-request metrics: request count, latency, status codes, and
//! active request gauge. All counters use `@atomicRmw` for thread safety.
//!
//! This middleware is self-contained — it does NOT depend on the observability
//! feature module (which would create a circular import). Instead it maintains
//! simple in-memory counters and a latency histogram that can be scraped in
//! Prometheus exposition format via `formatPrometheus`.

const std = @import("std");

/// Number of latency histogram buckets.
pub const BUCKET_COUNT: usize = 8;

/// Upper bounds (in microseconds) for each histogram bucket.
/// Buckets: 0-100us, 100-500us, 500-1ms, 1-5ms, 5-50ms, 50-200ms, 200ms-1s, 1s+
pub const bucket_bounds_us: [BUCKET_COUNT]u64 = .{
    100,
    500,
    1_000,
    5_000,
    50_000,
    200_000,
    1_000_000,
    std.math.maxInt(u64),
};

/// Human-readable labels for histogram buckets (used in Prometheus output).
const bucket_labels: [BUCKET_COUNT][]const u8 = .{
    "100",
    "500",
    "1000",
    "5000",
    "50000",
    "200000",
    "1000000",
    "+Inf",
};

/// Per-request timing token returned by `processRequest`.
pub const RequestMetrics = struct {
    /// Monotonic start time (CLOCK_MONOTONIC, nanoseconds).
    start_ns: i128,
};

/// Point-in-time snapshot of all metrics (read atomically).
pub const MetricsSnapshot = struct {
    total_requests: u64,
    total_errors: u64,
    active_requests: u64,
    request_durations_us: [BUCKET_COUNT]u64,
    status_counts: [6]u64,
};

/// Thread-safe, lock-free HTTP metrics collector.
///
/// All fields are updated via `@atomicRmw` so no mutex is required.
pub const MetricsMiddleware = struct {
    /// Total number of requests processed.
    total_requests: u64,
    /// Total number of error responses (status >= 400).
    total_errors: u64,
    /// Number of requests currently in flight.
    active_requests: u64,
    /// Latency histogram buckets (microseconds).
    request_durations_us: [BUCKET_COUNT]u64,
    /// Per-status-class counters: [0]=1xx, [1]=2xx, [2]=3xx, [3]=4xx, [4]=5xx, [5]=other.
    status_counts: [6]u64,

    /// Creates a zero-initialized metrics middleware.
    pub fn init() MetricsMiddleware {
        return .{
            .total_requests = 0,
            .total_errors = 0,
            .active_requests = 0,
            .request_durations_us = .{0} ** BUCKET_COUNT,
            .status_counts = .{0} ** 6,
        };
    }

    /// Marks the start of a request. Returns a token that must be passed to
    /// `recordResponse` when the response is ready.
    pub fn processRequest(self: *MetricsMiddleware) RequestMetrics {
        _ = @atomicRmw(u64, &self.total_requests, .Add, 1, .monotonic);
        _ = @atomicRmw(u64, &self.active_requests, .Add, 1, .monotonic);
        return .{
            .start_ns = nowMonotonicNs(),
        };
    }

    /// Finalizes timing for a completed request and increments the
    /// appropriate histogram bucket and status-class counter.
    pub fn recordResponse(self: *MetricsMiddleware, metrics: RequestMetrics, status_code: u16) void {
        // Decrement active requests.
        _ = @atomicRmw(u64, &self.active_requests, .Sub, 1, .monotonic);

        // Compute elapsed microseconds.
        const elapsed_ns = nowMonotonicNs() - metrics.start_ns;
        const elapsed_us: u64 = if (elapsed_ns <= 0) 0 else @intCast(@divTrunc(elapsed_ns, 1_000));

        // Record latency in the appropriate histogram bucket.
        const bucket_idx = bucketIndex(elapsed_us);
        _ = @atomicRmw(u64, &self.request_durations_us[bucket_idx], .Add, 1, .monotonic);

        // Record status class.
        const class_idx = statusClassIndex(status_code);
        _ = @atomicRmw(u64, &self.status_counts[class_idx], .Add, 1, .monotonic);

        // Track errors (4xx and 5xx).
        if (status_code >= 400) {
            _ = @atomicRmw(u64, &self.total_errors, .Add, 1, .monotonic);
        }
    }

    /// Returns a consistent snapshot of all metrics.
    pub fn getSnapshot(self: *const MetricsMiddleware) MetricsSnapshot {
        var durations: [BUCKET_COUNT]u64 = undefined;
        for (0..BUCKET_COUNT) |i| {
            durations[i] = @atomicLoad(u64, &self.request_durations_us[i], .monotonic);
        }

        var statuses: [6]u64 = undefined;
        for (0..6) |i| {
            statuses[i] = @atomicLoad(u64, &self.status_counts[i], .monotonic);
        }

        return .{
            .total_requests = @atomicLoad(u64, &self.total_requests, .monotonic),
            .total_errors = @atomicLoad(u64, &self.total_errors, .monotonic),
            .active_requests = @atomicLoad(u64, &self.active_requests, .monotonic),
            .request_durations_us = durations,
            .status_counts = statuses,
        };
    }

    /// Formats all metrics in Prometheus exposition format.
    ///
    /// The caller owns the returned slice and must free it with `allocator`.
    pub fn formatPrometheus(self: *const MetricsMiddleware, allocator: std.mem.Allocator) ![]u8 {
        const snap = self.getSnapshot();

        var aw: std.Io.Writer.Allocating = .init(allocator);
        errdefer aw.deinit();

        const w = &aw.writer;

        // --- total_requests counter ---
        try w.writeAll("# HELP http_requests_total Total number of HTTP requests.\n");
        try w.writeAll("# TYPE http_requests_total counter\n");
        try w.print("http_requests_total {d}\n\n", .{snap.total_requests});

        // --- total_errors counter ---
        try w.writeAll("# HELP http_errors_total Total number of HTTP error responses (status >= 400).\n");
        try w.writeAll("# TYPE http_errors_total counter\n");
        try w.print("http_errors_total {d}\n\n", .{snap.total_errors});

        // --- active_requests gauge ---
        try w.writeAll("# HELP http_active_requests Number of in-flight HTTP requests.\n");
        try w.writeAll("# TYPE http_active_requests gauge\n");
        try w.print("http_active_requests {d}\n\n", .{snap.active_requests});

        // --- request duration histogram ---
        try w.writeAll("# HELP http_request_duration_us HTTP request latency in microseconds.\n");
        try w.writeAll("# TYPE http_request_duration_us histogram\n");
        var cumulative: u64 = 0;
        for (0..BUCKET_COUNT) |i| {
            cumulative += snap.request_durations_us[i];
            try w.print("http_request_duration_us_bucket{{le=\"{s}\"}} {d}\n", .{
                bucket_labels[i],
                cumulative,
            });
        }
        try w.print("http_request_duration_us_count {d}\n\n", .{snap.total_requests});

        // --- status class counters ---
        try w.writeAll("# HELP http_responses_total HTTP responses by status class.\n");
        try w.writeAll("# TYPE http_responses_total counter\n");
        const class_labels = [_][]const u8{ "1xx", "2xx", "3xx", "4xx", "5xx", "other" };
        for (class_labels, 0..) |label, i| {
            try w.print("http_responses_total{{class=\"{s}\"}} {d}\n", .{
                label,
                snap.status_counts[i],
            });
        }

        return aw.toOwnedSlice();
    }

    /// Resets all counters to zero. Useful in tests.
    pub fn reset(self: *MetricsMiddleware) void {
        @atomicStore(u64, &self.total_requests, 0, .monotonic);
        @atomicStore(u64, &self.total_errors, 0, .monotonic);
        @atomicStore(u64, &self.active_requests, 0, .monotonic);
        for (0..BUCKET_COUNT) |i| {
            @atomicStore(u64, &self.request_durations_us[i], 0, .monotonic);
        }
        for (0..6) |i| {
            @atomicStore(u64, &self.status_counts[i], 0, .monotonic);
        }
    }
};

// ============================================================================
// Helpers
// ============================================================================

/// Returns the histogram bucket index for a given latency (in microseconds).
fn bucketIndex(elapsed_us: u64) usize {
    for (bucket_bounds_us, 0..) |bound, i| {
        if (elapsed_us <= bound) return i;
    }
    return BUCKET_COUNT - 1;
}

/// Maps an HTTP status code to a status-class index (0–5).
fn statusClassIndex(status: u16) usize {
    return switch (status) {
        100...199 => 0,
        200...299 => 1,
        300...399 => 2,
        400...499 => 3,
        500...599 => 4,
        else => 5,
    };
}

/// Reads `CLOCK_MONOTONIC` via `std.c.clock_gettime` and returns nanoseconds.
fn nowMonotonicNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

// ============================================================================
// Tests
// ============================================================================

test "MetricsMiddleware init is zero" {
    const mw = MetricsMiddleware.init();
    const snap = mw.getSnapshot();

    try std.testing.expectEqual(@as(u64, 0), snap.total_requests);
    try std.testing.expectEqual(@as(u64, 0), snap.total_errors);
    try std.testing.expectEqual(@as(u64, 0), snap.active_requests);
    for (snap.request_durations_us) |v| {
        try std.testing.expectEqual(@as(u64, 0), v);
    }
    for (snap.status_counts) |v| {
        try std.testing.expectEqual(@as(u64, 0), v);
    }
}

test "processRequest increments total and active" {
    var mw = MetricsMiddleware.init();

    const m1 = mw.processRequest();
    try std.testing.expectEqual(@as(u64, 1), @atomicLoad(u64, &mw.total_requests, .monotonic));
    try std.testing.expectEqual(@as(u64, 1), @atomicLoad(u64, &mw.active_requests, .monotonic));

    const m2 = mw.processRequest();
    try std.testing.expectEqual(@as(u64, 2), @atomicLoad(u64, &mw.total_requests, .monotonic));
    try std.testing.expectEqual(@as(u64, 2), @atomicLoad(u64, &mw.active_requests, .monotonic));

    // Complete both to avoid leaking active count.
    mw.recordResponse(m1, 200);
    mw.recordResponse(m2, 200);
}

test "recordResponse decrements active and records status" {
    var mw = MetricsMiddleware.init();

    const m = mw.processRequest();
    mw.recordResponse(m, 200);

    const snap = mw.getSnapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.active_requests);
    try std.testing.expectEqual(@as(u64, 1), snap.status_counts[1]); // 2xx
    try std.testing.expectEqual(@as(u64, 0), snap.total_errors);
}

test "recordResponse tracks errors for 4xx" {
    var mw = MetricsMiddleware.init();

    const m = mw.processRequest();
    mw.recordResponse(m, 404);

    const snap = mw.getSnapshot();
    try std.testing.expectEqual(@as(u64, 1), snap.total_errors);
    try std.testing.expectEqual(@as(u64, 1), snap.status_counts[3]); // 4xx
}

test "recordResponse tracks errors for 5xx" {
    var mw = MetricsMiddleware.init();

    const m = mw.processRequest();
    mw.recordResponse(m, 503);

    const snap = mw.getSnapshot();
    try std.testing.expectEqual(@as(u64, 1), snap.total_errors);
    try std.testing.expectEqual(@as(u64, 1), snap.status_counts[4]); // 5xx
}

test "recordResponse does not count 2xx as error" {
    var mw = MetricsMiddleware.init();

    const m = mw.processRequest();
    mw.recordResponse(m, 201);

    try std.testing.expectEqual(@as(u64, 0), mw.getSnapshot().total_errors);
}

test "recordResponse does not count 3xx as error" {
    var mw = MetricsMiddleware.init();

    const m = mw.processRequest();
    mw.recordResponse(m, 301);

    try std.testing.expectEqual(@as(u64, 0), mw.getSnapshot().total_errors);
}

test "statusClassIndex maps correctly" {
    try std.testing.expectEqual(@as(usize, 0), statusClassIndex(100));
    try std.testing.expectEqual(@as(usize, 0), statusClassIndex(101));
    try std.testing.expectEqual(@as(usize, 1), statusClassIndex(200));
    try std.testing.expectEqual(@as(usize, 1), statusClassIndex(204));
    try std.testing.expectEqual(@as(usize, 2), statusClassIndex(301));
    try std.testing.expectEqual(@as(usize, 2), statusClassIndex(304));
    try std.testing.expectEqual(@as(usize, 3), statusClassIndex(400));
    try std.testing.expectEqual(@as(usize, 3), statusClassIndex(404));
    try std.testing.expectEqual(@as(usize, 3), statusClassIndex(429));
    try std.testing.expectEqual(@as(usize, 4), statusClassIndex(500));
    try std.testing.expectEqual(@as(usize, 4), statusClassIndex(503));
    try std.testing.expectEqual(@as(usize, 5), statusClassIndex(0));
    try std.testing.expectEqual(@as(usize, 5), statusClassIndex(600));
}

test "bucketIndex maps latencies to correct buckets" {
    // 0–100us → bucket 0
    try std.testing.expectEqual(@as(usize, 0), bucketIndex(0));
    try std.testing.expectEqual(@as(usize, 0), bucketIndex(50));
    try std.testing.expectEqual(@as(usize, 0), bucketIndex(100));
    // 101–500us → bucket 1
    try std.testing.expectEqual(@as(usize, 1), bucketIndex(101));
    try std.testing.expectEqual(@as(usize, 1), bucketIndex(500));
    // 501–1000us → bucket 2
    try std.testing.expectEqual(@as(usize, 2), bucketIndex(501));
    try std.testing.expectEqual(@as(usize, 2), bucketIndex(1_000));
    // 1001–5000us → bucket 3
    try std.testing.expectEqual(@as(usize, 3), bucketIndex(1_001));
    try std.testing.expectEqual(@as(usize, 3), bucketIndex(5_000));
    // 5001–50000us → bucket 4
    try std.testing.expectEqual(@as(usize, 4), bucketIndex(5_001));
    try std.testing.expectEqual(@as(usize, 4), bucketIndex(50_000));
    // 50001–200000us → bucket 5
    try std.testing.expectEqual(@as(usize, 5), bucketIndex(50_001));
    try std.testing.expectEqual(@as(usize, 5), bucketIndex(200_000));
    // 200001–1000000us → bucket 6
    try std.testing.expectEqual(@as(usize, 6), bucketIndex(200_001));
    try std.testing.expectEqual(@as(usize, 6), bucketIndex(1_000_000));
    // >1s → bucket 7 (overflow / +Inf)
    try std.testing.expectEqual(@as(usize, 7), bucketIndex(1_000_001));
}

test "histogram records durations" {
    var mw = MetricsMiddleware.init();

    // Process and immediately record — latency should be in the low buckets.
    const m = mw.processRequest();
    mw.recordResponse(m, 200);

    const snap = mw.getSnapshot();
    // At least one bucket must have been incremented.
    var total_buckets: u64 = 0;
    for (snap.request_durations_us) |v| {
        total_buckets += v;
    }
    try std.testing.expectEqual(@as(u64, 1), total_buckets);
}

test "getSnapshot reads consistent data" {
    var mw = MetricsMiddleware.init();

    // Simulate 5 requests: 2×200, 1×404, 1×500, 1×301
    const statuses = [_]u16{ 200, 200, 404, 500, 301 };
    for (statuses) |status| {
        const m = mw.processRequest();
        mw.recordResponse(m, status);
    }

    const snap = mw.getSnapshot();
    try std.testing.expectEqual(@as(u64, 5), snap.total_requests);
    try std.testing.expectEqual(@as(u64, 2), snap.total_errors); // 404 + 500
    try std.testing.expectEqual(@as(u64, 0), snap.active_requests);
    try std.testing.expectEqual(@as(u64, 2), snap.status_counts[1]); // 2xx
    try std.testing.expectEqual(@as(u64, 1), snap.status_counts[2]); // 3xx
    try std.testing.expectEqual(@as(u64, 1), snap.status_counts[3]); // 4xx
    try std.testing.expectEqual(@as(u64, 1), snap.status_counts[4]); // 5xx
}

test "formatPrometheus produces valid output" {
    var mw = MetricsMiddleware.init();
    const allocator = std.testing.allocator;

    const m1 = mw.processRequest();
    mw.recordResponse(m1, 200);
    const m2 = mw.processRequest();
    mw.recordResponse(m2, 500);

    const output = try mw.formatPrometheus(allocator);
    defer allocator.free(output);

    // Verify key Prometheus metric names are present.
    try std.testing.expect(std.mem.indexOf(u8, output, "http_requests_total 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "http_errors_total 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "http_active_requests 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "http_request_duration_us_bucket") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "http_responses_total") != null);
    // Verify HELP and TYPE annotations.
    try std.testing.expect(std.mem.indexOf(u8, output, "# HELP http_requests_total") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "# TYPE http_requests_total counter") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "# TYPE http_active_requests gauge") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "# TYPE http_request_duration_us histogram") != null);
}

test "reset zeroes all counters" {
    var mw = MetricsMiddleware.init();

    const m = mw.processRequest();
    mw.recordResponse(m, 500);

    mw.reset();

    const snap = mw.getSnapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.total_requests);
    try std.testing.expectEqual(@as(u64, 0), snap.total_errors);
    try std.testing.expectEqual(@as(u64, 0), snap.active_requests);
    for (snap.request_durations_us) |v| {
        try std.testing.expectEqual(@as(u64, 0), v);
    }
    for (snap.status_counts) |v| {
        try std.testing.expectEqual(@as(u64, 0), v);
    }
}

test "concurrent processRequest and recordResponse" {
    if (@import("builtin").single_threaded) return error.SkipZigTest;

    var mw = MetricsMiddleware.init();
    const num_threads = 4;
    const iterations = 500;

    const Worker = struct {
        fn run(metrics_mw: *MetricsMiddleware) void {
            for (0..iterations) |i| {
                const m = metrics_mw.processRequest();
                // Alternate status codes to exercise different paths.
                const status: u16 = if (i % 3 == 0) 500 else if (i % 3 == 1) 404 else 200;
                metrics_mw.recordResponse(m, status);
            }
        }
    };

    var threads: [num_threads]std.Thread = undefined;
    for (&threads) |*t| {
        t.* = std.Thread.spawn(.{}, Worker.run, .{&mw}) catch return error.SkipZigTest;
    }
    for (threads) |t| t.join();

    const snap = mw.getSnapshot();
    const expected_total: u64 = num_threads * iterations;
    try std.testing.expectEqual(expected_total, snap.total_requests);
    try std.testing.expectEqual(@as(u64, 0), snap.active_requests);

    // Verify status distribution sums to total.
    var status_sum: u64 = 0;
    for (snap.status_counts) |v| status_sum += v;
    try std.testing.expectEqual(expected_total, status_sum);
}

test {
    std.testing.refAllDecls(@This());
}
