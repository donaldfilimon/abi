const std = @import("std");
const testing = std.testing;

/// Helper to obtain a general purpose allocator for tests.
fn getTestAllocator() std.mem.Allocator {
    return testing.allocator;
}

// -----------------------------------------------------------------------------
// Performance module tests
// -----------------------------------------------------------------------------
test "Performance: Metric init, add label and deinit" {
    const perf = @import("../tools/performance.zig");
    const allocator = getTestAllocator();

    // Create a counter metric
    const metric = try perf.Metric.init(
        allocator,
        "test_counter",
        .{ .counter = 42 },
    );
    defer metric.deinit(allocator);

    // Add a label
    try metric.addLabel(allocator, "env", "unit");
    // Verify that the label exists
    const got = metric.labels.get("env") orelse return error.LabelNotFound;
    try testing.expectEqualSlices(u8, "unit", got);
}

// Test histogram recording and percentile calculation
test "Performance: Histogram record & percentile" {
    const perf = @import("../tools/performance.zig");
    var hist = perf.HistogramData{};
    // Record a set of values that span several buckets
    const values = [_]f64{ 0.0005, 0.003, 0.007, 0.02, 0.15, 0.8, 3.0, 12.0 };
    for (values) |v| hist.record(v);

    // Percentiles must be within bucket bounds
    const p50 = hist.percentile(0.5);
    try testing.expect(p50 >= 0.01 and p50 <= 0.02);

    const p90 = hist.percentile(0.9);
    try testing.expect(p90 >= 2.0 and p90 <= 5.0);
}

// Test timer start/stop and average duration
test "Performance: Timer start/stop and average" {
    const perf = @import("../tools/performance.zig");
    var timer = perf.TimerData{
        .start_time = 0,
    };
    timer.start();
    // Simulate some work by sleeping a few milliseconds
    std.time.sleep(2 * std.time.ns_per_ms);
    timer.stop();

    // After a single measurement, averageDuration should be non‑zero
    const avg = timer.averageDuration();
    try testing.expect(avg > 0.0);
}

// -----------------------------------------------------------------------------
// Basic code analyzer tests
// -----------------------------------------------------------------------------
test "BasicCodeAnalyzer: can be instantiated" {
    const analyzer = @import("../tools/basic_code_analyzer.zig");
    const allocator = getTestAllocator();

    // The analyzer exports a Config struct we can instantiate
    const cfg = analyzer.Config{
        .max_file_size = 1024 * 1024,
        .ignore_patterns = &[_][]const u8{},
    };
    // Instantiate the analyzer (constructor may be a function or struct init)
    const instance = try analyzer.init(allocator, cfg);
    defer instance.deinit();

    // Run a trivial analysis on empty source – should succeed
    const result = try instance.analyze("");
    try testing.expect(result.issues.len == 0);
}

// -----------------------------------------------------------------------------
// Simple code analyzer tests
// -----------------------------------------------------------------------------
test "SimpleCodeAnalyzer: basic usage" {
    const simple = @import("../tools/simple_code_analyzer.zig");
    const allocator = getTestAllocator();

    const instance = try simple.init(allocator);
    defer instance.deinit();

    // A simple snippet that should produce no warnings
    const src = "fn add(a: i32, b: i32) i32 { return a + b; }";
    const report = try instance.analyze(src);
    try testing.expect(report.warnings.len == 0);
}

// -----------------------------------------------------------------------------
// Memory tracker tests
// -----------------------------------------------------------------------------
test "MemoryTracker: allocation tracking" {
    const memtrack = @import("../tools/memory_tracker.zig");
    const allocator = getTestAllocator();

    var tracker = try memtrack.Tracker.init(allocator);
    defer tracker.deinit();

    // Allocate a small buffer through the tracker
    const buf = try tracker.alloc(u8, 32);
    defer tracker.free(buf);

    // The tracker should report at least one active allocation
    const stats = tracker.stats();
    try testing.expect(stats.active_allocations >= 1);
}

// -----------------------------------------------------------------------------
// Continuous monitor tests
// -----------------------------------------------------------------------------
test "ContinuousMonitor: start and stop" {
    const monitor = @import("../tools/continuous_monitor.zig");
    const allocator = getTestAllocator();

    var cm = try monitor.Monitor.init(allocator);
    defer cm.deinit();

    // Start monitoring – should not error
    try cm.start();

    // Stop monitoring – should also succeed
    cm.stop();
}
