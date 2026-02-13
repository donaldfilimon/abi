//! Observability Edge Case and Module-Level Tests.

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");
const build_options = @import("build_options");

const observability = abi.observability;
const Histogram = observability.Histogram;
const FloatGauge = observability.FloatGauge;
const Gauge = observability.Gauge;
const Span = observability.Span;

// ============================================================================
// Edge Case Tests
// ============================================================================

test "observability: empty histogram operations" {
    const allocator = testing.allocator;
    const bounds = [_]u64{ 10, 50, 100 };
    var hist = try Histogram.init(allocator, "empty_hist", @constCast(&bounds));
    defer hist.deinit(allocator);

    // All buckets should be zero
    for (hist.buckets) |bucket| {
        try testing.expectEqual(@as(u64, 0), bucket);
    }
}

test "observability: very small float values" {
    var gauge = FloatGauge{ .name = "tiny_gauge" };

    gauge.set(std.math.floatMin(f64));
    try testing.expect(gauge.get() > 0);

    gauge.set(std.math.floatEps(f64));
    try testing.expect(gauge.get() > 0);
}

test "observability: very large float values" {
    var gauge = FloatGauge{ .name = "huge_gauge" };

    gauge.set(1e308);
    try testing.expectApproxEqAbs(@as(f64, 1e308), gauge.get(), 1e300);
}

test "observability: gauge extreme values" {
    var gauge = Gauge{ .name = "extreme_gauge" };

    gauge.set(std.math.maxInt(i64));
    try testing.expectEqual(std.math.maxInt(i64), gauge.get());

    gauge.set(std.math.minInt(i64));
    try testing.expectEqual(std.math.minInt(i64), gauge.get());
}

test "observability: span with many attributes" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "many-attributes", null, null, .internal);
    defer span.deinit();

    // Add many attributes
    for (0..100) |i| {
        var key_buf: [32]u8 = undefined;
        // Buffer is guaranteed large enough for "attr_" + up to 3 digits
        const key = std.fmt.bufPrint(&key_buf, "attr_{d}", .{i}) catch |err| {
            std.debug.panic("bufPrint failed unexpectedly: {}", .{err});
        };
        try span.setAttribute(key, .{ .int = @intCast(i) });
    }

    try testing.expectEqual(@as(usize, 100), span.attributes.items.len);
}

test "observability: span with many events" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "many-events", null, null, .internal);
    defer span.deinit();

    for (0..50) |i| {
        var name_buf: [32]u8 = undefined;
        // Buffer is guaranteed large enough for "event_" + up to 2 digits
        const name = std.fmt.bufPrint(&name_buf, "event_{d}", .{i}) catch |err| {
            std.debug.panic("bufPrint failed unexpectedly: {}", .{err});
        };
        try span.addEvent(name);
    }

    try testing.expectEqual(@as(usize, 50), span.events.items.len);
}

// ============================================================================
// Module State Tests
// ============================================================================

test "observability: module enabled check" {
    // This test verifies the module is enabled based on build options
    if (build_options.enable_profiling) {
        try testing.expect(observability.isEnabled());
    } else {
        try testing.expect(!observability.isEnabled());
    }
}

test "observability: create collector helper" {
    const allocator = testing.allocator;
    var collector = observability.createCollector(allocator);
    defer collector.deinit();

    const counter = try collector.registerCounter("test");
    counter.inc(1);
    try testing.expectEqual(@as(u64, 1), counter.get());
}
