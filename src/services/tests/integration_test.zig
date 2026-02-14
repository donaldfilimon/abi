//! Cross-Module Integration Tests
//!
//! Tests that verify correct interaction between major modules:
//! - Framework lifecycle: Init/shutdown ordering
//! - Feature registry: Enable/disable features
//! - Observability: Metrics collection
//!
//! These tests catch integration bugs that unit tests miss.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// ============================================================================
// Framework Lifecycle Integration
// ============================================================================

test "framework init/shutdown ordering" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework with default options
    var fw = try abi.initDefault(allocator);

    // Verify framework is in valid state
    const version = abi.version();
    try std.testing.expect(version.len > 0);

    // Shutdown should be clean
    fw.deinit();
}

test "framework reinitialize after shutdown" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // First initialization
    {
        var fw = try abi.initDefault(allocator);
        fw.deinit();
    }

    // Second initialization should work
    {
        var fw = try abi.initDefault(allocator);
        fw.deinit();
    }
}

test "framework with features enabled" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var fw = try abi.initDefault(allocator);
    defer fw.deinit();

    // Check feature flags match build options
    try std.testing.expectEqual(build_options.enable_gpu, fw.isEnabled(.gpu));
    try std.testing.expectEqual(build_options.enable_ai, fw.isEnabled(.ai));
    try std.testing.expectEqual(build_options.enable_web, fw.isEnabled(.web));
    try std.testing.expectEqual(build_options.enable_database, fw.isEnabled(.database));
    try std.testing.expectEqual(build_options.enable_network, fw.isEnabled(.network));
}

// ============================================================================
// Observability Integration
// ============================================================================

test "metrics collection" {
    // Create counter metric
    var counter = abi.observability.Counter{ .name = "test_ops_total" };

    // Simulate module activity
    counter.inc(1);
    counter.inc(5);

    try std.testing.expectEqual(@as(u64, 6), counter.get());
}

test "gauge metrics" {
    var gauge = abi.observability.Gauge{ .name = "test_active_connections" };

    gauge.set(10);
    gauge.inc();
    gauge.dec();

    try std.testing.expectEqual(@as(i64, 10), gauge.get());
}

// ============================================================================
// Runtime Integration
// ============================================================================

test "runtime context initialization" {
    var ctx = try abi.runtime.Context.init(std.testing.allocator);
    defer ctx.deinit();

    try std.testing.expect(ctx.initialized);
}

test "runtime engine lazy creation" {
    var ctx = try abi.runtime.Context.init(std.testing.allocator);
    defer ctx.deinit();

    try std.testing.expect(ctx.engine_ptr == null);
    const engine1 = try ctx.getEngine();
    try std.testing.expect(ctx.engine_ptr != null);
    const engine2 = try ctx.getEngine();
    try std.testing.expect(engine1 == engine2);
}

// ============================================================================
// Concurrency Primitives Integration
// ============================================================================

test "chase-lev deque basic operations" {
    var deque = try abi.runtime.ChaseLevDeque(u64).init(std.testing.allocator);
    defer deque.deinit();

    // Push items
    try deque.push(1);
    try deque.push(2);
    try deque.push(3);

    // Pop should return LIFO
    try std.testing.expectEqual(@as(?u64, 3), deque.pop());
    try std.testing.expectEqual(@as(?u64, 2), deque.pop());
    try std.testing.expectEqual(@as(?u64, 1), deque.pop());
    try std.testing.expectEqual(@as(?u64, null), deque.pop());
}

test "mpmc queue basic operations" {
    var queue = try abi.runtime.MpmcQueue(u64).init(std.testing.allocator, 64);
    defer queue.deinit();

    // Push items
    try queue.push(1);
    try queue.push(2);
    try queue.push(3);

    // Pop should return FIFO
    try std.testing.expectEqual(@as(?u64, 1), queue.pop());
    try std.testing.expectEqual(@as(?u64, 2), queue.pop());
    try std.testing.expectEqual(@as(?u64, 3), queue.pop());
    try std.testing.expectEqual(@as(?u64, null), queue.pop());
}

test "result cache basic operations" {
    var cache = try abi.runtime.ResultCache(u64, u64).init(std.testing.allocator, .{
        .max_entries = 100,
        .shard_count = 4,
    });
    defer cache.deinit();

    // Put and get
    try cache.put(1, 100);
    try cache.put(2, 200);

    try std.testing.expectEqual(@as(?u64, 100), cache.get(1));
    try std.testing.expectEqual(@as(?u64, 200), cache.get(2));
    try std.testing.expectEqual(@as(?u64, null), cache.get(3));
}

// ============================================================================
// Version and Build Info
// ============================================================================

test "version string is valid" {
    const version = abi.version();
    try std.testing.expect(version.len > 0);
    // Check it looks like a semver (X.Y.Z)
    var dot_count: usize = 0;
    for (version) |c| {
        if (c == '.') dot_count += 1;
    }
    try std.testing.expect(dot_count >= 2);
}
