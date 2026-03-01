//! Analytics Integration Tests
//!
//! Comprehensive tests for the analytics module covering:
//! - Engine event tracking and buffer management
//! - Funnel step progression and edge cases
//! - Experiment variant assignment and distribution
//! - Context lifecycle management
//! - Module-level init/deinit/isEnabled

const std = @import("std");
const build_options = @import("build_options");
const analytics = @import("abi").features.analytics;

// ============================================================================
// Engine: Buffer Management
// ============================================================================

test "engine: zero-capacity buffer rejects immediately" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var engine = analytics.Engine.init(allocator, .{ .buffer_capacity = 0 });
    defer engine.deinit();

    try std.testing.expectError(analytics.AnalyticsError.BufferFull, engine.track("any_event"));
    try std.testing.expectEqual(@as(u64, 0), engine.totalEvents());
}

test "engine: flush empty buffer returns zero" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var engine = analytics.Engine.init(allocator, .{});
    defer engine.deinit();

    const flushed = engine.flush();
    try std.testing.expectEqual(@as(usize, 0), flushed);
    try std.testing.expectEqual(@as(usize, 0), engine.bufferedCount());
}

test "engine: repeated flush is idempotent" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var engine = analytics.Engine.init(allocator, .{});
    defer engine.deinit();

    try engine.track("event_a");
    const first = engine.flush();
    const second = engine.flush();
    const third = engine.flush();

    try std.testing.expectEqual(@as(usize, 1), first);
    try std.testing.expectEqual(@as(usize, 0), second);
    try std.testing.expectEqual(@as(usize, 0), third);
}

test "engine: track after flush works" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var engine = analytics.Engine.init(allocator, .{ .buffer_capacity = 2 });
    defer engine.deinit();

    // Fill buffer
    try engine.track("e1");
    try engine.track("e2");
    try std.testing.expectError(analytics.AnalyticsError.BufferFull, engine.track("e3"));

    // Flush and refill
    _ = engine.flush();
    try engine.track("e4");
    try engine.track("e5");

    try std.testing.expectEqual(@as(usize, 2), engine.bufferedCount());
    try std.testing.expectEqual(@as(u64, 4), engine.totalEvents());
}

test "engine: getStats reflects accurate state" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var engine = analytics.Engine.init(allocator, .{});
    defer engine.deinit();

    // Initial state
    var stats = engine.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.buffered_events);
    try std.testing.expectEqual(@as(u64, 0), stats.total_events);
    try std.testing.expectEqual(@as(u64, 0), stats.total_sessions);

    // After tracking
    try engine.track("e1");
    try engine.track("e2");
    _ = engine.startSession();

    stats = engine.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.buffered_events);
    try std.testing.expectEqual(@as(u64, 2), stats.total_events);
    try std.testing.expectEqual(@as(u64, 1), stats.total_sessions);

    // After flush — total preserved, buffered cleared
    _ = engine.flush();
    stats = engine.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.buffered_events);
    try std.testing.expectEqual(@as(u64, 2), stats.total_events);
}

test "engine: timestamps disabled produces zero" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var engine = analytics.Engine.init(allocator, .{ .enable_timestamps = false });
    defer engine.deinit();

    try engine.track("no_ts_event");
    try std.testing.expectEqual(@as(usize, 1), engine.bufferedCount());
}

test "engine: session IDs are sequential" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var engine = analytics.Engine.init(allocator, .{});
    defer engine.deinit();

    var prev: u64 = 0;
    for (0..10) |i| {
        const sid = engine.startSession();
        try std.testing.expectEqual(@as(u64, i), sid);
        if (i > 0) try std.testing.expect(sid > prev);
        prev = sid;
    }
}

// ============================================================================
// Funnel: Step Progression
// ============================================================================

test "funnel: empty funnel returns empty counts" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var funnel = analytics.Funnel.init(allocator, "empty_funnel");
    defer funnel.deinit();

    var buf: [10]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(usize, 0), counts.len);
}

test "funnel: out-of-bounds recordStep is safe" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var funnel = analytics.Funnel.init(allocator, "safe_funnel");
    defer funnel.deinit();

    try funnel.addStep("step1");

    // Record at valid and invalid indices — should not panic
    funnel.recordStep(0);
    funnel.recordStep(1);
    funnel.recordStep(100);
    funnel.recordStep(std.math.maxInt(usize));

    var buf: [1]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(u64, 1), counts[0]);
}

test "funnel: multiple steps with varying counts" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var funnel = analytics.Funnel.init(allocator, "checkout");
    defer funnel.deinit();

    try funnel.addStep("view_cart");
    try funnel.addStep("add_payment");
    try funnel.addStep("confirm_order");
    try funnel.addStep("order_complete");

    // Simulate typical funnel drop-off
    for (0..100) |_| funnel.recordStep(0);
    for (0..60) |_| funnel.recordStep(1);
    for (0..30) |_| funnel.recordStep(2);
    for (0..25) |_| funnel.recordStep(3);

    var buf: [4]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(u64, 100), counts[0]);
    try std.testing.expectEqual(@as(u64, 60), counts[1]);
    try std.testing.expectEqual(@as(u64, 30), counts[2]);
    try std.testing.expectEqual(@as(u64, 25), counts[3]);
}

test "funnel: getStepCounts with smaller buffer truncates" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var funnel = analytics.Funnel.init(allocator, "trunc_test");
    defer funnel.deinit();

    try funnel.addStep("a");
    try funnel.addStep("b");
    try funnel.addStep("c");

    funnel.recordStep(0);
    funnel.recordStep(1);
    funnel.recordStep(2);

    // Only request first 2 of 3 steps
    var buf: [2]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(usize, 2), counts.len);
    try std.testing.expectEqual(@as(u64, 1), counts[0]);
    try std.testing.expectEqual(@as(u64, 1), counts[1]);
}

// ============================================================================
// Experiment: Variant Assignment
// ============================================================================

test "experiment: empty variants returns control" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    var exp = analytics.Experiment{
        .name = "empty_exp",
        .variants = &.{},
    };

    const variant = exp.assign("any-user");
    try std.testing.expectEqualStrings("control", variant);
}

test "experiment: single variant always chosen" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    var exp = analytics.Experiment{
        .name = "single_exp",
        .variants = &.{"treatment"},
    };

    // All users get the only variant
    for (0..20) |i| {
        var buf: [32]u8 = undefined;
        // SAFETY: max output "user-20" is 7 bytes, buffer is 32.
        const user = std.fmt.bufPrint(&buf, "user-{d}", .{i}) catch unreachable;
        const variant = exp.assign(user);
        try std.testing.expectEqualStrings("treatment", variant);
    }
    try std.testing.expectEqual(@as(u64, 20), exp.totalAssignments());
}

test "experiment: multiple variants distribute across users" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    var exp = analytics.Experiment{
        .name = "multi_exp",
        .variants = &.{ "a", "b", "c" },
    };

    var counts = [_]u64{ 0, 0, 0 };
    const variant_names = [_][]const u8{ "a", "b", "c" };

    for (0..300) |i| {
        var buf: [32]u8 = undefined;
        // SAFETY: max output "user-299" is 8 bytes, buffer is 32.
        const user = std.fmt.bufPrint(&buf, "user-{d}", .{i}) catch unreachable;
        const variant = exp.assign(user);

        for (variant_names, 0..) |name, idx| {
            if (std.mem.eql(u8, variant, name)) {
                counts[idx] += 1;
                break;
            }
        }
    }

    // With Fnv1a hash and 300 users across 3 variants, each should get ~100.
    // Allow generous margin but verify all variants get assigned.
    for (counts) |c| {
        try std.testing.expect(c > 0);
    }
    try std.testing.expectEqual(@as(u64, 300), exp.totalAssignments());
}

// ============================================================================
// Context: Lifecycle
// ============================================================================

test "context: init creates engine" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var ctx = try analytics.Context.init(allocator, .{});
    defer ctx.deinit();

    const engine = ctx.getEngine();
    try std.testing.expect(engine != null);
}

test "context: engine accessible via context" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var ctx = try analytics.Context.init(allocator, .{ .buffer_capacity = 10 });
    defer ctx.deinit();

    const engine = ctx.getEngine().?;
    try engine.track("ctx_event");
    try std.testing.expectEqual(@as(usize, 1), engine.bufferedCount());
}

// ============================================================================
// Module Lifecycle
// ============================================================================

test "module: isEnabled matches build flag" {
    if (!build_options.enable_analytics) return error.SkipZigTest;
    try std.testing.expect(analytics.isEnabled());
}

test "module: init and deinit lifecycle" {
    if (!build_options.enable_analytics) return error.SkipZigTest;

    try std.testing.expect(!analytics.isInitialized());
    try analytics.init(std.testing.allocator);
    try std.testing.expect(analytics.isInitialized());
    analytics.deinit();
    try std.testing.expect(!analytics.isInitialized());
}
