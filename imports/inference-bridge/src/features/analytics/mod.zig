//! Analytics Module
//!
//! Business event tracking, session analytics, and funnel instrumentation.
//! Unlike observability (system metrics, tracing, profiling), analytics
//! focuses on user-facing events and product usage patterns.
//!
//! ## Features
//! - Custom event tracking with typed properties
//! - Session lifecycle management
//! - Funnel step recording
//! - A/B experiment assignment tracking
//! - Thread-safe event buffer with configurable flush

const std = @import("std");
const build_options = @import("build_options");
const time = @import("../../foundation/mod.zig").time;
pub const types = @import("types.zig");
const engine_mod = @import("engine.zig");
const funnel_mod = @import("funnel.zig");

// ============================================================================
// Shared Types (from types.zig)
// ============================================================================

pub const Event = types.Event;
pub const AnalyticsConfig = types.AnalyticsConfig;
pub const AnalyticsError = types.AnalyticsError;
pub const Error = AnalyticsError;

// ============================================================================
// Analytics Engine (from engine.zig)
// ============================================================================

pub const Engine = engine_mod.Engine;

// ============================================================================
// Funnel Tracking (from funnel.zig)
// ============================================================================

pub const Funnel = funnel_mod.Funnel;

// ============================================================================
// Experiment Tracking
// ============================================================================

pub const Experiment = types.Experiment;

// ============================================================================
// Module Lifecycle
// ============================================================================

/// Analytics context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: AnalyticsConfig,
    engine: ?Engine = null,

    pub fn init(allocator: std.mem.Allocator, config: AnalyticsConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = config,
            .engine = Engine.init(allocator, config),
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.engine) |*eng| eng.deinit();
        self.allocator.destroy(self);
    }

    pub fn getEngine(self: *Context) ?*Engine {
        return if (self.engine != null) &self.engine.? else null;
    }
};

var initialized = std.atomic.Value(bool).init(false);

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    if (initialized.load(.acquire)) return;
    initialized.store(true, .release);
}

pub fn deinit() void {
    initialized.store(false, .release);
}

pub fn isEnabled() bool {
    return build_options.feat_analytics;
}

pub fn isInitialized() bool {
    return initialized.load(.acquire);
}

// ============================================================================
// Helpers
// ============================================================================

/// Application start anchor for monotonic timestamps.
var app_start: ?time.Instant = null;

fn timestampMs() u64 {
    const start = app_start orelse blk: {
        const s = time.Instant.now() catch return 0;
        app_start = s;
        break :blk s;
    };
    const now = time.Instant.now() catch return 0;
    return now.since(start) / std.time.ns_per_ms;
}

// ============================================================================
// Tests
// ============================================================================

test "Engine tracks events" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 100 });
    defer engine.deinit();

    try engine.track("page_view");
    try engine.track("button_click");
    try engine.trackWithSession("purchase", "session-1");

    try std.testing.expectEqual(@as(usize, 3), engine.bufferedCount());
    try std.testing.expectEqual(@as(u64, 3), engine.totalEvents());
}

test "Engine respects buffer capacity" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 2 });
    defer engine.deinit();

    try engine.track("e1");
    try engine.track("e2");
    try std.testing.expectError(AnalyticsError.BufferFull, engine.track("e3"));
}

test "Engine flush clears buffer" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 100 });
    defer engine.deinit();

    try engine.track("e1");
    try engine.track("e2");
    const flushed = engine.flush();
    try std.testing.expectEqual(@as(usize, 2), flushed);
    try std.testing.expectEqual(@as(usize, 0), engine.bufferedCount());
    // Total count preserved after flush
    try std.testing.expectEqual(@as(u64, 2), engine.totalEvents());
}

test "Engine session counting" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{});
    defer engine.deinit();

    const s1 = engine.startSession();
    const s2 = engine.startSession();
    try std.testing.expectEqual(@as(u64, 0), s1);
    try std.testing.expectEqual(@as(u64, 1), s2);

    const stats = engine.getStats();
    try std.testing.expectEqual(@as(u64, 2), stats.total_sessions);
}

test "Funnel tracks step progression" {
    const allocator = std.testing.allocator;
    var funnel = Funnel.init(allocator, "onboarding");
    defer funnel.deinit();

    try funnel.addStep("signup");
    try funnel.addStep("verify_email");
    try funnel.addStep("complete_profile");

    funnel.recordStep(0); // signup
    funnel.recordStep(0);
    funnel.recordStep(1); // verify
    funnel.recordStep(2); // complete

    var buf: [3]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(u64, 2), counts[0]);
    try std.testing.expectEqual(@as(u64, 1), counts[1]);
    try std.testing.expectEqual(@as(u64, 1), counts[2]);
}

test "Experiment assigns deterministically" {
    var exp = Experiment{
        .name = "color_test",
        .variants = &.{ "red", "blue", "green" },
    };

    const v1 = exp.assign("user-123");
    const v2 = exp.assign("user-123");
    try std.testing.expectEqualStrings(v1, v2); // same user = same variant

    try std.testing.expectEqual(@as(u64, 2), exp.totalAssignments());
}

test "AnalyticsConfig default values" {
    const config = AnalyticsConfig{};
    try std.testing.expectEqual(@as(u32, 1024), config.buffer_capacity);
    try std.testing.expect(config.enable_timestamps);
    try std.testing.expectEqualStrings("abi-app", config.app_id);
    try std.testing.expectEqual(@as(u64, 0), config.flush_interval_ms);
}

test "Engine getStats reflects state" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 50 });
    defer engine.deinit();

    try engine.track("ev1");
    try engine.track("ev2");
    _ = engine.startSession();

    const s = engine.getStats();
    try std.testing.expectEqual(@as(usize, 2), s.buffered_events);
    try std.testing.expectEqual(@as(u64, 2), s.total_events);
    try std.testing.expectEqual(@as(u64, 1), s.total_sessions);
}

test "Engine flush then track more" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 10 });
    defer engine.deinit();

    try engine.track("a");
    try engine.track("b");
    _ = engine.flush();

    try engine.track("c");
    try std.testing.expectEqual(@as(usize, 1), engine.bufferedCount());
    try std.testing.expectEqual(@as(u64, 3), engine.totalEvents());
}

test "Engine timestamps disabled" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{
        .buffer_capacity = 10,
        .enable_timestamps = false,
    });
    defer engine.deinit();

    try engine.track("no-ts");
    // When timestamps are disabled, the stored event should have timestamp 0
    try std.testing.expectEqual(@as(usize, 1), engine.bufferedCount());
}

test "Funnel empty step counts" {
    const allocator = std.testing.allocator;
    var funnel = Funnel.init(allocator, "empty_funnel");
    defer funnel.deinit();

    var buf: [4]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(usize, 0), counts.len);
}

test "Funnel recordStep out of bounds is safe" {
    const allocator = std.testing.allocator;
    var funnel = Funnel.init(allocator, "safe_funnel");
    defer funnel.deinit();

    try funnel.addStep("only_step");
    // Out-of-bounds index should be silently ignored
    funnel.recordStep(99);

    var buf: [1]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(u64, 0), counts[0]);
}

test "Funnel getStepCounts with smaller buffer" {
    const allocator = std.testing.allocator;
    var funnel = Funnel.init(allocator, "sized_funnel");
    defer funnel.deinit();

    try funnel.addStep("step1");
    try funnel.addStep("step2");
    try funnel.addStep("step3");

    funnel.recordStep(0);
    funnel.recordStep(1);
    funnel.recordStep(2);

    // Buffer smaller than step count — should only return buf.len entries
    var buf: [2]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(usize, 2), counts.len);
    try std.testing.expectEqual(@as(u64, 1), counts[0]);
    try std.testing.expectEqual(@as(u64, 1), counts[1]);
}

test "Experiment empty variants returns control" {
    var exp = Experiment{
        .name = "empty_exp",
        .variants = &.{},
    };
    const variant = exp.assign("any-user");
    try std.testing.expectEqualStrings("control", variant);
    // assignments counter should NOT increment for empty variants
    try std.testing.expectEqual(@as(u64, 0), exp.totalAssignments());
}

test "Experiment different users may get different variants" {
    var exp = Experiment{
        .name = "multi_variant",
        .variants = &.{ "a", "b", "c", "d" },
    };
    // Assign many users and verify all results are valid variants
    const users = [_][]const u8{ "u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8" };
    for (users) |uid| {
        const v = exp.assign(uid);
        var found = false;
        for (exp.variants) |candidate| {
            if (std.mem.eql(u8, v, candidate)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
    try std.testing.expectEqual(@as(u64, 8), exp.totalAssignments());
}

test "Context init and getEngine" {
    const allocator = std.testing.allocator;
    const ctx = try Context.init(allocator, .{});
    defer ctx.deinit();

    const engine_ptr = ctx.getEngine();
    try std.testing.expect(engine_ptr != null);
}

test "Event.Value union variants" {
    const str_val = Event.Value{ .string = "hello" };
    const int_val = Event.Value{ .int = 42 };
    const float_val = Event.Value{ .float = 3.14 };
    const bool_val = Event.Value{ .boolean = true };

    switch (str_val) {
        .string => |s| try std.testing.expectEqualStrings("hello", s),
        else => return error.TestUnexpectedResult,
    }
    switch (int_val) {
        .int => |i| try std.testing.expectEqual(@as(i64, 42), i),
        else => return error.TestUnexpectedResult,
    }
    switch (float_val) {
        .float => |f| try std.testing.expectApproxEqAbs(@as(f64, 3.14), f, 0.001),
        else => return error.TestUnexpectedResult,
    }
    switch (bool_val) {
        .boolean => |b| try std.testing.expect(b),
        else => return error.TestUnexpectedResult,
    }
}

test {
    std.testing.refAllDecls(@This());
}
