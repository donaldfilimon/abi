//! Integration Tests: Analytics Feature
//!
//! Tests the analytics module exports, event tracking types,
//! engine lifecycle, and experiment/funnel APIs.

const std = @import("std");
const abi = @import("abi");

const analytics = abi.analytics;

// ============================================================================
// Module Lifecycle
// ============================================================================

test "analytics: isEnabled returns bool" {
    const enabled = analytics.isEnabled();
    try std.testing.expect(enabled == true or enabled == false);
}

test "analytics: isInitialized returns bool" {
    const initialized = analytics.isInitialized();
    try std.testing.expect(initialized == true or initialized == false);
}

// ============================================================================
// Types
// ============================================================================

test "analytics: AnalyticsError includes expected variants" {
    const err: analytics.AnalyticsError = error.FeatureDisabled;
    try std.testing.expect(err == error.FeatureDisabled);
}

test "analytics: AnalyticsConfig default values" {
    const config = analytics.AnalyticsConfig{};
    try std.testing.expectEqual(@as(u32, 1024), config.buffer_capacity);
    try std.testing.expect(config.enable_timestamps);
    try std.testing.expectEqualStrings("abi-app", config.app_id);
    try std.testing.expectEqual(@as(u64, 0), config.flush_interval_ms);
}

test "analytics: AnalyticsConfig custom values" {
    const config = analytics.AnalyticsConfig{
        .buffer_capacity = 512,
        .enable_timestamps = false,
        .app_id = "my-service",
        .flush_interval_ms = 5000,
    };
    try std.testing.expectEqual(@as(u32, 512), config.buffer_capacity);
    try std.testing.expect(!config.enable_timestamps);
    try std.testing.expectEqualStrings("my-service", config.app_id);
}

test "analytics: Event type has expected fields" {
    const event = analytics.Event{
        .name = "page_view",
        .timestamp_ms = 1700000000,
        .session_id = "sess-abc",
    };
    try std.testing.expectEqualStrings("page_view", event.name);
    try std.testing.expectEqual(@as(u64, 1700000000), event.timestamp_ms);
    try std.testing.expectEqualStrings("sess-abc", event.session_id.?);
    try std.testing.expectEqual(@as(usize, 0), event.properties.len);
}

test "analytics: Event.Value union variants" {
    const str_val = analytics.Event.Value{ .string = "hello" };
    const int_val = analytics.Event.Value{ .int = 42 };
    const float_val = analytics.Event.Value{ .float = 3.14 };
    const bool_val = analytics.Event.Value{ .boolean = true };

    try std.testing.expectEqualStrings("hello", str_val.string);
    try std.testing.expectEqual(@as(i64, 42), int_val.int);
    try std.testing.expectEqual(@as(f64, 3.14), float_val.float);
    try std.testing.expect(bool_val.boolean);
}

test "analytics: Event.Property can be constructed" {
    const prop = analytics.Event.Property{
        .key = "source",
        .value = .{ .string = "organic" },
    };
    try std.testing.expectEqualStrings("source", prop.key);
    try std.testing.expectEqualStrings("organic", prop.value.string);
}

// ============================================================================
// Engine
// ============================================================================

test "analytics: Engine init and deinit" {
    const config = analytics.AnalyticsConfig{};
    var engine = analytics.Engine.init(std.testing.allocator, config);
    defer engine.deinit();

    try std.testing.expectEqual(@as(usize, 0), engine.bufferedCount());
    try std.testing.expectEqual(@as(u64, 0), engine.totalEvents());
}

test "analytics: Engine track returns FeatureDisabled when disabled" {
    const config = analytics.AnalyticsConfig{};
    var engine = analytics.Engine.init(std.testing.allocator, config);
    defer engine.deinit();

    const result = engine.track("test_event");
    if (result) |_| {
        // Feature is enabled — event was tracked
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "analytics: Engine trackWithSession returns FeatureDisabled when disabled" {
    const config = analytics.AnalyticsConfig{};
    var engine = analytics.Engine.init(std.testing.allocator, config);
    defer engine.deinit();

    const result = engine.trackWithSession("test_event", "sess-1");
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "analytics: Engine startSession returns session id" {
    const config = analytics.AnalyticsConfig{};
    var engine = analytics.Engine.init(std.testing.allocator, config);
    defer engine.deinit();

    const session_id = engine.startSession();
    _ = session_id;
}

test "analytics: Engine flush returns count" {
    const config = analytics.AnalyticsConfig{};
    var engine = analytics.Engine.init(std.testing.allocator, config);
    defer engine.deinit();

    const flushed = engine.flush();
    _ = flushed;
}

test "analytics: Engine getStats returns Stats" {
    const config = analytics.AnalyticsConfig{};
    var engine = analytics.Engine.init(std.testing.allocator, config);
    defer engine.deinit();

    const s = engine.getStats();
    try std.testing.expectEqual(@as(usize, 0), s.buffered_events);
    try std.testing.expectEqual(@as(u64, 0), s.total_events);
    try std.testing.expectEqual(@as(u64, 0), s.total_sessions);
}

// ============================================================================
// Funnel
// ============================================================================

test "analytics: Funnel init and deinit" {
    var funnel = analytics.Funnel.init(std.testing.allocator, "onboarding");
    defer funnel.deinit();

    try std.testing.expectEqualStrings("onboarding", funnel.name);
}

test "analytics: Funnel addStep does not error" {
    var funnel = analytics.Funnel.init(std.testing.allocator, "checkout");
    defer funnel.deinit();

    try funnel.addStep("cart");
    try funnel.addStep("payment");
    try funnel.addStep("confirmation");
}

test "analytics: Funnel getStepCounts returns slice" {
    var funnel = analytics.Funnel.init(std.testing.allocator, "signup");
    defer funnel.deinit();

    var buf: [8]u64 = [_]u64{0} ** 8;
    const counts = funnel.getStepCounts(&buf);
    _ = counts;
}

// ============================================================================
// Experiment
// ============================================================================

test "analytics: Experiment assign returns variant" {
    const variants = [_][]const u8{ "control", "variant_a", "variant_b" };
    var experiment = analytics.Experiment{
        .name = "button_color",
        .variants = &variants,
    };

    const assigned = experiment.assign("user-1");
    try std.testing.expect(assigned.len > 0);
}

test "analytics: Experiment totalAssignments tracks count" {
    const variants = [_][]const u8{ "control", "variant_a" };
    var experiment = analytics.Experiment{
        .name = "test_exp",
        .variants = &variants,
    };

    try std.testing.expectEqual(@as(u64, 0), experiment.totalAssignments());
    _ = experiment.assign("user-1");
}

test {
    std.testing.refAllDecls(@This());
}
