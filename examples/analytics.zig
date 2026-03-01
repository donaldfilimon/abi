//! Analytics Example
//!
//! Demonstrates the analytics module: event tracking, sessions,
//! funnels, and statistics gathering.
//!
//! Run with: `zig build run-analytics`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.App.builder(allocator);

    var framework = try builder
        .with(.analytics, abi.config.AnalyticsConfig{})
        .build();
    defer framework.deinit();

    if (!abi.features.analytics.isEnabled()) {
        std.debug.print("Analytics feature is disabled. Enable with -Denable-analytics=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Analytics Example ===\n\n", .{});

    // Initialize analytics engine
    var engine = abi.features.analytics.Engine.init(allocator, .{
        .buffer_capacity = 100,
        .flush_interval_ms = 5000,
    });
    defer engine.deinit();

    // Track events
    engine.track("page_view") catch |err| {
        std.debug.print("Failed to track: {t}\n", .{err});
        return;
    };
    engine.track("button_click") catch |err| {
        std.debug.print("Failed to track: {t}\n", .{err});
        return;
    };
    std.debug.print("Tracked 2 events\n", .{});

    // Session tracking
    const session_id = engine.startSession();
    std.debug.print("Started session: {}\n", .{session_id});

    engine.trackWithSession("login", null) catch |err| {
        std.debug.print("Failed to track with session: {t}\n", .{err});
        return;
    };

    // Buffer stats
    const buffered = engine.bufferedCount();
    const total = engine.totalEvents();
    std.debug.print("Buffered: {}, Total: {}\n", .{ buffered, total });

    // Flush
    const flushed = engine.flush();
    std.debug.print("Flushed {} events\n", .{flushed});

    // Final stats
    const s = engine.getStats();
    std.debug.print("\nAnalytics stats: {} total events, {} sessions\n", .{
        s.total_events, s.total_sessions,
    });
}
