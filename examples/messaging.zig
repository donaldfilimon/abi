//! Messaging Example
//!
//! Demonstrates pub/sub messaging with MQTT-style topic patterns.
//! Shows subscribe, publish, and wildcard pattern matching.
//!
//! Run with: `zig build run-messaging`

const std = @import("std");
const abi = @import("abi");

fn onUserEvent(msg: abi.features.messaging.Message, _: ?*anyopaque) abi.features.messaging.DeliveryResult {
    std.debug.print("  [{s}] {s}\n", .{ msg.topic, msg.payload });
    return .ok;
}

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.App.builder(allocator);

    var framework = try builder
        .with(.messaging, abi.config.MessagingConfig{})
        .build();
    defer framework.deinit();

    if (!abi.features.messaging.isEnabled()) {
        std.debug.print("Messaging feature is disabled. Enable with -Denable-messaging=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Messaging Example ===\n\n", .{});

    // Subscribe with MQTT wildcard pattern
    // '*' matches single level, '#' matches multiple levels
    const sub_id = abi.features.messaging.subscribe(
        allocator,
        "users/*",
        onUserEvent,
        null,
    ) catch |err| {
        std.debug.print("Failed to subscribe: {t}\n", .{err});
        return;
    };
    std.debug.print("Subscribed to 'users/*' (id={})\n", .{sub_id});

    // Publish messages
    const topics = [_]struct { topic: []const u8, payload: []const u8 }{
        .{ .topic = "users/login", .payload = "alice logged in" },
        .{ .topic = "users/signup", .payload = "bob signed up" },
        .{ .topic = "system/health", .payload = "all systems OK" },
    };

    std.debug.print("\nPublishing messages:\n", .{});
    for (topics) |t| {
        abi.features.messaging.publish(allocator, t.topic, t.payload) catch |err| {
            std.debug.print("  Publish to {s} failed: {t}\n", .{ t.topic, err });
            continue;
        };
    }

    // List topics
    const topic_list = abi.features.messaging.listTopics(allocator) catch |err| {
        std.debug.print("Failed to list topics: {t}\n", .{err});
        return;
    };
    defer allocator.free(topic_list);

    std.debug.print("\nActive topics:\n", .{});
    for (topic_list) |name| {
        std.debug.print("  {s}\n", .{name});
    }

    // Unsubscribe
    const removed = abi.features.messaging.unsubscribe(sub_id) catch false;
    std.debug.print("\nUnsubscribed (id={}): {}\n", .{ sub_id, removed });

    // Stats
    const s = abi.features.messaging.messagingStats();
    std.debug.print("Messaging stats: {} published, {} delivered, {} topics\n", .{
        s.total_published, s.total_delivered, s.active_topics,
    });
}
