//! Integration Tests: Messaging (Pub/Sub)
//!
//! Verifies the messaging module's public types, configuration,
//! pub/sub lifecycle, wildcard patterns, dead letter queue,
//! and statistics through the `abi.messaging` facade.

const std = @import("std");
const abi = @import("abi");

const messaging = abi.messaging;

// ── Type availability ──────────────────────────────────────────────────

test "messaging: core types are accessible" {
    _ = messaging.Message;
    _ = messaging.Channel;
    _ = messaging.MessagingConfig;
    _ = messaging.MessagingError;
    _ = messaging.MessagingStats;
    _ = messaging.TopicInfo;
    _ = messaging.DeadLetter;
    _ = messaging.DeliveryResult;
    _ = messaging.SubscriberCallback;
}

test "messaging: Message default fields" {
    const msg: messaging.Message = .{};
    try std.testing.expectEqualStrings("", msg.topic);
    try std.testing.expectEqualStrings("", msg.payload);
    try std.testing.expectEqual(@as(u64, 0), msg.timestamp);
    try std.testing.expectEqual(@as(u64, 0), msg.id);
}

test "messaging: Channel default fields" {
    const ch: messaging.Channel = .{};
    try std.testing.expectEqualStrings("", ch.name);
    try std.testing.expectEqual(@as(u32, 0), ch.subscriber_count);
}

test "messaging: MessagingStats default fields" {
    const stats: messaging.MessagingStats = .{};
    try std.testing.expectEqual(@as(u64, 0), stats.total_published);
    try std.testing.expectEqual(@as(u64, 0), stats.total_delivered);
    try std.testing.expectEqual(@as(u64, 0), stats.total_failed);
    try std.testing.expectEqual(@as(u32, 0), stats.active_topics);
    try std.testing.expectEqual(@as(u32, 0), stats.active_subscribers);
    try std.testing.expectEqual(@as(u32, 0), stats.dead_letter_count);
}

test "messaging: DeliveryResult enum values" {
    const ok = messaging.DeliveryResult.ok;
    const retry = messaging.DeliveryResult.retry;
    const discard = messaging.DeliveryResult.discard;

    try std.testing.expect(ok != retry);
    try std.testing.expect(retry != discard);
    try std.testing.expect(ok != discard);
}

// ── Config defaults ────────────────────────────────────────────────────

test "messaging: MessagingConfig defaults are sensible" {
    const cfg = messaging.MessagingConfig.defaults();
    try std.testing.expect(cfg.max_channels > 0);
    try std.testing.expect(cfg.buffer_size > 0);
}

// ── Context lifecycle ──────────────────────────────────────────────────

test "messaging: Context init and deinit" {
    const allocator = std.testing.allocator;
    const cfg = messaging.MessagingConfig.defaults();
    const ctx = try messaging.Context.init(allocator, cfg);
    defer ctx.deinit();

    try std.testing.expectEqual(allocator, ctx.allocator);
}

// ── Module lifecycle ───────────────────────────────────────────────────

test "messaging: isEnabled returns true" {
    try std.testing.expect(messaging.isEnabled());
}

test "messaging: init and deinit lifecycle" {
    const allocator = std.testing.allocator;
    const cfg = messaging.MessagingConfig.defaults();

    try messaging.init(allocator, cfg);
    try std.testing.expect(messaging.isInitialized());

    messaging.deinit();
    try std.testing.expect(!messaging.isInitialized());
}

// ── Pub/Sub round-trip ─────────────────────────────────────────────────

var integration_received: u32 = 0;
var integration_last_payload: []const u8 = "";

fn integrationCallback(msg: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    integration_received += 1;
    integration_last_payload = msg.payload;
    return .ok;
}

test "messaging: subscribe, publish, receive" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    integration_received = 0;
    const sub_id = try messaging.subscribe(allocator, "integration.test", integrationCallback, null);
    try std.testing.expect(sub_id > 0);

    try messaging.publish(allocator, "integration.test", "hello-integration");
    try std.testing.expectEqual(@as(u32, 1), integration_received);
    try std.testing.expectEqualStrings("hello-integration", integration_last_payload);

    const removed = try messaging.unsubscribe(sub_id);
    try std.testing.expect(removed);
}

// ── Statistics ─────────────────────────────────────────────────────────

test "messaging: stats track publish and delivery" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    integration_received = 0;
    _ = try messaging.subscribe(allocator, "stats.int", integrationCallback, null);

    try messaging.publish(allocator, "stats.int", "msg-a");
    try messaging.publish(allocator, "stats.int", "msg-b");

    const stats = messaging.messagingStats();
    try std.testing.expectEqual(@as(u64, 2), stats.total_published);
    try std.testing.expectEqual(@as(u64, 2), stats.total_delivered);
    try std.testing.expect(stats.active_subscribers > 0);
}

// ── Topic listing ──────────────────────────────────────────────────────

test "messaging: listTopics returns created topics" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "list.alpha", integrationCallback, null);
    try messaging.publish(allocator, "list.beta", "data");

    const topics = try messaging.listTopics(allocator);
    defer allocator.free(topics);
    try std.testing.expect(topics.len >= 2);
}

// ── Topic stats ────────────────────────────────────────────────────────

test "messaging: topicStats accuracy" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    integration_received = 0;
    _ = try messaging.subscribe(allocator, "tstats.int", integrationCallback, null);
    try messaging.publish(allocator, "tstats.int", "m1");
    try messaging.publish(allocator, "tstats.int", "m2");

    const info = try messaging.topicStats("tstats.int");
    try std.testing.expectEqual(@as(u64, 2), info.messages_published);
    try std.testing.expectEqual(@as(u64, 2), info.messages_delivered);
    try std.testing.expectEqual(@as(u32, 1), info.subscriber_count);
}

// ── Dead letter queue ──────────────────────────────────────────────────

fn discardCallback(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    return .discard;
}

test "messaging: dead letter queue captures discards" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "dlq.int", discardCallback, null);
    try messaging.publish(allocator, "dlq.int", "bad-payload");

    const dead = try messaging.getDeadLetters(allocator);
    defer allocator.free(dead);
    try std.testing.expectEqual(@as(usize, 1), dead.len);
    try std.testing.expectEqualStrings("dlq.int", dead[0].message.topic);
    try std.testing.expectEqualStrings("subscriber_discard", dead[0].reason);

    messaging.clearDeadLetters();
    const after = try messaging.getDeadLetters(allocator);
    defer allocator.free(after);
    try std.testing.expectEqual(@as(usize, 0), after.len);
}

test {
    std.testing.refAllDecls(@This());
}
