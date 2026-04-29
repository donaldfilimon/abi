const std = @import("std");
const messaging = @import("mod.zig");
const pattern_matching = @import("pattern_matching.zig");

var test_received_count: u32 = 0;
var test_last_payload: []const u8 = "";

fn testCallback(msg: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    test_received_count += 1;
    test_last_payload = msg.payload;
    return .ok;
}

fn testDiscardCallback(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    return .discard;
}

test "messaging basic pub/sub" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_received_count = 0;
    const sub_id = try messaging.subscribe(allocator, "events.user", testCallback, null);
    try std.testing.expect(sub_id > 0);

    try messaging.publish(allocator, "events.user", "hello");
    try std.testing.expectEqual(@as(u32, 1), test_received_count);
    try std.testing.expectEqualStrings("hello", test_last_payload);
}

test "messaging unsubscribe" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_received_count = 0;
    const sub_id = try messaging.subscribe(allocator, "test.topic", testCallback, null);
    try messaging.publish(allocator, "test.topic", "msg1");
    try std.testing.expectEqual(@as(u32, 1), test_received_count);

    const removed = try messaging.unsubscribe(sub_id);
    try std.testing.expect(removed);

    try messaging.publish(allocator, "test.topic", "msg2");
    try std.testing.expectEqual(@as(u32, 1), test_received_count);
}

test "messaging wildcard pattern *" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_received_count = 0;
    _ = try messaging.subscribe(allocator, "events.*", testCallback, null);

    try messaging.publish(allocator, "events.user", "payload1");
    try messaging.publish(allocator, "events.order", "payload2");
    try messaging.publish(allocator, "other.topic", "payload3");

    try std.testing.expectEqual(@as(u32, 2), test_received_count);
}

test "messaging multi-level pattern #" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_received_count = 0;
    _ = try messaging.subscribe(allocator, "events.#", testCallback, null);

    try messaging.publish(allocator, "events.user", "p1");
    try messaging.publish(allocator, "events.user.created", "p2");
    try messaging.publish(allocator, "events.order.updated.v2", "p3");
    try messaging.publish(allocator, "other.topic", "p4");

    try std.testing.expectEqual(@as(u32, 3), test_received_count);
}

test "messaging dead letter queue" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "dlq.test", testDiscardCallback, null);
    try messaging.publish(allocator, "dlq.test", "will_fail");

    const dead = try messaging.getDeadLetters(allocator);
    defer allocator.free(dead);
    try std.testing.expectEqual(@as(usize, 1), dead.len);

    messaging.clearDeadLetters();
    const after = try messaging.getDeadLetters(allocator);
    defer allocator.free(after);
    try std.testing.expectEqual(@as(usize, 0), after.len);
}

test "messaging stats" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "stats.test", testCallback, null);
    try messaging.publish(allocator, "stats.test", "msg1");

    const s = messaging.messagingStats();
    try std.testing.expect(s.total_published > 0);
    try std.testing.expect(s.active_subscribers > 0);
}

var test_fanout_count: u32 = 0;

fn testFanoutCallback(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    test_fanout_count += 1;
    return .ok;
}

test "messaging multiple subscribers fan-out" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_fanout_count = 0;
    _ = try messaging.subscribe(allocator, "fanout.topic", testFanoutCallback, null);
    _ = try messaging.subscribe(allocator, "fanout.topic", testFanoutCallback, null);
    _ = try messaging.subscribe(allocator, "fanout.topic", testFanoutCallback, null);

    try messaging.publish(allocator, "fanout.topic", "broadcast");
    try std.testing.expectEqual(@as(u32, 3), test_fanout_count);
}

test "messaging listTopics after publish" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "list.alpha", testCallback, null);
    try messaging.publish(allocator, "list.beta", "msg");

    const topics = try messaging.listTopics(allocator);
    defer allocator.free(topics);
    try std.testing.expect(topics.len >= 2);
}

test "messaging topicStats accuracy" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_received_count = 0;
    _ = try messaging.subscribe(allocator, "tstats.topic", testCallback, null);
    try messaging.publish(allocator, "tstats.topic", "msg1");
    try messaging.publish(allocator, "tstats.topic", "msg2");

    const info = try messaging.topicStats("tstats.topic");
    try std.testing.expectEqual(@as(u64, 2), info.messages_published);
    try std.testing.expectEqual(@as(u64, 2), info.messages_delivered);
    try std.testing.expectEqual(@as(u32, 1), info.subscriber_count);
}

test "messaging publish to non-existent topic auto-creates" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    try messaging.publish(allocator, "auto.created", "payload");

    const info = try messaging.topicStats("auto.created");
    try std.testing.expectEqual(@as(u64, 1), info.messages_published);
    try std.testing.expectEqual(@as(u32, 0), info.subscriber_count);
}

test "messaging unsubscribe invalid ID is no-op" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "test/topic", struct {
        fn cb(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
            return .ok;
        }
    }.cb, null);

    const removed = try messaging.unsubscribe(999999);
    try std.testing.expect(!removed);

    const s = messaging.messagingStats();
    try std.testing.expectEqual(@as(u32, 1), s.active_subscribers);
}

test "messaging publish to empty string topic" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    try messaging.publish(allocator, "", "data");
}

test "messaging multiple wildcard pattern" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "sensors/+/data", struct {
        fn cb(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
            return .ok;
        }
    }.cb, null);

    try messaging.publish(allocator, "sensors/temp/data", "25C");

    const info = try messaging.topicStats("sensors/temp/data");
    try std.testing.expectEqual(@as(u64, 1), info.messages_published);
}

var test_sub_count_a: u32 = 0;
var test_sub_count_b: u32 = 0;

fn testCallbackA(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    test_sub_count_a += 1;
    return .ok;
}

fn testCallbackB(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    test_sub_count_b += 1;
    return .ok;
}

test "messaging subscribe and unsubscribe lifecycle" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_sub_count_a = 0;
    const sub_id = try messaging.subscribe(allocator, "lifecycle.topic", testCallbackA, null);

    try messaging.publish(allocator, "lifecycle.topic", "msg1");
    try std.testing.expectEqual(@as(u32, 1), test_sub_count_a);

    const removed = try messaging.unsubscribe(sub_id);
    try std.testing.expect(removed);

    try messaging.publish(allocator, "lifecycle.topic", "msg2");
    try std.testing.expectEqual(@as(u32, 1), test_sub_count_a);

    const removed2 = try messaging.unsubscribe(sub_id);
    try std.testing.expect(!removed2);
}

test "messaging publish no subscribers does not error" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    try messaging.publish(allocator, "no.subs.topic", "lonely_msg");

    const info = try messaging.topicStats("no.subs.topic");
    try std.testing.expectEqual(@as(u64, 1), info.messages_published);
    try std.testing.expectEqual(@as(u64, 0), info.messages_delivered);
    try std.testing.expectEqual(@as(u32, 0), info.subscriber_count);
}

test "messaging multiple subscribers same topic" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_sub_count_a = 0;
    test_sub_count_b = 0;
    _ = try messaging.subscribe(allocator, "multi.topic", testCallbackA, null);
    _ = try messaging.subscribe(allocator, "multi.topic", testCallbackB, null);

    try messaging.publish(allocator, "multi.topic", "broadcast_data");

    try std.testing.expectEqual(@as(u32, 1), test_sub_count_a);
    try std.testing.expectEqual(@as(u32, 1), test_sub_count_b);
}

var test_retry_count: u32 = 0;

fn testRetryCallback(_: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    test_retry_count += 1;
    return .retry;
}

test "messaging acknowledge and reject via DeliveryResult" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_retry_count = 0;
    _ = try messaging.subscribe(allocator, "ack.topic", testRetryCallback, null);

    try messaging.publish(allocator, "ack.topic", "will_retry");
    try std.testing.expectEqual(@as(u32, 1), test_retry_count);

    const stats = messaging.messagingStats();
    try std.testing.expect(stats.total_failed > 0);
    try std.testing.expectEqual(@as(u32, 0), stats.dead_letter_count);
}

test "messaging discard sends to dead letter queue" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    _ = try messaging.subscribe(allocator, "discard.topic", testDiscardCallback, null);

    try messaging.publish(allocator, "discard.topic", "reject_me_1");
    try messaging.publish(allocator, "discard.topic", "reject_me_2");

    const dead = try messaging.getDeadLetters(allocator);
    defer allocator.free(dead);
    try std.testing.expectEqual(@as(usize, 2), dead.len);
    try std.testing.expectEqualStrings("discard.topic", dead[0].message.topic);
    try std.testing.expectEqualStrings("discard.topic", dead[1].message.topic);
    try std.testing.expectEqualStrings("subscriber_discard", dead[0].reason);
}

test "messaging wildcard # matches multiple levels" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_sub_count_a = 0;
    _ = try messaging.subscribe(allocator, "app.#", testCallbackA, null);

    try messaging.publish(allocator, "app.users", "p1");
    try messaging.publish(allocator, "app.users.created", "p2");
    try messaging.publish(allocator, "app.orders.shipped.v2", "p3");
    try messaging.publish(allocator, "other.topic", "p4");

    try std.testing.expectEqual(@as(u32, 3), test_sub_count_a);
}

test "messaging wildcard * matches single level" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    test_sub_count_a = 0;
    _ = try messaging.subscribe(allocator, "data.*", testCallbackA, null);

    try messaging.publish(allocator, "data.temp", "25C");
    try messaging.publish(allocator, "data.humidity", "60%");
    try messaging.publish(allocator, "data.temp.indoor", "22C");
    try messaging.publish(allocator, "info.temp", "30C");

    try std.testing.expectEqual(@as(u32, 2), test_sub_count_a);
}

test "messaging stats reflect publish and subscribe counts" {
    const allocator = std.testing.allocator;
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    const s0 = messaging.messagingStats();
    try std.testing.expectEqual(@as(u64, 0), s0.total_published);
    try std.testing.expectEqual(@as(u32, 0), s0.active_subscribers);

    _ = try messaging.subscribe(allocator, "stats2.topic", testCallbackA, null);
    _ = try messaging.subscribe(allocator, "stats2.topic", testCallbackB, null);

    try messaging.publish(allocator, "stats2.topic", "msg1");
    try messaging.publish(allocator, "stats2.topic", "msg2");

    const s1 = messaging.messagingStats();
    try std.testing.expectEqual(@as(u64, 2), s1.total_published);
    try std.testing.expectEqual(@as(u32, 2), s1.active_subscribers);
    try std.testing.expectEqual(@as(u64, 4), s1.total_delivered);
}

test "messaging init deinit cycle with allocator" {
    const allocator = std.testing.allocator;

    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    try std.testing.expect(messaging.isInitialized());
    _ = try messaging.subscribe(allocator, "cycle.test", testCallbackA, null);
    try messaging.publish(allocator, "cycle.test", "data");
    messaging.deinit();
    try std.testing.expect(!messaging.isInitialized());

    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    try std.testing.expect(messaging.isInitialized());
    messaging.deinit();
    try std.testing.expect(!messaging.isInitialized());
}

test "messaging patternMatches internal function" {
    try std.testing.expect(pattern_matching.patternMatches("foo.bar", "foo.bar"));
    try std.testing.expect(!pattern_matching.patternMatches("foo.bar", "foo.baz"));
    try std.testing.expect(pattern_matching.patternMatches("foo.*", "foo.bar"));
    try std.testing.expect(!pattern_matching.patternMatches("foo.*", "foo.bar.baz"));
    try std.testing.expect(pattern_matching.patternMatches("foo.#", "foo.bar"));
    try std.testing.expect(pattern_matching.patternMatches("foo.#", "foo.bar.baz"));
    try std.testing.expect(pattern_matching.patternMatches("foo.#", "foo.bar.baz.qux"));
    try std.testing.expect(pattern_matching.patternMatches("#", "anything.at.all"));
    try std.testing.expect(pattern_matching.patternMatches("", ""));
    try std.testing.expect(!pattern_matching.patternMatches("foo", ""));
}

test {
    std.testing.refAllDecls(@This());
}
