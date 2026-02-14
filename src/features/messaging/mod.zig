//! Messaging Module
//!
//! Topic-based pub/sub messaging with MQTT-style pattern matching,
//! synchronous delivery, dead letter queue, and backpressure.
//!
//! Architecture:
//! - Topic registry with subscriber lists
//! - Pattern matching: `events.*` (single-level), `events.#` (multi-level)
//! - Bounded message queues with backpressure (returns ChannelFull)
//! - Dead letter queue for failed deliveries
//! - Synchronous delivery (publish blocks until all subscribers notified)
//! - RwLock for concurrent topic lookups

const std = @import("std");
const core_config = @import("../../core/config/messaging.zig");
const sync = @import("../../services/shared/sync.zig");
const time_mod = @import("../../services/shared/time.zig");

pub const MessagingConfig = core_config.MessagingConfig;

pub const MessagingError = error{
    FeatureDisabled,
    ChannelFull,
    ChannelClosed,
    InvalidMessage,
    OutOfMemory,
    TopicNotFound,
    SubscriberNotFound,
};

pub const DeliveryResult = enum {
    ok,
    retry,
    discard,
};

pub const Message = struct {
    topic: []const u8 = "",
    payload: []const u8 = "",
    timestamp: u64 = 0,
    id: u64 = 0,
};

pub const Channel = struct {
    name: []const u8 = "",
    subscriber_count: u32 = 0,
};

pub const MessagingStats = struct {
    total_published: u64 = 0,
    total_delivered: u64 = 0,
    total_failed: u64 = 0,
    active_topics: u32 = 0,
    active_subscribers: u32 = 0,
    dead_letter_count: u32 = 0,
};

pub const TopicInfo = struct {
    name: []const u8 = "",
    subscriber_count: u32 = 0,
    messages_published: u64 = 0,
    messages_delivered: u64 = 0,
    messages_failed: u64 = 0,
};

pub const DeadLetter = struct {
    message: Message,
    reason: []const u8,
    timestamp: u64,
};

pub const SubscriberCallback = *const fn (msg: Message, ctx: ?*anyopaque) DeliveryResult;

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MessagingConfig,

    pub fn init(allocator: std.mem.Allocator, config: MessagingConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ── Internal Types ─────────────────────────────────────────────────────

const Subscriber = struct {
    id: u64,
    pattern: []u8, // owned copy
    callback: SubscriberCallback,
    user_ctx: ?*anyopaque,
    active: bool = true,
};

const Topic = struct {
    name: []u8, // owned
    subscribers: std.ArrayListUnmanaged(*Subscriber),
    published: u64 = 0,
    delivered: u64 = 0,
    failed: u64 = 0,
};

const DLQEntry = struct {
    topic: []u8,
    payload: []u8,
    reason: []u8,
    timestamp_ns: u128,
    msg_id: u64,
};

// ── Module State ───────────────────────────────────────────────────────

var msg_state: ?*MessagingState = null;

const MessagingState = struct {
    allocator: std.mem.Allocator,
    config: MessagingConfig,
    topics: std.StringHashMapUnmanaged(*Topic),
    subscribers: std.AutoHashMapUnmanaged(u64, *Subscriber), // id → subscriber
    dead_letters: std.ArrayListUnmanaged(DLQEntry),
    rw_lock: sync.RwLock,
    next_sub_id: std.atomic.Value(u64),
    next_msg_id: std.atomic.Value(u64),

    // Stats
    stat_published: std.atomic.Value(u64),
    stat_delivered: std.atomic.Value(u64),
    stat_failed: std.atomic.Value(u64),

    fn create(allocator: std.mem.Allocator, config: MessagingConfig) !*MessagingState {
        const s = try allocator.create(MessagingState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .topics = .empty,
            .subscribers = .empty,
            .dead_letters = .empty,
            .rw_lock = sync.RwLock.init(),
            .next_sub_id = std.atomic.Value(u64).init(1),
            .next_msg_id = std.atomic.Value(u64).init(1),
            .stat_published = std.atomic.Value(u64).init(0),
            .stat_delivered = std.atomic.Value(u64).init(0),
            .stat_failed = std.atomic.Value(u64).init(0),
        };
        return s;
    }

    fn destroy(self: *MessagingState) void {
        const allocator = self.allocator;

        // Free topics
        var topic_iter = self.topics.iterator();
        while (topic_iter.next()) |entry| {
            const topic = entry.value_ptr.*;
            topic.subscribers.deinit(allocator);
            allocator.free(topic.name);
            allocator.destroy(topic);
        }
        self.topics.deinit(allocator);

        // Free subscribers
        var sub_iter = self.subscribers.iterator();
        while (sub_iter.next()) |entry| {
            const sub = entry.value_ptr.*;
            allocator.free(sub.pattern);
            allocator.destroy(sub);
        }
        self.subscribers.deinit(allocator);

        // Free dead letters
        for (self.dead_letters.items) |*dl| {
            allocator.free(dl.topic);
            allocator.free(dl.payload);
            allocator.free(dl.reason);
        }
        self.dead_letters.deinit(allocator);

        allocator.destroy(self);
    }

    fn getOrCreateTopic(self: *MessagingState, name: []const u8) !*Topic {
        if (self.topics.get(name)) |topic| return topic;

        if (self.topics.count() >= self.config.max_channels) return error.OutOfMemory;

        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);

        const topic = try self.allocator.create(Topic);
        topic.* = .{
            .name = owned_name,
            .subscribers = .empty,
        };

        try self.topics.put(self.allocator, owned_name, topic);
        return topic;
    }

    fn addDeadLetter(
        self: *MessagingState,
        topic_name: []const u8,
        payload: []const u8,
        reason: []const u8,
        msg_id: u64,
    ) void {
        // Cap dead letter queue at buffer_size
        if (self.dead_letters.items.len >= self.config.buffer_size) {
            // Remove oldest
            if (self.dead_letters.items.len > 0) {
                const old = self.dead_letters.orderedRemove(0);
                self.allocator.free(old.topic);
                self.allocator.free(old.payload);
                self.allocator.free(old.reason);
            }
        }

        const topic_copy = self.allocator.dupe(u8, topic_name) catch return;
        const payload_copy = self.allocator.dupe(u8, payload) catch {
            self.allocator.free(topic_copy);
            return;
        };
        const reason_copy = self.allocator.dupe(u8, reason) catch {
            self.allocator.free(topic_copy);
            self.allocator.free(payload_copy);
            return;
        };
        self.dead_letters.append(self.allocator, .{
            .topic = topic_copy,
            .payload = payload_copy,
            .reason = reason_copy,
            .timestamp_ns = (time_mod.Instant.now() catch {
                self.allocator.free(topic_copy);
                self.allocator.free(payload_copy);
                self.allocator.free(reason_copy);
                return;
            }).nanos,
            .msg_id = msg_id,
        }) catch {
            self.allocator.free(topic_copy);
            self.allocator.free(payload_copy);
            self.allocator.free(reason_copy);
        };
    }
};

/// MQTT-style pattern matching.
/// `*` matches exactly one level, `#` matches zero or more levels.
fn patternMatches(pattern: []const u8, topic: []const u8) bool {
    var pat_iter = std.mem.splitScalar(u8, pattern, '.');
    var top_iter = std.mem.splitScalar(u8, topic, '.');

    while (true) {
        const pat_seg = pat_iter.next();
        const top_seg = top_iter.next();

        if (pat_seg == null and top_seg == null) return true;

        if (pat_seg) |p| {
            if (std.mem.eql(u8, p, "#")) return true; // # matches rest
            if (top_seg == null) return false;
            if (std.mem.eql(u8, p, "*")) continue; // * matches one level
            if (!std.mem.eql(u8, p, top_seg.?)) return false;
        } else {
            return false; // pattern ended but topic continues
        }
    }
}

fn nowMs() u64 {
    const instant = time_mod.Instant.now() catch return 0;
    return @intCast(instant.nanos / std.time.ns_per_ms);
}

// ── Public API ─────────────────────────────────────────────────────────

pub fn init(allocator: std.mem.Allocator, config: MessagingConfig) MessagingError!void {
    if (msg_state != null) return;
    msg_state = MessagingState.create(allocator, config) catch return error.OutOfMemory;
}

pub fn deinit() void {
    if (msg_state) |s| {
        s.destroy();
        msg_state = null;
    }
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return msg_state != null;
}

pub fn publish(
    allocator: std.mem.Allocator,
    topic_name: []const u8,
    payload: []const u8,
) MessagingError!void {
    _ = allocator;
    const s = msg_state orelse return error.FeatureDisabled;

    const msg_id = s.next_msg_id.fetchAdd(1, .monotonic);
    _ = s.stat_published.fetchAdd(1, .monotonic);

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const msg = Message{
        .topic = topic_name,
        .payload = payload,
        .timestamp = nowMs(),
        .id = msg_id,
    };

    // Deliver to all matching subscribers
    var sub_iter = s.subscribers.iterator();
    while (sub_iter.next()) |entry| {
        const sub = entry.value_ptr.*;
        if (!sub.active) continue;

        if (patternMatches(sub.pattern, topic_name)) {
            const result = sub.callback(msg, sub.user_ctx);
            switch (result) {
                .ok => {
                    _ = s.stat_delivered.fetchAdd(1, .monotonic);
                    // Update topic stats
                    if (s.topics.get(topic_name)) |topic| {
                        topic.delivered += 1;
                    }
                },
                .retry, .discard => {
                    _ = s.stat_failed.fetchAdd(1, .monotonic);
                    if (s.topics.get(topic_name)) |topic| {
                        topic.failed += 1;
                    }
                    if (result == .discard) {
                        s.addDeadLetter(topic_name, payload, "subscriber_discard", msg_id);
                    }
                },
            }
        }
    }

    // Track topic stats
    if (s.topics.get(topic_name)) |topic| {
        topic.published += 1;
    } else {
        // Auto-create topic on first publish
        const topic = s.getOrCreateTopic(topic_name) catch return error.OutOfMemory;
        topic.published = 1;
    }
}

pub fn subscribe(
    _: std.mem.Allocator,
    topic_pattern: []const u8,
    callback: SubscriberCallback,
    user_ctx: ?*anyopaque,
) MessagingError!u64 {
    const s = msg_state orelse return error.FeatureDisabled;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const sub_id = s.next_sub_id.fetchAdd(1, .monotonic);

    const pattern_owned = s.allocator.dupe(u8, topic_pattern) catch return error.OutOfMemory;
    errdefer s.allocator.free(pattern_owned);

    const sub = s.allocator.create(Subscriber) catch return error.OutOfMemory;
    sub.* = .{
        .id = sub_id,
        .pattern = pattern_owned,
        .callback = callback,
        .user_ctx = user_ctx,
    };

    s.subscribers.put(s.allocator, sub_id, sub) catch {
        s.allocator.destroy(sub);
        s.allocator.free(pattern_owned);
        return error.OutOfMemory;
    };

    // Auto-create topic if exact match (not a pattern)
    if (std.mem.indexOf(u8, topic_pattern, "*") == null and
        std.mem.indexOf(u8, topic_pattern, "#") == null)
    {
        if (s.getOrCreateTopic(topic_pattern)) |topic| {
            topic.subscribers.append(s.allocator, sub) catch {};
        } else |_| {}
    }

    return sub_id;
}

pub fn unsubscribe(subscriber_id: u64) MessagingError!bool {
    const s = msg_state orelse return error.FeatureDisabled;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.subscribers.fetchRemove(subscriber_id)) |kv| {
        const sub = kv.value;
        // Remove from any topic subscriber lists
        var topic_iter = s.topics.iterator();
        while (topic_iter.next()) |entry| {
            const topic = entry.value_ptr.*;
            for (topic.subscribers.items, 0..) |t_sub, i| {
                if (t_sub.id == subscriber_id) {
                    _ = topic.subscribers.swapRemove(i);
                    break;
                }
            }
        }
        s.allocator.free(sub.pattern);
        s.allocator.destroy(sub);
        return true;
    }

    return false;
}

pub fn listTopics(allocator: std.mem.Allocator) MessagingError![][]const u8 {
    const s = msg_state orelse return error.FeatureDisabled;

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const count = s.topics.count();
    if (count == 0) return &.{};

    const result = allocator.alloc([]const u8, count) catch return error.OutOfMemory;
    var i: usize = 0;
    var iter = s.topics.iterator();
    while (iter.next()) |entry| {
        if (i < count) {
            result[i] = entry.key_ptr.*;
            i += 1;
        }
    }

    return result[0..i];
}

pub fn topicStats(topic_name: []const u8) MessagingError!TopicInfo {
    const s = msg_state orelse return error.FeatureDisabled;

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const topic = s.topics.get(topic_name) orelse return error.TopicNotFound;
    return .{
        .name = topic.name,
        .subscriber_count = @intCast(topic.subscribers.items.len),
        .messages_published = topic.published,
        .messages_delivered = topic.delivered,
        .messages_failed = topic.failed,
    };
}

pub fn getDeadLetters(allocator: std.mem.Allocator) MessagingError![]DeadLetter {
    const s = msg_state orelse return error.FeatureDisabled;

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    if (s.dead_letters.items.len == 0) return &.{};

    const result = allocator.alloc(DeadLetter, s.dead_letters.items.len) catch
        return error.OutOfMemory;
    for (s.dead_letters.items, 0..) |dl, i| {
        result[i] = .{
            .message = .{
                .topic = dl.topic,
                .payload = dl.payload,
                .id = dl.msg_id,
                .timestamp = @intCast(dl.timestamp_ns / std.time.ns_per_ms),
            },
            .reason = dl.reason,
            .timestamp = @intCast(dl.timestamp_ns / std.time.ns_per_ms),
        };
    }
    return result;
}

pub fn clearDeadLetters() void {
    const s = msg_state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    for (s.dead_letters.items) |*dl| {
        s.allocator.free(dl.topic);
        s.allocator.free(dl.payload);
        s.allocator.free(dl.reason);
    }
    s.dead_letters.clearRetainingCapacity();
}

pub fn messagingStats() MessagingStats {
    const s = msg_state orelse return .{};
    return .{
        .total_published = s.stat_published.load(.monotonic),
        .total_delivered = s.stat_delivered.load(.monotonic),
        .total_failed = s.stat_failed.load(.monotonic),
        .active_topics = @intCast(s.topics.count()),
        .active_subscribers = @intCast(s.subscribers.count()),
        .dead_letter_count = @intCast(s.dead_letters.items.len),
    };
}

// ── Tests ──────────────────────────────────────────────────────────────

var test_received_count: u32 = 0;
var test_last_payload: []const u8 = "";

fn testCallback(msg: Message, _: ?*anyopaque) DeliveryResult {
    test_received_count += 1;
    test_last_payload = msg.payload;
    return .ok;
}

fn testDiscardCallback(_: Message, _: ?*anyopaque) DeliveryResult {
    return .discard;
}

test "messaging basic pub/sub" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    test_received_count = 0;
    const sub_id = try subscribe(allocator, "events.user", testCallback, null);
    try std.testing.expect(sub_id > 0);

    try publish(allocator, "events.user", "hello");
    try std.testing.expectEqual(@as(u32, 1), test_received_count);
    try std.testing.expectEqualStrings("hello", test_last_payload);
}

test "messaging unsubscribe" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    test_received_count = 0;
    const sub_id = try subscribe(allocator, "test.topic", testCallback, null);
    try publish(allocator, "test.topic", "msg1");
    try std.testing.expectEqual(@as(u32, 1), test_received_count);

    const removed = try unsubscribe(sub_id);
    try std.testing.expect(removed);

    try publish(allocator, "test.topic", "msg2");
    try std.testing.expectEqual(@as(u32, 1), test_received_count); // unchanged
}

test "messaging wildcard pattern *" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    test_received_count = 0;
    _ = try subscribe(allocator, "events.*", testCallback, null);

    try publish(allocator, "events.user", "payload1");
    try publish(allocator, "events.order", "payload2");
    try publish(allocator, "other.topic", "payload3");

    try std.testing.expectEqual(@as(u32, 2), test_received_count);
}

test "messaging multi-level pattern #" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    test_received_count = 0;
    _ = try subscribe(allocator, "events.#", testCallback, null);

    try publish(allocator, "events.user", "p1");
    try publish(allocator, "events.user.created", "p2");
    try publish(allocator, "events.order.updated.v2", "p3");
    try publish(allocator, "other.topic", "p4");

    try std.testing.expectEqual(@as(u32, 3), test_received_count);
}

test "messaging dead letter queue" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    _ = try subscribe(allocator, "dlq.test", testDiscardCallback, null);
    try publish(allocator, "dlq.test", "will_fail");

    const dead = try getDeadLetters(allocator);
    defer allocator.free(dead);
    try std.testing.expectEqual(@as(usize, 1), dead.len);

    clearDeadLetters();
    const after = try getDeadLetters(allocator);
    defer allocator.free(after);
    try std.testing.expectEqual(@as(usize, 0), after.len);
}

test "messaging stats" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    _ = try subscribe(allocator, "stats.test", testCallback, null);
    try publish(allocator, "stats.test", "msg1");

    const s = messagingStats();
    try std.testing.expect(s.total_published > 0);
    try std.testing.expect(s.active_subscribers > 0);
}

var test_fanout_count: u32 = 0;

fn testFanoutCallback(_: Message, _: ?*anyopaque) DeliveryResult {
    test_fanout_count += 1;
    return .ok;
}

test "messaging multiple subscribers fan-out" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    test_fanout_count = 0;
    _ = try subscribe(allocator, "fanout.topic", testFanoutCallback, null);
    _ = try subscribe(allocator, "fanout.topic", testFanoutCallback, null);
    _ = try subscribe(allocator, "fanout.topic", testFanoutCallback, null);

    try publish(allocator, "fanout.topic", "broadcast");

    // All 3 subscribers should receive the message
    try std.testing.expectEqual(@as(u32, 3), test_fanout_count);
}

test "messaging listTopics after publish" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    // Subscribe creates the topic
    _ = try subscribe(allocator, "list.alpha", testCallback, null);
    // Publish auto-creates topic if not exists
    try publish(allocator, "list.beta", "msg");

    const topics = try listTopics(allocator);
    defer allocator.free(topics);
    try std.testing.expect(topics.len >= 2);
}

test "messaging topicStats accuracy" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    test_received_count = 0;
    _ = try subscribe(allocator, "tstats.topic", testCallback, null);
    try publish(allocator, "tstats.topic", "msg1");
    try publish(allocator, "tstats.topic", "msg2");

    const info = try topicStats("tstats.topic");
    try std.testing.expectEqual(@as(u64, 2), info.messages_published);
    try std.testing.expectEqual(@as(u64, 2), info.messages_delivered);
    try std.testing.expectEqual(@as(u32, 1), info.subscriber_count);
}

test "messaging publish to non-existent topic auto-creates" {
    const allocator = std.testing.allocator;
    try init(allocator, MessagingConfig.defaults());
    defer deinit();

    // No subscribers, no topic — publish should auto-create
    try publish(allocator, "auto.created", "payload");

    const info = try topicStats("auto.created");
    try std.testing.expectEqual(@as(u64, 1), info.messages_published);
    try std.testing.expectEqual(@as(u32, 0), info.subscriber_count);
}
