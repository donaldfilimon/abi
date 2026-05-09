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
pub const types = @import("types.zig");

// Submodules
const state_mod = @import("state.zig");
const lifecycle = @import("lifecycle.zig");
const publish_mod = @import("publish.zig");
const queries = @import("queries.zig");
const subscriptions = @import("subscriptions.zig");

const MessagingState = state_mod.MessagingState;

// Re-export public types from types.zig
pub const MessagingConfig = types.MessagingConfig;
pub const MessagingError = types.MessagingError;
pub const Error = MessagingError;
pub const DeliveryResult = types.DeliveryResult;
pub const Message = types.Message;
pub const Channel = types.Channel;
pub const MessagingStats = types.MessagingStats;
pub const TopicInfo = types.TopicInfo;
pub const DeadLetter = types.DeadLetter;
pub const SubscriberCallback = types.SubscriberCallback;

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

// ── Module State ───────────────────────────────────────────────────────

var msg_state: ?*MessagingState = null;

// ── Public API ─────────────────────────────────────────────────────────

/// Initialize the global messaging singleton.
pub fn init(allocator: std.mem.Allocator, config: MessagingConfig) MessagingError!void {
    try lifecycle.init(&msg_state, allocator, config);
}

/// Tear down the messaging subsystem and all topics/subscribers.
pub fn deinit() void {
    lifecycle.deinit(&msg_state);
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return lifecycle.isInitialized(msg_state);
}

/// Publish a message to a topic. Delivers synchronously to all matching
/// subscribers (exact match and MQTT-style wildcards `*`, `#`).
pub fn publish(
    allocator: std.mem.Allocator,
    topic_name: []const u8,
    payload: []const u8,
) MessagingError!void {
    _ = allocator;
    const s = msg_state orelse return error.FeatureDisabled;
    try publish_mod.publish(s, topic_name, payload);
}

/// Register a callback for messages matching `topic_pattern`.
/// Returns a subscriber ID that can be passed to `unsubscribe`.
pub fn subscribe(
    _: std.mem.Allocator,
    topic_pattern: []const u8,
    callback: SubscriberCallback,
    user_ctx: ?*anyopaque,
) MessagingError!u64 {
    const s = msg_state orelse return error.FeatureDisabled;
    return subscriptions.subscribe(s, topic_pattern, callback, user_ctx);
}

/// Remove a subscriber by ID. Returns `true` if the subscriber was found.
pub fn unsubscribe(subscriber_id: u64) MessagingError!bool {
    const s = msg_state orelse return error.FeatureDisabled;
    return subscriptions.unsubscribe(s, subscriber_id);
}

/// List all active topic names. Caller owns the returned slice.
pub fn listTopics(allocator: std.mem.Allocator) MessagingError![][]const u8 {
    const s = msg_state orelse return error.FeatureDisabled;
    return queries.listTopics(s, allocator);
}

pub fn topicStats(topic_name: []const u8) MessagingError!TopicInfo {
    const s = msg_state orelse return error.FeatureDisabled;
    return queries.topicStats(s, topic_name);
}

/// Retrieve all dead letter entries. Caller owns the returned slice.
pub fn getDeadLetters(allocator: std.mem.Allocator) MessagingError![]DeadLetter {
    const s = msg_state orelse return error.FeatureDisabled;
    return queries.getDeadLetters(s, allocator);
}

/// Discard all dead letter entries.
pub fn clearDeadLetters() void {
    const s = msg_state orelse return;
    queries.clearDeadLetters(s);
}

/// Snapshot publish/deliver/fail counters and active topic/subscriber counts.
pub fn messagingStats() MessagingStats {
    const s = msg_state orelse return .{};
    return queries.messagingStats(s);
}

test {
    std.testing.refAllDecls(@This());
}
