//! Messaging Stub Module
//!
//! API-compatible no-op implementations when messaging is disabled.

const std = @import("std");
const core_config = @import("../../core/config/messaging.zig");
const stub_context = @import("../../core/stub_context.zig");

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

pub const Context = stub_context.StubContextWithConfig(MessagingConfig);

pub fn init(_: std.mem.Allocator, _: MessagingConfig) MessagingError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

pub fn publish(_: std.mem.Allocator, _: []const u8, _: []const u8) MessagingError!void {
    return error.FeatureDisabled;
}
pub fn subscribe(
    _: std.mem.Allocator,
    _: []const u8,
    _: SubscriberCallback,
    _: ?*anyopaque,
) MessagingError!u64 {
    return error.FeatureDisabled;
}
pub fn unsubscribe(_: u64) MessagingError!bool {
    return error.FeatureDisabled;
}
pub fn listTopics(_: std.mem.Allocator) MessagingError![][]const u8 {
    return error.FeatureDisabled;
}
pub fn topicStats(_: []const u8) MessagingError!TopicInfo {
    return error.FeatureDisabled;
}
pub fn getDeadLetters(_: std.mem.Allocator) MessagingError![]DeadLetter {
    return error.FeatureDisabled;
}
pub fn clearDeadLetters() void {}
pub fn messagingStats() MessagingStats {
    return .{};
}
