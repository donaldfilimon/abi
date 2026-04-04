//! Messaging Stub Module
//!
//! API-compatible no-op implementations when messaging is disabled.

const std = @import("std");
const stub_context = @import("../core/stub_helpers.zig");
pub const types = @import("types.zig");

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

const feature = stub_context.StubFeature(MessagingConfig, MessagingError);
pub const Context = feature.Context;
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;

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

test {
    std.testing.refAllDecls(@This());
}
