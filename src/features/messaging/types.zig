const std = @import("std");
const core_config = @import("../../core/config/platform.zig");

pub const MessagingConfig = core_config.MessagingConfig;

/// Errors returned by messaging operations.
pub const MessagingError = error{
    MessagingDisabled,
    FeatureDisabled,
    ChannelFull,
    ChannelClosed,
    InvalidMessage,
    OutOfMemory,
    TopicNotFound,
    SubscriberNotFound,
};

/// Subscriber callback return value controlling message acknowledgement.
pub const DeliveryResult = enum {
    ok,
    retry,
    discard,
};

/// A single message delivered to subscribers (topic, payload, timestamp, id).
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

/// Aggregate statistics for the messaging subsystem.
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
