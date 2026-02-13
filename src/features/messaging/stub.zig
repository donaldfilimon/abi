//! Messaging Stub Module
//!
//! API-compatible no-op implementations when messaging is disabled.

const std = @import("std");
const core_config = @import("../../core/config/messaging.zig");

pub const MessagingConfig = core_config.MessagingConfig;

pub const MessagingError = error{
    FeatureDisabled,
    ChannelFull,
    ChannelClosed,
    InvalidMessage,
    OutOfMemory,
};

pub const Message = struct {
    topic: []const u8 = "",
    payload: []const u8 = "",
    timestamp: u64 = 0,
};

pub const Channel = struct {
    name: []const u8 = "",
    subscriber_count: u32 = 0,
};

pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: MessagingConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

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
pub fn subscribe(_: std.mem.Allocator, _: []const u8) MessagingError!void {
    return error.FeatureDisabled;
}
