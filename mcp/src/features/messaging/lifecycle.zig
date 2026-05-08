const std = @import("std");
const state_mod = @import("state.zig");
pub const types = @import("types.zig");

const MessagingState = state_mod.MessagingState;
const MessagingConfig = types.MessagingConfig;
const MessagingError = types.MessagingError;

pub fn init(msg_state: *?*MessagingState, allocator: std.mem.Allocator, config: MessagingConfig) MessagingError!void {
    if (msg_state.* != null) return;
    msg_state.* = MessagingState.create(allocator, config) catch return error.OutOfMemory;
}

pub fn deinit(msg_state: *?*MessagingState) void {
    if (msg_state.*) |s| {
        s.destroy();
        msg_state.* = null;
    }
}

pub fn isInitialized(msg_state: ?*MessagingState) bool {
    return msg_state != null;
}
