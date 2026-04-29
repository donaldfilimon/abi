const std = @import("std");
const types = @import("types.zig");

pub const EventReader = struct {
    fd: void = {},

    pub fn init() EventReader {
        return .{};
    }

    pub fn readEvent(self: *EventReader) !types.Key {
        _ = self;
        return error.Unsupported;
    }

    pub fn parseKey(byte: u8) types.Key {
        _ = byte;
        return .escape;
    }

    pub fn parseEscapeSequence(seq: []const u8) types.Key {
        _ = seq;
        return .escape;
    }
};
