//! Local scheduler connector stub — returns ConnectorsDisabled for all operations.

const std = @import("std");

pub const Config = struct {
    url: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.url);
        self.* = undefined;
    }

    pub fn endpoint(_: *const Config, _: std.mem.Allocator, _: []const u8) ![]u8 {
        return error.ConnectorsDisabled;
    }

    pub fn healthUrl(_: *const Config, _: std.mem.Allocator) ![]u8 {
        return error.ConnectorsDisabled;
    }

    pub fn submitUrl(_: *const Config, _: std.mem.Allocator) ![]u8 {
        return error.ConnectorsDisabled;
    }
};

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return error.ConnectorsDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
