//! Stub for AI feature when disabled
const std = @import("std");

pub const agent = struct {
    pub const Agent = struct {
        pub fn init(allocator: std.mem.Allocator, config: anytype) !Agent {
            _ = allocator;
            _ = config;
            return error.AiDisabled;
        }
        pub fn deinit(self: *Agent) void {
            _ = self;
        }
        pub fn process(self: *Agent, input: []const u8, allocator: std.mem.Allocator) ![]u8 {
            _ = self;
            _ = input;
            _ = allocator;
            return error.AiDisabled;
        }
    };
};

pub fn isEnabled() bool {
    return false;
}

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    return error.AiDisabled;
}

pub fn deinit() void {}
