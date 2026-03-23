//! AI Model management stub — no-op when AI is disabled.

const std = @import("std");

pub const ModelInfo = struct {
    name: []const u8 = "",
    provider: []const u8 = "",
    context_length: u32 = 0,
    max_tokens: u32 = 0,
    supports_streaming: bool = false,
};

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *ModelRegistry) void {}

    pub fn lookup(_: *const ModelRegistry, _: []const u8) ?ModelInfo {
        return null;
    }

    pub fn count(_: *const ModelRegistry) usize {
        return 0;
    }
};

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
