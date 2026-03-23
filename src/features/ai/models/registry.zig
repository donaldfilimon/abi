//! Model registry — tracks available models and their metadata.

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

test "ModelRegistry basic operations" {
    var reg = ModelRegistry.init(std.testing.allocator);
    defer reg.deinit();
    try std.testing.expectEqual(@as(usize, 0), reg.count());
    try std.testing.expect(reg.lookup("gpt-4") == null);
}

test {
    std.testing.refAllDecls(@This());
}
