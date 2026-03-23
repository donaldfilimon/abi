//! Model registry — tracks available AI models and their capabilities.

const std = @import("std");

/// Metadata about a registered AI model.
pub const ModelInfo = struct {
    name: []const u8,
    provider: []const u8 = "unknown",
    context_window: u32 = 4096,
    supports_streaming: bool = true,
    supports_tools: bool = false,
};

/// Registry of available AI models.
pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    models: std.ArrayListUnmanaged(ModelInfo) = .empty,

    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ModelRegistry) void {
        self.models.deinit(self.allocator);
    }

    pub fn register(self: *ModelRegistry, info: ModelInfo) !void {
        try self.models.append(self.allocator, info);
    }

    pub fn count(self: *const ModelRegistry) usize {
        return self.models.items.len;
    }

    pub fn find(self: *const ModelRegistry, name: []const u8) ?ModelInfo {
        for (self.models.items) |m| {
            if (std.mem.eql(u8, m.name, name)) return m;
        }
        return null;
    }
};

test "ModelRegistry basic operations" {
    var reg = ModelRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.register(.{ .name = "gpt-4", .provider = "openai" });
    try std.testing.expectEqual(@as(usize, 1), reg.count());
    try std.testing.expect(reg.find("gpt-4") != null);
    try std.testing.expect(reg.find("nonexistent") == null);
}

test {
    std.testing.refAllDecls(@This());
}
