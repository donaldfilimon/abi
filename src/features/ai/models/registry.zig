//! Model registry — tracks available AI models and their metadata.

const std = @import("std");

pub const ModelInfo = struct {
    name: []const u8 = "",
    provider: []const u8 = "",
    context_length: u32 = 0,
    parameters: u64 = 0,
};

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(ModelInfo) = .empty,

    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ModelRegistry) void {
        self.entries.deinit(self.allocator);
    }

    pub fn count(self: *const ModelRegistry) usize {
        return self.entries.items.len;
    }

    pub fn register(self: *ModelRegistry, info: ModelInfo) !void {
        try self.entries.append(self.allocator, info);
    }

    pub fn get(self: *const ModelRegistry, name: []const u8) ?ModelInfo {
        for (self.entries.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }
};

test "registry basic operations" {
    var reg = ModelRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try std.testing.expectEqual(@as(usize, 0), reg.count());

    try reg.register(.{ .name = "test-model", .provider = "local", .context_length = 4096 });
    try std.testing.expectEqual(@as(usize, 1), reg.count());

    const found = reg.get("test-model");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("local", found.?.provider);

    try std.testing.expect(reg.get("nonexistent") == null);
}
