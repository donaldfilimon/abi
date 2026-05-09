const std = @import("std");

pub const ModelInfo = struct {
    name: []const u8,
    provider: []const u8 = "unknown",
    context_window: u32 = 4096,
};

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    models: std.StringHashMapUnmanaged(ModelInfo),

    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{
            .allocator = allocator,
            .models = .{},
        };
    }

    pub fn deinit(self: *ModelRegistry) void {
        self.models.deinit(self.allocator);
    }

    pub fn register(self: *ModelRegistry, name: []const u8, info: ModelInfo) !void {
        try self.models.put(self.allocator, name, info);
    }

    pub fn get(self: *const ModelRegistry, name: []const u8) ?ModelInfo {
        return self.models.get(name);
    }

    pub fn count(self: *const ModelRegistry) usize {
        return self.models.count();
    }
};

test {
    std.testing.refAllDecls(@This());
}
