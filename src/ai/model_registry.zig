//! Model Registry
//!
//! Tracks registered AI models with their metadata including name,
//! parameter count, and description. Used for model discovery and
//! capability management.

const std = @import("std");

pub const ModelRegistryError = error{
    DuplicateModel,
};

pub const ModelInfo = struct {
    name: []const u8,
    parameters: u64,
    description: []const u8 = "",
};

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    models: std.ArrayListUnmanaged(ModelInfo) = .{},

    pub fn init(allocator: std.mem.Allocator) ModelRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ModelRegistry) void {
        for (self.models.items) |model| {
            self.allocator.free(model.name);
            if (model.description.len > 0) {
                self.allocator.free(model.description);
            }
        }
        self.models.deinit(self.allocator);
    }

    pub fn register(self: *ModelRegistry, model: ModelInfo) ModelRegistryError!void {
        if (self.findIndex(model.name) != null) return ModelRegistryError.DuplicateModel;
        const name_copy = try self.allocator.dupe(u8, model.name);
        errdefer self.allocator.free(name_copy);
        const desc_copy = if (model.description.len > 0)
            try self.allocator.dupe(u8, model.description)
        else
            "";
        errdefer if (desc_copy.len > 0) self.allocator.free(desc_copy);
        try self.models.append(self.allocator, .{
            .name = name_copy,
            .parameters = model.parameters,
            .description = desc_copy,
        });
    }

    pub fn find(self: *ModelRegistry, name: []const u8) ?ModelInfo {
        for (self.models.items) |model| {
            if (std.mem.eql(u8, model.name, name)) return model;
        }
        return null;
    }

    pub fn update(self: *ModelRegistry, model: ModelInfo) !bool {
        const index = self.findIndex(model.name) orelse return false;
        const desc_copy = if (model.description.len > 0)
            try self.allocator.dupe(u8, model.description)
        else
            "";
        const name_copy = try self.allocator.dupe(u8, model.name);

        const existing = self.models.items[index];
        self.allocator.free(existing.name);
        if (existing.description.len > 0) {
            self.allocator.free(existing.description);
        }

        self.models.items[index] = .{
            .name = name_copy,
            .parameters = model.parameters,
            .description = desc_copy,
        };
        return true;
    }

    pub fn remove(self: *ModelRegistry, name: []const u8) bool {
        const index = self.findIndex(name) orelse return false;
        const removed = self.models.swapRemove(index);
        self.allocator.free(removed.name);
        if (removed.description.len > 0) {
            self.allocator.free(removed.description);
        }
        return true;
    }

    pub fn list(self: *ModelRegistry) []const ModelInfo {
        return self.models.items;
    }

    pub fn count(self: *ModelRegistry) usize {
        return self.models.items.len;
    }

    fn findIndex(self: *ModelRegistry, name: []const u8) ?usize {
        for (self.models.items, 0..) |model, i| {
            if (std.mem.eql(u8, model.name, name)) return i;
        }
        return null;
    }
};

test "model registry register update remove" {
    var registry = ModelRegistry.init(std.testing.allocator);
    defer registry.deinit();

    try registry.register(.{ .name = "tiny", .parameters = 123, .description = "test" });
    try std.testing.expectEqual(@as(usize, 1), registry.count());
    try std.testing.expectError(ModelRegistryError.DuplicateModel, registry.register(.{
        .name = "tiny",
        .parameters = 100,
    }));

    const updated = try registry.update(.{ .name = "tiny", .parameters = 456 });
    try std.testing.expect(updated);
    const info = registry.find("tiny") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 456), info.parameters);

    try std.testing.expect(registry.remove("tiny"));
    try std.testing.expectEqual(@as(usize, 0), registry.count());
}
