const std = @import("std");

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

    pub fn register(self: *ModelRegistry, model: ModelInfo) !void {
        const name_copy = try self.allocator.dupe(u8, model.name);
        const desc_copy = if (model.description.len > 0)
            try self.allocator.dupe(u8, model.description)
        else
            "";
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
};
