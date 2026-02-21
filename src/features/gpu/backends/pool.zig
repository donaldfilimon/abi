const std = @import("std");
const backend_mod = @import("../backend.zig");
const backend_factory = @import("../backend_factory.zig");

pub const BackendPool = struct {
    allocator: std.mem.Allocator,
    instances: std.AutoHashMapUnmanaged(backend_mod.Backend, *backend_factory.BackendInstance),

    pub fn init(allocator: std.mem.Allocator) BackendPool {
        return .{
            .allocator = allocator,
            .instances = .{},
        };
    }

    pub fn deinit(self: *BackendPool) void {
        var iter = self.instances.iterator();
        while (iter.next()) |entry| {
            backend_factory.destroyBackend(entry.value_ptr.*);
        }
        self.instances.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn getOrCreate(
        self: *BackendPool,
        backend_type: backend_mod.Backend,
    ) !*backend_factory.BackendInstance {
        if (self.instances.get(backend_type)) |instance| {
            return instance;
        }

        const instance = try backend_factory.createBackend(self.allocator, backend_type);
        errdefer backend_factory.destroyBackend(instance);

        try self.instances.put(self.allocator, backend_type, instance);
        return instance;
    }
};

test "pool rejects unavailable explicit backend" {
    var pool = BackendPool.init(std.testing.allocator);
    defer pool.deinit();

    if (!backend_factory.isBackendAvailable(.tpu)) {
        try std.testing.expectError(
            error.BackendNotAvailable,
            pool.getOrCreate(.tpu),
        );
    }
}
