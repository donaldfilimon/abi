const std = @import("std");
const types = @import("../types.zig");

pub fn deinit(self: anytype) void {
    var it = self.models.iterator();
    while (it.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
    }
    self.models.deinit(self.allocator);

    self.router_instance.deinit();
    if (self.ensemble_instance) |*e| e.deinit();
    self.fallback_manager.deinit();
}

pub fn registerModel(self: anytype, config: types.ModelConfig) types.OrchestrationError!void {
    self.mutex.lock();
    defer self.mutex.unlock();

    if (self.models.contains(config.id)) {
        return types.OrchestrationError.DuplicateModelId;
    }

    const id_copy = self.allocator.dupe(u8, config.id) catch return types.OrchestrationError.OutOfMemory;
    errdefer self.allocator.free(id_copy);

    const entry = types.ModelEntry{
        .config = config,
    };

    self.models.put(self.allocator, id_copy, entry) catch return types.OrchestrationError.OutOfMemory;
}

pub fn unregisterModel(self: anytype, model_id: []const u8) types.OrchestrationError!void {
    self.mutex.lock();
    defer self.mutex.unlock();

    const kv = self.models.fetchRemove(model_id) orelse return types.OrchestrationError.ModelNotFound;
    self.allocator.free(kv.key);
}

pub fn getModel(self: anytype, model_id: []const u8) ?*types.ModelEntry {
    return self.models.getPtr(model_id);
}

pub fn setModelEnabled(
    self: anytype,
    model_id: []const u8,
    enabled: bool,
) types.OrchestrationError!void {
    self.mutex.lock();
    defer self.mutex.unlock();

    const entry = self.models.getPtr(model_id) orelse return types.OrchestrationError.ModelNotFound;
    entry.config.enabled = enabled;
}

pub fn setModelHealth(
    self: anytype,
    model_id: []const u8,
    status: types.HealthStatus,
) types.OrchestrationError!void {
    self.mutex.lock();
    defer self.mutex.unlock();

    const entry = self.models.getPtr(model_id) orelse return types.OrchestrationError.ModelNotFound;
    entry.status = status;
}

pub fn getStats(self: anytype) types.OrchestratorStats {
    self.mutex.lock();
    defer self.mutex.unlock();

    var stats = types.OrchestratorStats{};

    var it = self.models.iterator();
    while (it.next()) |entry| {
        stats.total_models += 1;
        if (entry.value_ptr.isAvailable()) {
            stats.available_models += 1;
        }
        stats.total_requests += entry.value_ptr.total_requests;
        stats.total_failures += entry.value_ptr.total_failures;
        stats.active_requests += entry.value_ptr.active_requests;
    }

    return stats;
}

pub fn listModels(self: anytype, allocator: std.mem.Allocator) ![][]const u8 {
    self.mutex.lock();
    defer self.mutex.unlock();

    var ids = std.ArrayListUnmanaged([]const u8).empty;
    errdefer ids.deinit(allocator);

    var it = self.models.iterator();
    while (it.next()) |entry| {
        try ids.append(allocator, entry.key_ptr.*);
    }

    return ids.toOwnedSlice(allocator);
}


test {
    std.testing.refAllDecls(@This());
}
