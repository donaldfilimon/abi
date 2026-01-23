//! Federated learning registry and coordinator utilities.
const std = @import("std");
const time = @import("../../shared/utils.zig");

pub const NodeInfo = struct {
    id: []const u8,
    last_update: i64,
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(NodeInfo) = .{},

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Registry) void {
        for (self.nodes.items) |node| {
            self.allocator.free(node.id);
        }
        self.nodes.deinit(self.allocator);
    }

    pub fn touch(self: *Registry, id: []const u8) !void {
        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, id)) {
                node.last_update = time.nowSeconds();
                return;
            }
        }
        const copy = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(copy);
        try self.nodes.append(self.allocator, .{
            .id = copy,
            .last_update = time.nowSeconds(),
        });
    }

    pub fn remove(self: *Registry, id: []const u8) bool {
        for (self.nodes.items, 0..) |node, i| {
            if (std.mem.eql(u8, node.id, id)) {
                const removed = self.nodes.swapRemove(i);
                self.allocator.free(removed.id);
                return true;
            }
        }
        return false;
    }

    pub fn list(self: *const Registry) []const NodeInfo {
        return self.nodes.items;
    }

    pub fn count(self: *const Registry) usize {
        return self.nodes.items.len;
    }

    pub fn prune(self: *Registry, max_age_seconds: i64) usize {
        if (max_age_seconds <= 0) return 0;
        const now = time.nowSeconds();
        var removed: usize = 0;
        var i: usize = 0;
        while (i < self.nodes.items.len) {
            const node = self.nodes.items[i];
            if (now - node.last_update > max_age_seconds) {
                const removed_node = self.nodes.swapRemove(i);
                self.allocator.free(removed_node.id);
                removed += 1;
                continue;
            }
            i += 1;
        }
        return removed;
    }
};

pub const CoordinatorError = error{
    InsufficientUpdates,
    InvalidUpdate,
};

pub const AggregationStrategy = enum {
    mean,
    weighted_mean,
};

pub const ModelUpdateView = struct {
    node_id: []const u8,
    step: u64,
    weights: []const f32,
    sample_count: u32 = 1,
};

pub const ModelUpdate = struct {
    node_id: []const u8,
    step: u64,
    timestamp: u64,
    weights: []f32,
    sample_count: u32,

    pub fn deinit(self: *ModelUpdate, allocator: std.mem.Allocator) void {
        allocator.free(self.node_id);
        allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const CoordinatorConfig = struct {
    min_updates: usize = 1,
    max_updates: usize = 64,
    max_staleness_seconds: u64 = 300,
    strategy: AggregationStrategy = .mean,
};

pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    registry: Registry,
    updates: std.ArrayListUnmanaged(ModelUpdate),
    global_weights: []f32,
    scratch: []f32,
    config: CoordinatorConfig,
    current_step: u64 = 0,

    /// Initialize a federated coordinator for a fixed-size model.
    /// @param allocator Memory allocator for allocations
    /// @param config Coordinator configuration
    /// @param model_size Number of weights in the model
    /// @return Initialized Coordinator
    pub fn init(
        allocator: std.mem.Allocator,
        config: CoordinatorConfig,
        model_size: usize,
    ) !Coordinator {
        const global_weights = try allocator.alloc(f32, model_size);
        const scratch = try allocator.alloc(f32, model_size);
        @memset(global_weights, 0);
        @memset(scratch, 0);
        return .{
            .allocator = allocator,
            .registry = Registry.init(allocator),
            .updates = std.ArrayListUnmanaged(ModelUpdate).empty,
            .global_weights = global_weights,
            .scratch = scratch,
            .config = config,
            .current_step = 0,
        };
    }

    /// Release coordinator resources.
    pub fn deinit(self: *Coordinator) void {
        for (self.updates.items) |*update| {
            update.deinit(self.allocator);
        }
        self.updates.deinit(self.allocator);
        self.registry.deinit();
        self.allocator.free(self.global_weights);
        self.allocator.free(self.scratch);
        self.* = undefined;
    }

    /// Register or refresh a node in the registry.
    /// @param node_id Node identifier to register
    pub fn registerNode(self: *Coordinator, node_id: []const u8) !void {
        try self.registry.touch(node_id);
    }

    /// Submit a model update from a node.
    /// @param update Model update view
    /// @return Error if the update is invalid
    pub fn submitUpdate(self: *Coordinator, update: ModelUpdateView) !void {
        if (update.weights.len != self.global_weights.len) {
            return CoordinatorError.InvalidUpdate;
        }
        const node_copy = try self.allocator.dupe(u8, update.node_id);
        errdefer self.allocator.free(node_copy);
        const weight_copy = try self.allocator.alloc(f32, update.weights.len);
        errdefer self.allocator.free(weight_copy);
        std.mem.copyForwards(f32, weight_copy, update.weights);

        const owned = ModelUpdate{
            .node_id = node_copy,
            .step = update.step,
            .timestamp = unixTimestamp(),
            .weights = weight_copy,
            .sample_count = update.sample_count,
        };
        try self.updates.append(self.allocator, owned);
        try self.registry.touch(update.node_id);
        self.pruneUpdates();
    }

    /// Aggregate updates into the global model weights.
    /// @return Updated global weights slice
    pub fn aggregate(self: *Coordinator) CoordinatorError![]const f32 {
        if (self.updates.items.len == 0) return CoordinatorError.InsufficientUpdates;
        const now = unixTimestamp();
        @memset(self.scratch, 0);

        var max_step: u64 = self.current_step;
        var total_weight: f32 = 0.0;
        var valid_updates: usize = 0;

        var i: usize = 0;
        while (i < self.updates.items.len) {
            const update = self.updates.items[i];
            if (self.config.max_staleness_seconds > 0 and
                now > update.timestamp and
                now - update.timestamp > self.config.max_staleness_seconds)
            {
                const removed = self.updates.swapRemove(i);
                removed.deinit(self.allocator);
                continue;
            }

            const weight = switch (self.config.strategy) {
                .mean => 1.0,
                .weighted_mean => @as(f32, @floatFromInt(update.sample_count)),
            };
            if (weight > 0) {
                for (self.scratch, update.weights) |*acc, value| {
                    acc.* += value * weight;
                }
                total_weight += weight;
                valid_updates += 1;
                if (update.step > max_step) max_step = update.step;
            }
            i += 1;
        }

        if (valid_updates < self.config.min_updates or total_weight == 0) {
            return CoordinatorError.InsufficientUpdates;
        }

        for (self.global_weights, 0..) |*value, index| {
            value.* = self.scratch[index] / total_weight;
        }
        self.current_step = max_step;
        self.clearUpdates();
        return self.global_weights;
    }

    /// Access the latest global weights.
    pub fn globalWeights(self: *const Coordinator) []const f32 {
        return self.global_weights;
    }

    /// Access the latest global step.
    pub fn step(self: *const Coordinator) u64 {
        return self.current_step;
    }

    fn pruneUpdates(self: *Coordinator) void {
        if (self.config.max_updates == 0) return;
        while (self.updates.items.len > self.config.max_updates) {
            const removed = self.updates.orderedRemove(0);
            removed.deinit(self.allocator);
        }
    }

    fn clearUpdates(self: *Coordinator) void {
        for (self.updates.items) |*update| {
            update.deinit(self.allocator);
        }
        self.updates.clearRetainingCapacity();
    }
};

fn unixTimestamp() u64 {
    const ts = time.unixSeconds();
    if (ts <= 0) return 0;
    return @intCast(ts);
}

test "federated registry prune and remove" {
    var registry = Registry.init(std.testing.allocator);
    defer registry.deinit();

    try registry.touch("node-a");
    try registry.touch("node-b");
    try std.testing.expectEqual(@as(usize, 2), registry.count());

    registry.nodes.items[0].last_update -= 120;
    const removed = registry.prune(60);
    try std.testing.expectEqual(@as(usize, 1), removed);
    try std.testing.expectEqual(@as(usize, 1), registry.count());

    try std.testing.expect(registry.remove("node-b"));
    try std.testing.expectEqual(@as(usize, 0), registry.count());
}

test "federated coordinator aggregates mean updates" {
    var coordinator = try Coordinator.init(std.testing.allocator, .{}, 2);
    defer coordinator.deinit();

    try coordinator.registerNode("node-a");
    try coordinator.registerNode("node-b");

    try coordinator.submitUpdate(.{
        .node_id = "node-a",
        .step = 1,
        .weights = &.{ 1.0, 2.0 },
        .sample_count = 1,
    });
    try coordinator.submitUpdate(.{
        .node_id = "node-b",
        .step = 1,
        .weights = &.{ 3.0, 4.0 },
        .sample_count = 1,
    });

    const aggregated = try coordinator.aggregate();
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 3.0 }, aggregated);
}

test "federated coordinator rejects stale updates" {
    var coordinator = try Coordinator.init(std.testing.allocator, .{
        .min_updates = 1,
        .max_staleness_seconds = 1,
    }, 2);
    defer coordinator.deinit();

    try coordinator.registerNode("node-a");
    try coordinator.submitUpdate(.{
        .node_id = "node-a",
        .step = 1,
        .weights = &.{ 1.0, 2.0 },
        .sample_count = 1,
    });

    coordinator.updates.items[0].timestamp = 0;
    try std.testing.expectError(CoordinatorError.InsufficientUpdates, coordinator.aggregate());
}
