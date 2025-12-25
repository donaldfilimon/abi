const std = @import("std");
const build_options = @import("build_options");

const registry = @import("registry.zig");
const protocol = @import("protocol.zig");

pub const NodeRegistry = registry.NodeRegistry;
pub const NodeInfo = registry.NodeInfo;
pub const NodeStatus = registry.NodeStatus;

pub const TaskEnvelope = protocol.TaskEnvelope;
pub const ResultEnvelope = protocol.ResultEnvelope;
pub const ResultStatus = protocol.ResultStatus;
pub const encodeTask = protocol.encodeTask;
pub const decodeTask = protocol.decodeTask;
pub const encodeResult = protocol.encodeResult;
pub const decodeResult = protocol.decodeResult;

pub const NetworkError = error{
    NetworkDisabled,
    NotInitialized,
};

pub const NetworkConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_timeout_ms: u64 = 30_000,
    max_nodes: usize = 256,
};

pub const NetworkState = struct {
    allocator: std.mem.Allocator,
    config: NetworkConfig,
    registry: NodeRegistry,

    pub fn init(allocator: std.mem.Allocator, config: NetworkConfig) !NetworkState {
        const cluster_id = try allocator.dupe(u8, config.cluster_id);
        return .{
            .allocator = allocator,
            .config = .{
                .cluster_id = cluster_id,
                .heartbeat_timeout_ms = config.heartbeat_timeout_ms,
                .max_nodes = config.max_nodes,
            },
            .registry = NodeRegistry.init(allocator),
        };
    }

    pub fn deinit(self: *NetworkState) void {
        self.registry.deinit();
        self.allocator.free(self.config.cluster_id);
        self.* = undefined;
    }
};

var default_state: ?NetworkState = null;
var initialized: bool = false;

pub fn isEnabled() bool {
    return build_options.enable_network;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn init(allocator: std.mem.Allocator) !void {
    return initWithConfig(allocator, .{});
}

pub fn initWithConfig(allocator: std.mem.Allocator, config: NetworkConfig) !void {
    if (!isEnabled()) return NetworkError.NetworkDisabled;
    if (default_state == null) {
        default_state = try NetworkState.init(allocator, config);
    }
    initialized = true;
}

pub fn deinit() void {
    if (default_state) |*state| {
        state.deinit();
        default_state = null;
    }
    initialized = false;
}

pub fn defaultRegistry() NetworkError!*NodeRegistry {
    if (default_state) |*state| {
        return &state.registry;
    }
    return NetworkError.NotInitialized;
}

pub fn defaultConfig() ?NetworkConfig {
    return if (default_state) |state| state.config else null;
}

test "network state tracks nodes" {
    var state = try NetworkState.init(std.testing.allocator, .{ .cluster_id = "test" });
    defer state.deinit();

    try state.registry.register("node-a", "127.0.0.1:9000");
    try state.registry.register("node-b", "127.0.0.1:9001");
    try std.testing.expectEqual(@as(usize, 2), state.registry.list().len);
}

test "network default state" {
    if (!isEnabled()) return;

    try initWithConfig(std.testing.allocator, .{ .cluster_id = "cluster-a" });
    defer deinit();

    const registry_ptr = try defaultRegistry();
    try registry_ptr.register("node-a", "127.0.0.1:9000");

    const config = defaultConfig() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("cluster-a", config.cluster_id);
}
