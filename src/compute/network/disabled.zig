//! Disabled network stubs
//!
//! Provides compile-time compatible placeholders when network features are
//! disabled. All operations return `error.NetworkDisabled` or act as no-ops.

const std = @import("std");
const workload = @import("../runtime/workload.zig");

pub const SerializationFormat = enum {
    binary,
    json,
};

pub const NetworkConfig = struct {
    listen_address: []const u8 = "0.0.0.0",
    listen_port: u16 = 8080,
    discovery_enabled: bool = true,
    discovery_multicast_address: []const u8 = "239.255.0.1",
    discovery_port: u16 = 9000,
    max_connections: u32 = 32,
    serialization_format: SerializationFormat = .binary,
};

pub const NetworkEngine = struct {
    config: NetworkConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cfg: NetworkConfig) !NetworkEngine {
        return NetworkEngine{ .config = cfg, .allocator = allocator };
    }

    pub fn deinit(self: *NetworkEngine) void {
        _ = self;
    }

    pub fn start(self: *NetworkEngine) !void {
        _ = self;
        return error.NetworkDisabled;
    }

    pub fn stop(self: *NetworkEngine) void {
        _ = self;
    }

    pub fn submitRemote(self: *NetworkEngine, task_id: u64, item: *const workload.WorkItem) !void {
        _ = self;
        _ = task_id;
        _ = item;
        return error.NetworkDisabled;
    }

    pub fn pollRemoteResult(self: *NetworkEngine, task_id: u64) ?workload.ResultHandle {
        _ = self;
        _ = task_id;
        return null;
    }
};

pub const NodeRegistry = struct {
    pub fn init(_: std.mem.Allocator, _: usize) !NodeRegistry {
        return NodeRegistry{};
    }

    pub fn deinit(self: *NodeRegistry, _: std.mem.Allocator) void {
        _ = self;
    }

    pub fn addNode(self: *NodeRegistry, _: NodeInfo) !void {
        _ = self;
        return error.NetworkDisabled;
    }

    pub fn removeNode(self: *NodeRegistry, _: []const u8) void {
        _ = self;
    }

    pub fn getBestNode(self: *NodeRegistry, _: u64) ?*NodeInfo {
        _ = self;
        return null;
    }
};

pub const NodeInfo = struct {
    address: []const u8 = "",
    port: u16 = 0,
    cpu_count: u32 = 0,
    total_memory_bytes: u64 = 0,
    current_task_count: u32 = 0,
    last_seen_timestamp_ns: u64 = 0,
};

pub const TaskMessage = struct {
    task_id: u64,
    payload_type: []const u8,
    payload_data: []const u8,
    hints: workload.WorkloadHints,
};

pub const ResultMessage = struct {
    task_id: u64,
    success: bool,
    payload_data: []const u8,
    error_message: ?[]const u8 = null,
};

pub fn serializeTask(
    allocator: std.mem.Allocator,
    item: *const workload.WorkItem,
    payload_type: []const u8,
    user_data: []const u8,
) ![]u8 {
    _ = allocator;
    _ = item;
    _ = payload_type;
    _ = user_data;
    return error.NetworkDisabled;
}

pub fn deserializeTask(
    allocator: std.mem.Allocator,
    data: []const u8,
) !struct { item: workload.WorkItem, payload_type: []const u8, user_data: []const u8 } {
    _ = allocator;
    _ = data;
    return error.NetworkDisabled;
}

pub fn serializeResult(
    allocator: std.mem.Allocator,
    task_id: u64,
    handle: workload.ResultHandle,
    success: bool,
    error_message: ?[]const u8,
    payload_data: []const u8,
) ![]u8 {
    _ = allocator;
    _ = task_id;
    _ = handle;
    _ = success;
    _ = error_message;
    _ = payload_data;
    return error.NetworkDisabled;
}

pub fn deserializeResult(
    allocator: std.mem.Allocator,
    data: []const u8,
) !struct { task_id: u64, success: bool, error_message: ?[]const u8, payload_data: []const u8 } {
    _ = allocator;
    _ = data;
    return error.NetworkDisabled;
}

pub const DEFAULT_NETWORK_CONFIG = NetworkConfig{};
