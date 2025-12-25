//! Network distributed compute module
//!
//! Provides distributed task execution and result serialization.
//! Feature-gated: only compiled when enable_network is true.

const std = @import("std");

const build_options = @import("build_options");
const workload = @import("../runtime/workload.zig");

const DEFAULT_CPU_AFFINITY: u32 = 2;

pub const NetworkConfig = struct {
    listen_address: []const u8 = "0.0.0.0",
    listen_port: u16 = 8080,
    discovery_enabled: bool = true,
    discovery_multicast_address: []const u8 = "239.255.0.1",
    discovery_port: u16 = 9000,
    max_connections: u32 = 32,
    serialization_format: SerializationFormat = .binary,
};

pub const SerializationFormat = enum {
    binary,
    json,
};

pub const NetworkEngine = struct {
    config: NetworkConfig,
    allocator: std.mem.Allocator,
    nodes: NodeRegistry,
    listener: ?*anyopaque = null,
    running: std.atomic.Value(bool),

    pub fn init(allocator: std.mem.Allocator, cfg: NetworkConfig) !NetworkEngine {
        const nodes = try NodeRegistry.init(allocator, cfg.max_connections);

        return NetworkEngine{
            .config = cfg,
            .allocator = allocator,
            .nodes = nodes,
            .running = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *NetworkEngine) void {
        self.nodes.deinit(self.allocator);
    }

    pub fn start(self: *NetworkEngine) !void {
        self.running.store(true, .release);
    }

    pub fn stop(self: *NetworkEngine) void {
        self.running.store(false, .release);
    }

    pub fn submitRemote(self: *NetworkEngine, task_id: u64, item: *const workload.WorkItem) !void {
        _ = self;
        _ = task_id;
        _ = item;
    }

    pub fn pollRemoteResult(self: *NetworkEngine, task_id: u64) ?workload.ResultHandle {
        _ = self;
        _ = task_id;
        return null;
    }
};

pub const NodeRegistry = struct {
    nodes: std.StringHashMap(NodeInfo),
    max_size: usize,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) !NodeRegistry {
        return NodeRegistry{
            .nodes = std.StringHashMap(NodeInfo).init(allocator),
            .max_size = max_size,
        };
    }

    pub fn deinit(self: *NodeRegistry, allocator: std.mem.Allocator) void {
        var iter = self.nodes.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        self.nodes.deinit();
    }

    pub fn addNode(self: *NodeRegistry, node: NodeInfo) !void {
        const key = try self.nodes.allocator.dupe(u8, node.address);
        try self.nodes.put(key, node);
    }

    pub fn removeNode(self: *NodeRegistry, address: []const u8) void {
        if (self.nodes.fetchRemove(address)) |entry| {
            self.nodes.allocator.free(entry.key);
        }
    }

    pub fn getBestNode(self: *NodeRegistry, estimated_workload: u64) ?*NodeInfo {
        _ = estimated_workload;

        var iter = self.nodes.iterator();
        if (iter.next()) |entry| {
            return entry.value_ptr;
        }

        return null;
    }
};

pub const NodeInfo = struct {
    address: []const u8,
    port: u16,
    cpu_count: u32,
    total_memory_bytes: u64,
    current_task_count: u32,
    last_seen_timestamp_ns: u64,
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

const TaskSerializationHeader = struct {
    task_id: u64,
    payload_type_len: u32,
    payload_data_len: u32,
    cpu_affinity: u32,
    estimated_duration_us: u64,
    prefers_gpu: u8,
    requires_gpu: u8,
};

pub fn serializeTask(allocator: std.mem.Allocator, item: *const workload.WorkItem, payload_type: []const u8, user_data: []const u8) ![]u8 {
    const header = TaskSerializationHeader{
        .task_id = item.id,
        .payload_type_len = @intCast(payload_type.len),
        .payload_data_len = @intCast(user_data.len),
        .cpu_affinity = if (item.hints.cpu_affinity) |aff| aff else std.math.maxInt(u32),
        .estimated_duration_us = if (item.hints.estimated_duration_us) |dur| dur else std.math.maxInt(u64),
        .prefers_gpu = if (item.hints.prefers_gpu) 1 else 0,
        .requires_gpu = if (item.hints.requires_gpu) 1 else 0,
    };

    const total_size = @sizeOf(TaskSerializationHeader) + payload_type.len + user_data.len;
    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);

    var offset: usize = 0;

    const header_bytes = std.mem.asBytes(&header);
    @memcpy(buffer[offset..][0..@sizeOf(TaskSerializationHeader)], header_bytes);
    offset += @sizeOf(TaskSerializationHeader);

    @memcpy(buffer[offset..][0..payload_type.len], payload_type);
    offset += payload_type.len;

    @memcpy(buffer[offset..][0..user_data.len], user_data);

    return buffer;
}

pub fn deserializeTask(allocator: std.mem.Allocator, data: []const u8) !struct { item: workload.WorkItem, payload_type: []const u8, user_data: []const u8 } {
    if (data.len < @sizeOf(TaskSerializationHeader)) {
        return error.InvalidData;
    }

    const header_bytes = data[0..@sizeOf(TaskSerializationHeader)];
    const header = std.mem.bytesToValue(TaskSerializationHeader, header_bytes);

    var offset: usize = @sizeOf(TaskSerializationHeader);

    if (offset + header.payload_type_len > data.len) {
        return error.InvalidData;
    }
    const payload_type = try allocator.dupe(u8, data[offset..][0..header.payload_type_len]);
    errdefer allocator.free(payload_type);
    offset += header.payload_type_len;

    if (offset + header.payload_data_len > data.len) {
        return error.InvalidData;
    }
    const user_data = try allocator.dupe(u8, data[offset..][0..header.payload_data_len]);
    errdefer allocator.free(user_data);

    const hints = workload.WorkloadHints{
        .cpu_affinity = if (header.cpu_affinity == std.math.maxInt(u32)) DEFAULT_CPU_AFFINITY else header.cpu_affinity,
        .estimated_duration_us = if (header.estimated_duration_us == std.math.maxInt(u64)) null else header.estimated_duration_us,
        .prefers_gpu = header.prefers_gpu == 1,
        .requires_gpu = header.requires_gpu == 1,
    };

    const item = workload.WorkItem{
        .id = header.task_id,
        .user = undefined,
        .vtable = undefined,
        .priority = 0,
        .hints = hints,
        .gpu_vtable = null,
    };

    return .{
        .item = item,
        .payload_type = payload_type,
        .user_data = user_data,
    };
}

const ResultSerializationHeader = struct {
    task_id: u64,
    success: u8,
    payload_data_len: u32,
    error_message_len: u32,
};

pub fn serializeResult(allocator: std.mem.Allocator, task_id: u64, handle: workload.ResultHandle, success: bool, error_message: ?[]const u8, payload_data: []const u8) ![]u8 {
    _ = handle;

    const header = ResultSerializationHeader{
        .task_id = task_id,
        .success = if (success) 1 else 0,
        .payload_data_len = @intCast(payload_data.len),
        .error_message_len = if (error_message) |msg| @intCast(msg.len) else 0,
    };

    const error_msg_data = error_message orelse "";
    const total_size = @sizeOf(ResultSerializationHeader) + error_msg_data.len + payload_data.len;

    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);

    var offset: usize = 0;

    const header_bytes = std.mem.asBytes(&header);
    @memcpy(buffer[offset..][0..@sizeOf(ResultSerializationHeader)], header_bytes);
    offset += @sizeOf(ResultSerializationHeader);

    @memcpy(buffer[offset..][0..error_msg_data.len], error_msg_data);
    offset += error_msg_data.len;

    @memcpy(buffer[offset..][0..payload_data.len], payload_data);

    return buffer;
}

pub fn deserializeResult(allocator: std.mem.Allocator, data: []const u8) !struct { task_id: u64, success: bool, error_message: ?[]const u8, payload_data: []const u8 } {
    if (data.len < @sizeOf(ResultSerializationHeader)) {
        return error.InvalidData;
    }

    const header_bytes = data[0..@sizeOf(ResultSerializationHeader)];
    const header = std.mem.bytesToValue(ResultSerializationHeader, header_bytes);

    var offset: usize = @sizeOf(ResultSerializationHeader);

    var error_message: ?[]const u8 = null;
    if (header.error_message_len > 0) {
        if (offset + header.error_message_len > data.len) {
            return error.InvalidData;
        }
        error_message = try allocator.dupe(u8, data[offset..][0..header.error_message_len]);
        errdefer allocator.free(error_message.?);
        offset += header.error_message_len;
    }

    if (offset + header.payload_data_len > data.len) {
        return error.InvalidData;
    }
    const payload_data = try allocator.dupe(u8, data[offset..][0..header.payload_data_len]);

    return .{
        .task_id = header.task_id,
        .success = header.success == 1,
        .error_message = error_message,
        .payload_data = payload_data,
    };
}

pub const DEFAULT_NETWORK_CONFIG = NetworkConfig{};
