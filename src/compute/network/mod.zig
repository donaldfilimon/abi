//! Network distributed compute module
//!
//! Provides distributed task execution and result serialization.
//! Feature-gated: only compiled when enable_network is true.

const std = @import("std");
const time = @import("../../shared/utils/time.zig");

const build_options = @import("build_options");
const workload = @import("../runtime/workload.zig");

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
    allocator: std.mem.Allocator,
    config: NetworkConfig,
    nodes: NodeRegistry,
    listener: ?*anyopaque = null,
    running: std.atomic.Value(bool),
    pending_tasks: std.AutoHashMapUnmanaged(u64, *PendingTask) = .{},

    pub const PendingTask = struct {
        task_id: u64,
        node_address: []const u8,
        submitted_at: i64,
        completed_at: ?i64 = null,
        status: TaskStatus,
        result: ?workload.ResultHandle = null,
        error_message: ?[]const u8 = null,
    };

    pub fn init(allocator: std.mem.Allocator, cfg: NetworkConfig) !NetworkEngine {
        const nodes = try NodeRegistry.init(allocator, cfg.max_connections);

        return NetworkEngine{
            .allocator = allocator,
            .config = cfg,
            .nodes = nodes,
            .running = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *NetworkEngine) void {
        // Clean up pending tasks
        var iter = self.pending_tasks.iterator();
        while (iter.next()) |entry| {
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.pending_tasks.deinit(self.allocator);

        self.nodes.deinit(self.allocator);
    }

    pub fn start(self: *NetworkEngine) !void {
        self.running.store(true, .release);
    }

    pub fn stop(self: *NetworkEngine) void {
        self.running.store(false, .release);
    }

    /// Submit a task for remote execution on the best available node
    pub fn submitRemote(self: *NetworkEngine, task_id: u64, item: *const workload.WorkItem) !void {
        if (!self.running.load(.acquire)) {
            return error.NetworkNotRunning;
        }

        // Find the best node for this workload
        const best_node = self.nodes.getBestNode(item.hints.estimated_duration_us orelse 1000);
        if (best_node == null) {
            return error.NoAvailableNodes;
        }

        // Serialize the task
        const serialized = try serializeTask(
            self.allocator,
            item,
            "workload",
            &.{}, // No additional user data
        );
        defer self.allocator.free(serialized);

        // Track the pending task
        const pending = try self.allocator.create(PendingTask);
        pending.* = .{
            .task_id = task_id,
            .node_address = best_node.?.address,
            .submitted_at = time.unixSeconds(),
            .status = .pending,
        };
        try self.pending_tasks.put(self.allocator, task_id, pending);

        // In a full implementation, this would:
        // 1. Open a TCP/UDP connection to the node
        // 2. Send the serialized task
        // 3. Wait for acknowledgment

        std.log.info("Network: Submitted task {d} to node {s}:{d}", .{
            task_id,
            best_node.?.address,
            best_node.?.port,
        });

        // Update node task count
        best_node.?.current_task_count += 1;
    }

    /// Poll for remote task completion
    pub fn pollRemoteResult(self: *NetworkEngine, task_id: u64) ?workload.ResultHandle {
        if (!self.running.load(.acquire)) {
            return null;
        }

        // Check if we have a pending task with this ID
        const pending = self.pending_tasks.get(task_id) orelse return null;

        // In a full implementation, this would:
        // 1. Check if we've received a result from the remote node
        // 2. Deserialize the result
        // 3. Return the result handle

        switch (pending.status) {
            .pending, .running => {
                // Task still in progress
                return null;
            },
            .completed => {
                // Task completed - return result
                const result = pending.result orelse return null;

                // Clean up pending task
                if (self.pending_tasks.fetchRemove(task_id)) |entry| {
                    self.allocator.destroy(entry.value);
                }

                return result;
            },
            .failed => {
                // Task failed - return error indicator
                if (self.pending_tasks.fetchRemove(task_id)) |entry| {
                    self.allocator.destroy(entry.value);
                }
                return null;
            },
        }
    }

    /// Mark a task as completed with a result (called when result is received)
    pub fn completeRemoteTask(self: *NetworkEngine, task_id: u64, result: workload.ResultHandle) void {
        if (self.pending_tasks.get(task_id)) |pending| {
            pending.status = .completed;
            pending.result = result;
            pending.completed_at = time.unixSeconds();
        }
    }

    /// Mark a task as failed (called when error is received or timeout occurs)
    pub fn failRemoteTask(self: *NetworkEngine, task_id: u64, error_message: ?[]const u8) void {
        if (self.pending_tasks.get(task_id)) |pending| {
            pending.status = .failed;
            pending.error_message = error_message;
            pending.completed_at = time.unixSeconds();
        }
    }

    /// Get the number of pending remote tasks
    pub fn getPendingTaskCount(self: *NetworkEngine) usize {
        return self.pending_tasks.count();
    }

    /// Check for timed-out tasks and mark them as failed
    pub fn checkTimeouts(self: *NetworkEngine, timeout_seconds: i64) void {
        const now = time.unixSeconds();

        var iter = self.pending_tasks.iterator();
        while (iter.next()) |entry| {
            const pending = entry.value_ptr.*;
            if (pending.status == .pending or pending.status == .running) {
                if (now - pending.submitted_at > timeout_seconds) {
                    pending.status = .failed;
                    pending.error_message = "Task timed out";
                    pending.completed_at = now;
                }
            }
        }
    }

    pub const TaskStatus = enum {
        pending,
        running,
        completed,
        failed,
    };

    pub const NetworkError = error{
        NetworkNotRunning,
        NoAvailableNodes,
        SerializationFailed,
        ConnectionFailed,
        Timeout,
    };
};

pub const NodeRegistry = struct {
    allocator: std.mem.Allocator,
    nodes: std.StringHashMapUnmanaged(NodeInfo),
    max_size: usize,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) !NodeRegistry {
        return NodeRegistry{
            .allocator = allocator,
            .nodes = .{},
            .max_size = max_size,
        };
    }

    pub fn deinit(self: *NodeRegistry, allocator: std.mem.Allocator) void {
        var iter = self.nodes.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        self.nodes.deinit(allocator);
    }

    pub fn addNode(self: *NodeRegistry, node: NodeInfo) !void {
        const key = try self.allocator.dupe(u8, node.address);
        try self.nodes.put(self.allocator, key, node);
    }

    pub fn removeNode(self: *NodeRegistry, address: []const u8) void {
        if (self.nodes.fetchRemove(address)) |entry| {
            self.allocator.free(entry.key);
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

pub const DeserializedTask = struct {
    item: workload.WorkItem,
    payload_type: []const u8,
    user_data: []const u8,
};

pub const DeserializedResult = struct {
    task_id: u64,
    success: bool,
    error_message: ?[]const u8,
    payload_data: []const u8,
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

pub fn serializeTask(
    allocator: std.mem.Allocator,
    item: *const workload.WorkItem,
    payload_type: []const u8,
    user_data: []const u8,
) ![]u8 {
    const header = TaskSerializationHeader{
        .task_id = item.id,
        .payload_type_len = @intCast(payload_type.len),
        .payload_data_len = @intCast(user_data.len),
        .cpu_affinity = if (item.hints.cpu_affinity) |aff|
            aff
        else
            std.math.maxInt(u32),
        .estimated_duration_us = if (item.hints.estimated_duration_us) |dur|
            dur
        else
            std.math.maxInt(u64),
        .prefers_gpu = if (item.hints.prefers_gpu) 1 else 0,
        .requires_gpu = if (item.hints.requires_gpu) 1 else 0,
    };

    const total_size = @sizeOf(TaskSerializationHeader) +
        payload_type.len +
        user_data.len;
    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);

    var offset: usize = 0;

    const header_bytes = std.mem.asBytes(&header);
    @memcpy(
        buffer[offset..][0..@sizeOf(TaskSerializationHeader)],
        header_bytes,
    );
    offset += @sizeOf(TaskSerializationHeader);

    @memcpy(buffer[offset..][0..payload_type.len], payload_type);
    offset += payload_type.len;

    @memcpy(buffer[offset..][0..user_data.len], user_data);

    return buffer;
}

pub fn deserializeTask(
    allocator: std.mem.Allocator,
    data: []const u8,
) !DeserializedTask {
    if (data.len < @sizeOf(TaskSerializationHeader)) {
        std.log.err("Invalid task data: expected at least {d} bytes, got {d}", .{ @sizeOf(TaskSerializationHeader), data.len });
        return error.InvalidData;
    }

    const header_bytes = data[0..@sizeOf(TaskSerializationHeader)];
    const header = std.mem.bytesToValue(
        TaskSerializationHeader,
        header_bytes,
    );

    var offset: usize = @sizeOf(TaskSerializationHeader);

    if (offset + header.payload_type_len > data.len) {
        std.log.err("Invalid task data: payload type length {d} exceeds available data", .{header.payload_type_len});
        return error.InvalidData;
    }
    const payload_type = try allocator.dupe(
        u8,
        data[offset..][0..header.payload_type_len],
    );
    errdefer allocator.free(payload_type);
    offset += header.payload_type_len;

    if (offset + header.payload_data_len > data.len) {
        std.log.err("Invalid task data: payload data length {d} exceeds available data", .{header.payload_data_len});
        return error.InvalidData;
    }
    const user_data = try allocator.dupe(
        u8,
        data[offset..][0..header.payload_data_len],
    );
    errdefer allocator.free(user_data);

    const hints = workload.WorkloadHints{
        .cpu_affinity = if (header.cpu_affinity == std.math.maxInt(u32))
            null
        else
            header.cpu_affinity,
        .estimated_duration_us = if (header.estimated_duration_us == std.math.maxInt(u64))
            null
        else
            header.estimated_duration_us,
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

pub fn serializeResult(
    allocator: std.mem.Allocator,
    task_id: u64,
    handle: workload.ResultHandle,
    success: bool,
    error_message: ?[]const u8,
    payload_data: []const u8,
) ![]u8 {
    _ = handle;

    const header = ResultSerializationHeader{
        .task_id = task_id,
        .success = if (success) 1 else 0,
        .payload_data_len = @intCast(payload_data.len),
        .error_message_len = if (error_message) |msg| @intCast(msg.len) else 0,
    };

    const error_msg_data = error_message orelse "";
    const total_size = @sizeOf(ResultSerializationHeader) +
        error_msg_data.len +
        payload_data.len;

    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);

    var offset: usize = 0;

    const header_bytes = std.mem.asBytes(&header);
    @memcpy(
        buffer[offset..][0..@sizeOf(ResultSerializationHeader)],
        header_bytes,
    );
    offset += @sizeOf(ResultSerializationHeader);

    @memcpy(buffer[offset..][0..error_msg_data.len], error_msg_data);
    offset += error_msg_data.len;

    @memcpy(buffer[offset..][0..payload_data.len], payload_data);

    return buffer;
}

pub fn deserializeResult(
    allocator: std.mem.Allocator,
    data: []const u8,
) !DeserializedResult {
    if (data.len < @sizeOf(ResultSerializationHeader)) {
        std.log.err("Invalid result data: expected at least {d} bytes, got {d}", .{ @sizeOf(ResultSerializationHeader), data.len });
        return error.InvalidData;
    }

    const header_bytes = data[0..@sizeOf(ResultSerializationHeader)];
    const header = std.mem.bytesToValue(
        ResultSerializationHeader,
        header_bytes,
    );

    var offset: usize = @sizeOf(ResultSerializationHeader);

    var error_message: ?[]const u8 = null;
    if (header.error_message_len > 0) {
        if (offset + header.error_message_len > data.len) {
            std.log.err("Invalid result data: error message length {d} exceeds available data", .{header.error_message_len});
            return error.InvalidData;
        }
        error_message = try allocator.dupe(
            u8,
            data[offset..][0..header.error_message_len],
        );
        errdefer allocator.free(error_message.?);
        offset += header.error_message_len;
    }

    if (offset + header.payload_data_len > data.len) {
        std.log.err("Invalid result data: payload data length {d} exceeds available data", .{header.payload_data_len});
        return error.InvalidData;
    }
    const payload_data = try allocator.dupe(
        u8,
        data[offset..][0..header.payload_data_len],
    );

    return .{
        .task_id = header.task_id,
        .success = header.success == 1,
        .error_message = error_message,
        .payload_data = payload_data,
    };
}

pub const DEFAULT_NETWORK_CONFIG = NetworkConfig{};

test "task serialization preserves null cpu affinity" {
    const allocator = std.testing.allocator;
    var stub: u8 = 0;
    const vtable = workload.WorkloadVTable{ .execute = unsupportedExecute };
    const item = workload.WorkItem{
        .id = 99,
        .user = &stub,
        .vtable = &vtable,
        .priority = 0,
        .hints = .{},
        .gpu_vtable = null,
    };

    const payload_type = "test";
    const payload = "data";
    const encoded = try serializeTask(allocator, &item, payload_type, payload);
    defer allocator.free(encoded);

    const decoded = try deserializeTask(allocator, encoded);
    defer allocator.free(decoded.payload_type);
    defer allocator.free(decoded.user_data);

    try std.testing.expectEqual(@as(?u32, null), decoded.item.hints.cpu_affinity);
    try std.testing.expectEqual(@as(?u64, null), decoded.item.hints.estimated_duration_us);
}

fn unsupportedExecute(_: *workload.ExecutionContext, _: *anyopaque) workload.WorkloadError!workload.ResultHandle {
    return workload.ResultHandle.fromSlice(&.{});
}
