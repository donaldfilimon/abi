//! Remote Pointer Abstraction
//!
//! Provides pointer-like semantics for accessing memory on remote nodes.
//! Handles serialization, network transfer, and caching transparently.

const std = @import("std");
const memory_region = @import("memory_region.zig");
const mod = @import("mod.zig");

const RegionId = memory_region.RegionId;
const UnifiedMemoryManager = mod.UnifiedMemoryManager;
const UnifiedMemoryError = mod.UnifiedMemoryError;

/// Error type for remote memory operations.
pub const RemoteMemoryError = error{
    /// Node is not connected.
    NodeNotConnected,
    /// Region not found on remote node.
    RegionNotFound,
    /// Access denied (insufficient permissions).
    AccessDenied,
    /// Operation timed out.
    Timeout,
    /// Data transfer failed.
    TransferFailed,
    /// Invalid offset (out of bounds).
    InvalidOffset,
    /// Type size mismatch.
    SizeMismatch,
    /// Network error.
    NetworkError,
    /// Coherence violation.
    CoherenceViolation,
    /// Serialization error.
    SerializationError,
    /// Out of memory.
    OutOfMemory,
};

/// Remote pointer to a typed value on another node.
pub fn RemotePtr(comptime T: type) type {
    return struct {
        /// Reference to the unified memory manager.
        manager: *UnifiedMemoryManager,

        /// Node ID where the data resides.
        node_id: u64,

        /// Region ID on the remote node.
        region_id: RegionId,

        /// Offset within the region.
        offset: usize,

        const Self = @This();

        /// Read the value from remote memory.
        pub fn read(self: Self) RemoteMemoryError!T {
            var buffer: [@sizeOf(T)]u8 = undefined;
            self.manager.readRemote(
                self.node_id,
                self.region_id,
                self.offset,
                &buffer,
            ) catch |err| {
                return switch (err) {
                    error.NodeNotConnected => error.NodeNotConnected,
                    error.RegionNotFound => error.RegionNotFound,
                    error.AccessDenied => error.AccessDenied,
                    error.OperationTimeout => error.Timeout,
                    error.TransferFailed => error.TransferFailed,
                    error.CoherenceViolation => error.CoherenceViolation,
                    else => error.NetworkError,
                };
            };
            return std.mem.bytesToValue(T, &buffer);
        }

        /// Write a value to remote memory.
        pub fn write(self: Self, value: T) RemoteMemoryError!void {
            const bytes = std.mem.asBytes(&value);
            self.manager.writeRemote(
                self.node_id,
                self.region_id,
                self.offset,
                bytes,
            ) catch |err| {
                return switch (err) {
                    error.NodeNotConnected => error.NodeNotConnected,
                    error.RegionNotFound => error.RegionNotFound,
                    error.AccessDenied => error.AccessDenied,
                    error.OperationTimeout => error.Timeout,
                    error.TransferFailed => error.TransferFailed,
                    error.CoherenceViolation => error.CoherenceViolation,
                    else => error.NetworkError,
                };
            };
        }

        /// Atomically add a value and return the old value.
        pub fn fetchAdd(self: Self, value: T) RemoteMemoryError!T {
            // This would require atomic RMW support from the network layer
            const old = try self.read();
            try self.write(old + value);
            return old;
        }

        /// Atomically compare and swap.
        pub fn compareAndSwap(self: Self, expected: T, new_value: T) RemoteMemoryError!?T {
            const current = try self.read();
            if (std.mem.eql(u8, std.mem.asBytes(&current), std.mem.asBytes(&expected))) {
                try self.write(new_value);
                return null; // Success
            }
            return current; // Return current value on failure
        }

        /// Get a pointer to a field within the remote struct.
        pub fn field(self: Self, comptime field_name: []const u8) RemotePtr(@TypeOf(@field(@as(T, undefined), field_name))) {
            const field_offset = @offsetOf(T, field_name);
            return .{
                .manager = self.manager,
                .node_id = self.node_id,
                .region_id = self.region_id,
                .offset = self.offset + field_offset,
            };
        }

        /// Offset the pointer by a number of elements.
        pub fn add(self: Self, count: usize) Self {
            return .{
                .manager = self.manager,
                .node_id = self.node_id,
                .region_id = self.region_id,
                .offset = self.offset + count * @sizeOf(T),
            };
        }

        /// Get pointer to element at index (for array-like access).
        pub fn at(self: Self, index: usize) Self {
            return self.add(index);
        }

        /// Cast to a different pointer type.
        pub fn cast(self: Self, comptime U: type) RemotePtr(U) {
            return .{
                .manager = self.manager,
                .node_id = self.node_id,
                .region_id = self.region_id,
                .offset = self.offset,
            };
        }

        /// Get the address as a RemoteAddress.
        pub fn address(self: Self) RemoteAddress {
            return .{
                .node_id = self.node_id,
                .region_id = self.region_id,
                .offset = self.offset,
            };
        }

        /// Check if this pointer is null (invalid).
        pub fn isNull(self: Self) bool {
            return self.node_id == 0 and self.region_id == 0 and self.offset == 0;
        }

        /// Create a null pointer.
        pub fn null_ptr(manager: *UnifiedMemoryManager) Self {
            return .{
                .manager = manager,
                .node_id = 0,
                .region_id = 0,
                .offset = 0,
            };
        }
    };
}

/// Remote slice for accessing arrays of values.
pub fn RemoteSlice(comptime T: type) type {
    return struct {
        /// Base remote pointer.
        ptr: RemotePtr(T),

        /// Number of elements.
        len: usize,

        const Self = @This();

        /// Read the entire slice into a local buffer.
        pub fn readAll(self: Self, allocator: std.mem.Allocator) RemoteMemoryError![]T {
            const buffer = allocator.alloc(T, self.len) catch return error.OutOfMemory;
            errdefer allocator.free(buffer);

            const byte_len = self.len * @sizeOf(T);
            var bytes = std.mem.sliceAsBytes(buffer);

            self.ptr.manager.readRemote(
                self.ptr.node_id,
                self.ptr.region_id,
                self.ptr.offset,
                bytes[0..byte_len],
            ) catch |err| {
                return switch (err) {
                    error.NodeNotConnected => error.NodeNotConnected,
                    error.RegionNotFound => error.RegionNotFound,
                    error.AccessDenied => error.AccessDenied,
                    error.OperationTimeout => error.Timeout,
                    error.TransferFailed => error.TransferFailed,
                    error.CoherenceViolation => error.CoherenceViolation,
                    else => error.NetworkError,
                };
            };

            return buffer;
        }

        /// Write an entire slice to remote memory.
        pub fn writeAll(self: Self, data: []const T) RemoteMemoryError!void {
            if (data.len != self.len) return error.SizeMismatch;

            const bytes = std.mem.sliceAsBytes(data);
            self.ptr.manager.writeRemote(
                self.ptr.node_id,
                self.ptr.region_id,
                self.ptr.offset,
                bytes,
            ) catch |err| {
                return switch (err) {
                    error.NodeNotConnected => error.NodeNotConnected,
                    error.RegionNotFound => error.RegionNotFound,
                    error.AccessDenied => error.AccessDenied,
                    error.OperationTimeout => error.Timeout,
                    error.TransferFailed => error.TransferFailed,
                    error.CoherenceViolation => error.CoherenceViolation,
                    else => error.NetworkError,
                };
            };
        }

        /// Read a single element at index.
        pub fn get(self: Self, index: usize) RemoteMemoryError!T {
            if (index >= self.len) return error.InvalidOffset;
            return self.ptr.at(index).read();
        }

        /// Write a single element at index.
        pub fn set(self: Self, index: usize, value: T) RemoteMemoryError!void {
            if (index >= self.len) return error.InvalidOffset;
            return self.ptr.at(index).write(value);
        }

        /// Get a sub-slice.
        pub fn slice(self: Self, start: usize, end: usize) RemoteMemoryError!Self {
            if (start > end or end > self.len) return error.InvalidOffset;
            return .{
                .ptr = self.ptr.add(start),
                .len = end - start,
            };
        }

        /// Iterator for remote slice.
        pub fn iterator(self: Self) Iterator {
            return .{
                .slice = self,
                .index = 0,
            };
        }

        pub const Iterator = struct {
            slice: Self,
            index: usize,

            pub fn next(it: *Iterator) ?RemotePtr(T) {
                if (it.index >= it.slice.len) return null;
                const ptr = it.slice.ptr.at(it.index);
                it.index += 1;
                return ptr;
            }

            pub fn reset(it: *Iterator) void {
                it.index = 0;
            }
        };
    };
}

/// Remote address for serialization and transfer.
pub const RemoteAddress = struct {
    node_id: u64,
    region_id: RegionId,
    offset: usize,

    /// Serialize to bytes.
    pub fn toBytes(self: RemoteAddress) [24]u8 {
        var result: [24]u8 = undefined;
        std.mem.writeInt(u64, result[0..8], self.node_id, .little);
        std.mem.writeInt(u64, result[8..16], self.region_id, .little);
        std.mem.writeInt(u64, result[16..24], @intCast(self.offset), .little);
        return result;
    }

    /// Deserialize from bytes.
    pub fn fromBytes(bytes: [24]u8) RemoteAddress {
        return .{
            .node_id = std.mem.readInt(u64, bytes[0..8], .little),
            .region_id = std.mem.readInt(u64, bytes[8..16], .little),
            .offset = @intCast(std.mem.readInt(u64, bytes[16..24], .little)),
        };
    }

    /// Create a typed remote pointer from this address.
    pub fn toPtr(self: RemoteAddress, comptime T: type, manager: *UnifiedMemoryManager) RemotePtr(T) {
        return .{
            .manager = manager,
            .node_id = self.node_id,
            .region_id = self.region_id,
            .offset = self.offset,
        };
    }

    /// Check if this is a null address.
    pub fn isNull(self: RemoteAddress) bool {
        return self.node_id == 0 and self.region_id == 0 and self.offset == 0;
    }

    /// Null address constant.
    pub const null_addr: RemoteAddress = .{ .node_id = 0, .region_id = 0, .offset = 0 };
};

/// Remote memory handle for bulk operations.
pub const RemoteMemoryHandle = struct {
    /// Node ID.
    node_id: u64,

    /// Region ID.
    region_id: RegionId,

    /// Base offset.
    base_offset: usize,

    /// Size of the memory region.
    size: usize,

    /// Permissions.
    permissions: Permissions,

    /// Handle state.
    state: State,

    pub const Permissions = struct {
        read: bool = false,
        write: bool = false,
        execute: bool = false,
    };

    pub const State = enum {
        invalid,
        valid,
        locked,
        migrating,
    };

    /// Create a slice view of this handle.
    pub fn asSlice(self: RemoteMemoryHandle, comptime T: type, manager: *UnifiedMemoryManager) RemoteSlice(T) {
        return .{
            .ptr = .{
                .manager = manager,
                .node_id = self.node_id,
                .region_id = self.region_id,
                .offset = self.base_offset,
            },
            .len = self.size / @sizeOf(T),
        };
    }

    /// Create a pointer to the start of this handle.
    pub fn asPtr(self: RemoteMemoryHandle, comptime T: type, manager: *UnifiedMemoryManager) RemotePtr(T) {
        return .{
            .manager = manager,
            .node_id = self.node_id,
            .region_id = self.region_id,
            .offset = self.base_offset,
        };
    }
};

/// Batched remote memory operations for efficiency.
pub const BatchOperation = struct {
    /// Operation type.
    op_type: OpType,

    /// Target address.
    address: RemoteAddress,

    /// Data buffer (for writes).
    data: ?[]const u8,

    /// Result buffer (for reads).
    result: ?[]u8,

    /// Size of operation.
    size: usize,

    /// Operation status.
    status: Status,

    pub const OpType = enum {
        read,
        write,
        atomic_add,
        atomic_cas,
        prefetch,
        invalidate,
    };

    pub const Status = enum {
        pending,
        in_progress,
        completed,
        failed,
    };
};

/// Batch executor for multiple remote operations.
pub const BatchExecutor = struct {
    operations: std.ArrayListUnmanaged(BatchOperation),
    allocator: std.mem.Allocator,
    manager: *UnifiedMemoryManager,

    pub fn init(allocator: std.mem.Allocator, manager: *UnifiedMemoryManager) BatchExecutor {
        return .{
            .operations = .{},
            .allocator = allocator,
            .manager = manager,
        };
    }

    pub fn deinit(self: *BatchExecutor) void {
        self.operations.deinit(self.allocator);
    }

    /// Add a read operation to the batch.
    pub fn addRead(self: *BatchExecutor, address: RemoteAddress, result: []u8) !void {
        try self.operations.append(self.allocator, .{
            .op_type = .read,
            .address = address,
            .data = null,
            .result = result,
            .size = result.len,
            .status = .pending,
        });
    }

    /// Add a write operation to the batch.
    pub fn addWrite(self: *BatchExecutor, address: RemoteAddress, data: []const u8) !void {
        try self.operations.append(self.allocator, .{
            .op_type = .write,
            .address = address,
            .data = data,
            .result = null,
            .size = data.len,
            .status = .pending,
        });
    }

    /// Execute all operations in the batch.
    pub fn execute(self: *BatchExecutor) !usize {
        var completed: usize = 0;

        // Group operations by node for efficiency
        for (self.operations.items) |*op| {
            op.status = .in_progress;

            switch (op.op_type) {
                .read => {
                    if (op.result) |result| {
                        self.manager.readRemote(
                            op.address.node_id,
                            op.address.region_id,
                            op.address.offset,
                            result,
                        ) catch {
                            op.status = .failed;
                            continue;
                        };
                    }
                },
                .write => {
                    if (op.data) |data| {
                        self.manager.writeRemote(
                            op.address.node_id,
                            op.address.region_id,
                            op.address.offset,
                            data,
                        ) catch {
                            op.status = .failed;
                            continue;
                        };
                    }
                },
                else => {},
            }

            op.status = .completed;
            completed += 1;
        }

        return completed;
    }

    /// Clear all operations.
    pub fn clear(self: *BatchExecutor) void {
        self.operations.clearRetainingCapacity();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "RemoteAddress serialization" {
    const addr = RemoteAddress{
        .node_id = 12345,
        .region_id = 67890,
        .offset = 4096,
    };

    const bytes = addr.toBytes();
    const restored = RemoteAddress.fromBytes(bytes);

    try std.testing.expectEqual(addr.node_id, restored.node_id);
    try std.testing.expectEqual(addr.region_id, restored.region_id);
    try std.testing.expectEqual(addr.offset, restored.offset);
}

test "RemoteAddress null check" {
    try std.testing.expect(RemoteAddress.null_addr.isNull());

    const valid = RemoteAddress{ .node_id = 1, .region_id = 1, .offset = 0 };
    try std.testing.expect(!valid.isNull());
}

test "RemoteMemoryHandle creation" {
    const handle = RemoteMemoryHandle{
        .node_id = 1,
        .region_id = 100,
        .base_offset = 0,
        .size = 4096,
        .permissions = .{ .read = true, .write = true },
        .state = .valid,
    };

    try std.testing.expect(handle.permissions.read);
    try std.testing.expect(handle.permissions.write);
    try std.testing.expectEqual(RemoteMemoryHandle.State.valid, handle.state);
}
