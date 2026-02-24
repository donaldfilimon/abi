//! Host-Staged Peer Transfer Backend
//!
//! Universal fallback implementation that transfers data between GPUs
//! through host memory. Works with any GPU backend.
//!
//! ## Implementation Strategy
//!
//! 1. Allocate pinned host memory for staging
//! 2. Copy from source GPU to host staging buffer
//! 3. Copy from host staging buffer to destination GPU
//! 4. Use thread pool for async semantics
//!
//! This is the slowest transfer method but always works.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const multi_device = @import("../multi_device.zig");

pub const DeviceId = multi_device.DeviceId;

/// Transfer request for async operations.
const TransferRequest = struct {
    src_device: DeviceId,
    dst_device: DeviceId,
    data: []u8,
    completed: std.atomic.Value(bool),
    error_code: ?anyerror,
};

/// Host-staged transfer backend.
///
/// Uses pinned host memory as an intermediate staging area for
/// GPU-to-GPU transfers when direct peer access is unavailable.
pub const HostStagedBackend = struct {
    allocator: std.mem.Allocator,

    // Pinned memory pool for staging buffers
    staging_pool: StagingPool,

    // Thread pool for async transfers
    thread_pool: ?*ThreadPool,

    // Statistics
    transfers_completed: std.atomic.Value(u64),
    bytes_transferred: std.atomic.Value(u64),

    const Self = @This();

    /// Initialize the host-staged backend.
    pub fn init(allocator: std.mem.Allocator) !Self {
        const pool_ptr: ?*ThreadPool = blk: {
            const pool = allocator.create(ThreadPool) catch break :blk null;
            pool.init(allocator, 4) catch {
                allocator.destroy(pool);
                break :blk null;
            };
            break :blk pool;
        };

        return .{
            .allocator = allocator,
            .staging_pool = try StagingPool.init(allocator),
            .thread_pool = pool_ptr,
            .transfers_completed = std.atomic.Value(u64).init(0),
            .bytes_transferred = std.atomic.Value(u64).init(0),
        };
    }

    /// Deinitialize the backend.
    pub fn deinit(self: *Self) void {
        if (self.thread_pool) |pool| {
            pool.deinit();
            self.allocator.destroy(pool);
        }
        self.staging_pool.deinit();
        self.* = undefined;
    }

    /// Synchronous transfer between devices via host memory.
    pub fn transfer(
        self: *Self,
        src_device: DeviceId,
        dst_device: DeviceId,
        data: []u8,
    ) !void {
        if (data.len == 0) return;

        // Get staging buffer
        const staging = try self.staging_pool.acquire(data.len);
        defer self.staging_pool.release(staging);

        // Step 1: Copy from source GPU to host staging
        const staging_slice = staging[0..data.len];
        try self.deviceToHost(src_device, data, staging_slice);

        // Step 2: Copy from host staging to destination GPU
        try self.hostToDevice(dst_device, staging_slice, data);

        // Update statistics
        _ = self.transfers_completed.fetchAdd(1, .monotonic);
        _ = self.bytes_transferred.fetchAdd(data.len, .monotonic);
    }

    /// Async transfer with callback.
    pub fn transferAsync(
        self: *Self,
        src_device: DeviceId,
        dst_device: DeviceId,
        data: []u8,
        callback: ?*const fn (@"error": ?anyerror) void,
    ) !void {
        if (self.thread_pool) |pool| {
            const task = TransferTask{
                .backend = self,
                .src_device = src_device,
                .dst_device = dst_device,
                .data = data,
                .callback = callback,
            };
            try pool.submit(task);
        } else {
            // No thread pool, fall back to sync
            self.transfer(src_device, dst_device, data) catch |err| {
                if (callback) |cb| {
                    cb(err);
                }
                return err;
            };
            if (callback) |cb| {
                cb(null);
            }
        }
    }

    /// Copy from device to host staging buffer.
    fn deviceToHost(self: *Self, device_id: DeviceId, src: []u8, dst: []u8) !void {
        _ = self;
        _ = device_id;

        // In a real implementation, this would call the appropriate backend's
        // device-to-host copy function (e.g., cudaMemcpy, vkCmdCopyBuffer)
        //
        // For simulation/fallback, we just do a memcpy since the "GPU" memory
        // is actually host memory in our simulated environment
        if (src.len > dst.len) {
            return error.BufferTooSmall;
        }

        @memcpy(dst[0..src.len], src);
    }

    /// Copy from host staging buffer to device.
    fn hostToDevice(self: *Self, device_id: DeviceId, src: []u8, dst: []u8) !void {
        _ = self;
        _ = device_id;

        // In a real implementation, this would call the appropriate backend's
        // host-to-device copy function
        if (src.len > dst.len) {
            return error.BufferTooSmall;
        }

        @memcpy(dst[0..src.len], src);
    }

    /// Perform AllReduce using host staging.
    pub fn allReduce(
        self: *Self,
        buffers: []const DeviceBufferRef,
        op: multi_device.ReduceOp,
    ) !void {
        if (buffers.len <= 1) return;

        const data_len = buffers[0].data.len;
        const byte_len = data_len * @sizeOf(f32);

        // Allocate reduction buffer on host
        const reduce_buffer = try self.allocator.alloc(f32, data_len);
        defer self.allocator.free(reduce_buffer);

        // Initialize with first buffer
        const first_staging = try self.staging_pool.acquire(byte_len);
        defer self.staging_pool.release(first_staging);
        const first_staging_slice = first_staging[0..byte_len];

        const first_data = std.mem.sliceAsBytes(buffers[0].data);
        try self.deviceToHost(buffers[0].device_id, @constCast(first_data), first_staging_slice);

        const first_floats = std.mem.bytesAsSlice(f32, first_staging_slice);
        @memcpy(reduce_buffer[0..data_len], first_floats[0..data_len]);

        // Reduce from all other devices
        const staging = try self.staging_pool.acquire(byte_len);
        defer self.staging_pool.release(staging);
        const staging_slice = staging[0..byte_len];

        for (buffers[1..]) |buf| {
            const buf_data = std.mem.sliceAsBytes(buf.data);
            try self.deviceToHost(buf.device_id, @constCast(buf_data), staging_slice);

            const floats = std.mem.bytesAsSlice(f32, staging_slice);
            for (reduce_buffer, 0..) |*val, i| {
                val.* = applyOp(val.*, floats[i], op);
            }
        }

        // Handle average
        if (op == .avg) {
            const scale = 1.0 / @as(f32, @floatFromInt(buffers.len));
            for (reduce_buffer) |*val| {
                val.* *= scale;
            }
        }

        // Broadcast result back to all devices
        const result_bytes = std.mem.sliceAsBytes(reduce_buffer);
        @memcpy(staging_slice, result_bytes);

        for (buffers) |buf| {
            const dst_bytes = std.mem.sliceAsBytes(buf.data);
            try self.hostToDevice(buf.device_id, staging_slice, @constCast(dst_bytes));
        }

        _ = self.transfers_completed.fetchAdd(buffers.len * 2, .monotonic);
        _ = self.bytes_transferred.fetchAdd(buffers.len * data_len * @sizeOf(f32) * 2, .monotonic);
    }

    fn applyOp(a: f32, b: f32, op: multi_device.ReduceOp) f32 {
        return switch (op) {
            .sum => a + b,
            .product => a * b,
            .min => @min(a, b),
            .max => @max(a, b),
            .avg => a + b,
        };
    }

    /// Get transfer statistics.
    pub fn getStats(self: *const Self) struct { transfers: u64, bytes: u64 } {
        return .{
            .transfers = self.transfers_completed.load(.acquire),
            .bytes = self.bytes_transferred.load(.acquire),
        };
    }
};

/// Device buffer reference for AllReduce.
pub const DeviceBufferRef = struct {
    device_id: DeviceId,
    data: []f32,
};

/// Staging buffer pool for efficient memory reuse.
pub const StagingPool = struct {
    allocator: std.mem.Allocator,
    buffers: std.ArrayListUnmanaged(StagingBuffer),
    mutex: sync.Mutex,

    // Pool configuration
    const MIN_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB minimum
    const MAX_CACHED_BUFFERS: usize = 8;

    const StagingBuffer = struct {
        data: []align(64) u8,
        size: usize,
        in_use: bool,
    };

    pub fn init(allocator: std.mem.Allocator) !StagingPool {
        return .{
            .allocator = allocator,
            .buffers = std.ArrayListUnmanaged(StagingBuffer).empty,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *StagingPool) void {
        for (self.buffers.items) |buffer| {
            self.allocator.free(buffer.data);
        }
        self.buffers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Acquire a staging buffer of at least the requested size.
    pub fn acquire(self: *StagingPool, min_size: usize) ![]u8 {
        const size = @max(min_size, MIN_BUFFER_SIZE);

        self.mutex.lock();
        defer self.mutex.unlock();

        // Try to find an existing buffer
        for (self.buffers.items) |*buffer| {
            if (!buffer.in_use and buffer.size >= size) {
                buffer.in_use = true;
                return buffer.data[0..size];
            }
        }

        // Allocate a new buffer if under limit
        if (self.buffers.items.len < MAX_CACHED_BUFFERS) {
            const aligned_size = std.mem.alignForward(usize, size, 64);
            const data = try self.allocator.alignedAlloc(u8, .@"64", aligned_size);
            try self.buffers.append(self.allocator, .{
                .data = data,
                .size = aligned_size,
                .in_use = true,
            });
            return data[0..size];
        }

        // All buffers in use, allocate temporary
        const aligned_size = std.mem.alignForward(usize, size, 64);
        return self.allocator.alignedAlloc(u8, .@"64", aligned_size);
    }

    /// Release a staging buffer back to the pool.
    pub fn release(self: *StagingPool, buffer: []u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.buffers.items) |*staged| {
            if (staged.data.ptr == buffer.ptr) {
                staged.in_use = false;
                return;
            }
        }

        // Not from the pool, free it
        const aligned: []align(64) u8 = @alignCast(buffer);
        self.allocator.free(aligned);
    }
};

/// Task for async transfer execution.
const TransferTask = struct {
    backend: *HostStagedBackend,
    src_device: DeviceId,
    dst_device: DeviceId,
    data: []u8,
    callback: ?*const fn (@"error": ?anyerror) void,
};

/// Simple thread pool for async transfers.
const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    queue: std.ArrayListUnmanaged(TransferTask),
    mutex: sync.Mutex,
    condition: sync.Condition,
    shutdown: std.atomic.Value(bool),

    pub fn init(self: *ThreadPool, allocator: std.mem.Allocator, num_threads: usize) !void {
        self.* = ThreadPool{
            .allocator = allocator,
            .threads = try allocator.alloc(std.Thread, num_threads),
            .queue = std.ArrayListUnmanaged(TransferTask).empty,
            .mutex = .{},
            .condition = .{},
            .shutdown = std.atomic.Value(bool).init(false),
        };

        for (self.threads, 0..) |*thread, i| {
            _ = i;
            thread.* = try std.Thread.spawn(.{}, workerThread, .{self});
        }
    }

    pub fn deinit(self: *ThreadPool) void {
        self.shutdown.store(true, .release);
        self.condition.broadcast();

        for (self.threads) |thread| {
            thread.join();
        }

        self.allocator.free(self.threads);
        self.queue.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn submit(self: *ThreadPool, task: TransferTask) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.queue.append(self.allocator, task);
        self.condition.signal();
    }

    fn workerThread(pool: *ThreadPool) void {
        while (!pool.shutdown.load(.acquire)) {
            pool.mutex.lock();

            while (pool.queue.items.len == 0 and !pool.shutdown.load(.acquire)) {
                pool.condition.wait(&pool.mutex);
            }

            if (pool.shutdown.load(.acquire)) {
                pool.mutex.unlock();
                break;
            }

            const task = pool.queue.orderedRemove(0);
            pool.mutex.unlock();

            // Execute the task
            task.backend.transfer(task.src_device, task.dst_device, task.data) catch |err| {
                if (task.callback) |cb| {
                    cb(err);
                }
                continue;
            };

            if (task.callback) |cb| {
                cb(null);
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "HostStagedBackend initialization" {
    const allocator = std.testing.allocator;
    var backend = try HostStagedBackend.init(allocator);
    defer backend.deinit();

    const stats = backend.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.transfers);
    try std.testing.expectEqual(@as(u64, 0), stats.bytes);
}

test "HostStagedBackend transfer" {
    const allocator = std.testing.allocator;
    var backend = try HostStagedBackend.init(allocator);
    defer backend.deinit();

    var data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try backend.transfer(0, 1, &data);

    const stats = backend.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.transfers);
    try std.testing.expectEqual(@as(u64, 8), stats.bytes);
}

test "StagingPool acquire and release" {
    const allocator = std.testing.allocator;
    var pool = try StagingPool.init(allocator);
    defer pool.deinit();

    const buf1 = try pool.acquire(100);
    const buf2 = try pool.acquire(200);

    try std.testing.expect(buf1.len >= 100);
    try std.testing.expect(buf2.len >= 200);

    pool.release(buf1);
    pool.release(buf2);
}

test "HostStagedBackend allReduce" {
    const allocator = std.testing.allocator;
    var backend = try HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var buf2 = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const buffers = [_]DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
        .{ .device_id = 1, .data = &buf2 },
    };

    try backend.allReduce(&buffers, .sum);

    // After sum reduction, both buffers should have same values
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), buf1[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), buf1[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), buf1[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), buf1[3], 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
