//! CUDA stream management with true async execution.
//!
//! Provides real CUDA stream creation, synchronization, and
//! asynchronous kernel execution.

const std = @import("std");
const loader = @import("loader.zig");

pub const StreamError = error{
    CreationFailed,
    DestroyFailed,
    SynchronizeFailed,
    RecordFailed,
    WaitFailed,
    InitializationFailed,
};

const CUresult = enum(i32) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    invalid_context = 6,
    invalid_handle = 400,
    launch_failed = 700,
    launch_timeout = 702,
    launch_incompatible_texturing = 703,
    peer_access_already_enabled = 704,
    peer_access_not_enabled = 705,
    error_launch_max_exceeded = 716,
    error_launch_file_not_found = 717,
    error_launch_shared_memory_exceeded = 718,
    error_launch_no_execute_permission = 719,
    error_launch_mismatch = 720,
};

const CUstream = ?*anyopaque;
const CUevent = ?*anyopaque;

const CuStreamCreateFn = loader.CuStreamCreateFn;
const CuStreamDestroyFn = loader.CuStreamDestroyFn;
const CuStreamSynchronizeFn = loader.CuStreamSynchronizeFn;
const CuStreamWaitEventFn = loader.CuStreamWaitEventFn;
const CuEventCreateFn = loader.CuEventCreateFn;
const CuEventDestroyFn = loader.CuEventDestroyFn;
const CuEventRecordFn = loader.CuEventRecordFn;
const CuEventSynchronizeFn = loader.CuEventSynchronizeFn;
const CuEventElapsedTimeFn = loader.CuEventElapsedTimeFn;

var cuStreamCreate: ?CuStreamCreateFn = null;
var cuStreamDestroy: ?CuStreamDestroyFn = null;
var cuStreamSynchronize: ?CuStreamSynchronizeFn = null;
var cuStreamWaitEvent: ?CuStreamWaitEventFn = null;
var cuEventCreate: ?CuEventCreateFn = null;
var cuEventDestroy: ?CuEventDestroyFn = null;
var cuEventRecord: ?CuEventRecordFn = null;
var cuEventSynchronize: ?CuEventSynchronizeFn = null;
var cuEventElapsedTime: ?CuEventElapsedTimeFn = null;

var initialized = false;

fn checkResult(result: CUresult) StreamError!void {
    return switch (result) {
        .success => {},
        .invalid_value => StreamError.CreationFailed,
        .out_of_memory => StreamError.CreationFailed,
        .not_initialized => StreamError.InitializationFailed,
        .invalid_context => StreamError.InitializationFailed,
        .invalid_handle => StreamError.DestroyFailed,
        .launch_failed => StreamError.SynchronizeFailed,
        .launch_timeout => StreamError.SynchronizeFailed,
        else => StreamError.CreationFailed,
    };
}

pub fn init() !void {
    if (initialized) return;

    const funcs = loader.getFunctions() orelse loader.load(std.heap.page_allocator) catch return StreamError.InitializationFailed;

    cuStreamCreate = funcs.stream.cuStreamCreate;
    cuStreamDestroy = funcs.stream.cuStreamDestroy;
    cuStreamSynchronize = funcs.stream.cuStreamSynchronize;
    cuStreamWaitEvent = funcs.stream.cuStreamWaitEvent;
    cuEventCreate = funcs.event.cuEventCreate;
    cuEventDestroy = funcs.event.cuEventDestroy;
    cuEventRecord = funcs.event.cuEventRecord;
    cuEventSynchronize = funcs.event.cuEventSynchronize;
    cuEventElapsedTime = funcs.event.cuEventElapsedTime;

    if (cuStreamCreate == null or cuStreamDestroy == null or cuStreamSynchronize == null) {
        return StreamError.InitializationFailed;
    }

    initialized = true;
}

pub fn deinit() void {
    cuStreamCreate = null;
    cuStreamDestroy = null;
    cuStreamSynchronize = null;
    cuStreamWaitEvent = null;
    cuEventCreate = null;
    cuEventDestroy = null;
    cuEventRecord = null;
    cuEventSynchronize = null;
    cuEventElapsedTime = null;
    initialized = false;
}

pub const CudaStream = struct {
    handle: CUstream,
    device_id: i32,

    pub fn create(device_id: i32) StreamError!CudaStream {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const create_fn = cuStreamCreate orelse return StreamError.InitializationFailed;
        var stream: CUstream = null;

        const result = create_fn(&stream, 0);
        try checkResult(result);

        return .{
            .handle = stream,
            .device_id = device_id,
        };
    }

    pub fn destroy(self: *CudaStream) void {
        if (self.handle) |handle| {
            if (cuStreamDestroy) |destroy_fn| {
                _ = destroy_fn(handle);
            }
        }
        self.* = undefined;
    }

    pub fn synchronize(self: *CudaStream) StreamError!void {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const sync_fn = cuStreamSynchronize orelse return StreamError.InitializationFailed;
        const result = sync_fn(self.handle orelse return StreamError.CreationFailed);
        try checkResult(result);
    }

    pub fn recordEvent(self: *CudaStream, event: *CudaEvent) StreamError!void {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const record_fn = cuEventRecord orelse return StreamError.InitializationFailed;
        const result = record_fn(event.handle orelse return StreamError.RecordFailed, self.handle orelse return StreamError.CreationFailed);
        try checkResult(result);
    }

    pub fn waitEvent(self: *CudaStream, event: *CudaEvent) StreamError!void {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const wait_fn = cuStreamWaitEvent orelse return StreamError.InitializationFailed;
        const result = wait_fn(self.handle orelse return StreamError.CreationFailed, event.handle orelse return StreamError.RecordFailed, 0);
        try checkResult(result);
    }
};

pub const CudaEvent = struct {
    handle: CUevent,

    pub fn create() StreamError!CudaEvent {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const create_fn = cuEventCreate orelse return StreamError.InitializationFailed;
        var event: CUevent = null;

        const result = create_fn(&event, 0);
        try checkResult(result);

        return .{
            .handle = event,
        };
    }

    pub fn destroy(self: *CudaEvent) void {
        if (self.handle) |handle| {
            const destroy_fn = cuEventDestroy orelse return;
            _ = destroy_fn(handle);
        }
        self.* = undefined;
    }

    pub fn record(self: *CudaEvent, stream: *CudaStream) StreamError!void {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const record_fn = cuEventRecord orelse return StreamError.InitializationFailed;
        const result = record_fn(self.handle orelse return StreamError.RecordFailed, stream.handle orelse return StreamError.CreationFailed);
        try checkResult(result);
    }

    pub fn synchronize(self: *CudaEvent) StreamError!void {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const sync_fn = cuEventSynchronize orelse return StreamError.InitializationFailed;
        const result = sync_fn(self.handle orelse return StreamError.RecordFailed);
        try checkResult(result);
    }

    pub fn elapsed(self: *CudaEvent, start: *CudaEvent) StreamError!f32 {
        if (!initialized) {
            return StreamError.InitializationFailed;
        }

        const elapsed_fn = cuEventElapsedTime orelse return StreamError.InitializationFailed;
        var ms: f32 = 0;

        const result = elapsed_fn(&ms, self.handle orelse return StreamError.RecordFailed, start.handle orelse return StreamError.RecordFailed);
        try checkResult(result);

        return ms;
    }
};

pub const StreamPool = struct {
    streams: std.ArrayListUnmanaged(CudaStream),
    allocator: std.mem.Allocator,
    next_stream: usize,

    pub fn init(allocator: std.mem.Allocator, count: usize) !StreamPool {
        var pool = StreamPool{
            .streams = std.ArrayListUnmanaged(CudaStream).empty,
            .allocator = allocator,
            .next_stream = 0,
        };

        try pool.streams.ensureTotalCapacity(allocator, count);

        for (0..count) |_| {
            const stream = try CudaStream.create(0);
            try pool.streams.append(allocator, stream);
        }

        return pool;
    }

    pub fn deinit(self: *StreamPool) void {
        for (self.streams.items) |*stream| {
            stream.destroy();
        }
        self.streams.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn acquire(self: *StreamPool) !*CudaStream {
        const idx = self.next_stream;
        self.next_stream = (self.next_stream + 1) % self.streams.items.len;
        return &self.streams.items[idx];
    }

    pub fn synchronizeAll(self: *StreamPool) !void {
        for (self.streams.items) |*stream| {
            try stream.synchronize();
        }
    }
};

/// Lightweight stream pool for multi-stream kernel overlap.
///
/// Unlike `StreamPool` (which wraps real CUDA streams and requires driver
/// initialization), `MultiStreamPool` is a simple index-based pool that
/// tracks which stream slots are in use. This is useful for scheduling
/// concurrent compute and data-transfer kernels across multiple CUDA
/// streams without requiring the full CUDA driver to be loaded.
///
/// ## Usage
/// ```zig
/// var pool = MultiStreamPool.init(4);
/// const s0 = try pool.acquire(); // 0
/// const s1 = try pool.acquire(); // 1
/// pool.release(s0);
/// const s2 = try pool.acquire(); // 0 (reused)
/// ```
pub const MultiStreamPool = struct {
    num_streams: u32,
    in_use: [MAX_STREAMS]bool = [_]bool{false} ** MAX_STREAMS,

    pub const MAX_STREAMS = 16;

    pub fn init(num_streams: u32) MultiStreamPool {
        return .{
            .num_streams = @min(num_streams, MAX_STREAMS),
        };
    }

    /// Acquire the next available stream index.
    /// Returns `error.NoStreamsAvailable` if all streams are in use.
    pub fn acquire(self: *MultiStreamPool) !u32 {
        for (self.in_use[0..self.num_streams], 0..) |*used, i| {
            if (!used.*) {
                used.* = true;
                return @intCast(i);
            }
        }
        return error.NoStreamsAvailable;
    }

    /// Release a previously acquired stream index back to the pool.
    pub fn release(self: *MultiStreamPool, idx: u32) void {
        if (idx < self.num_streams) {
            self.in_use[idx] = false;
        }
    }

    /// Count the number of currently active (in-use) streams.
    pub fn activeCount(self: *const MultiStreamPool) u32 {
        var count: u32 = 0;
        for (self.in_use[0..self.num_streams]) |used| {
            if (used) count += 1;
        }
        return count;
    }

    /// Release all streams back to the pool.
    pub fn releaseAll(self: *MultiStreamPool) void {
        for (self.in_use[0..self.num_streams]) |*used| {
            used.* = false;
        }
    }
};

// ============================================================================
// Tests (MultiStreamPool — does not require CUDA driver)
// ============================================================================

test "MultiStreamPool acquire and release" {
    var pool = MultiStreamPool.init(4);

    const s0 = try pool.acquire();
    try std.testing.expectEqual(@as(u32, 0), s0);
    try std.testing.expectEqual(@as(u32, 1), pool.activeCount());

    const s1 = try pool.acquire();
    try std.testing.expectEqual(@as(u32, 1), s1);
    try std.testing.expectEqual(@as(u32, 2), pool.activeCount());

    pool.release(s0);
    try std.testing.expectEqual(@as(u32, 1), pool.activeCount());

    // Re-acquire should return the released slot
    const s2 = try pool.acquire();
    try std.testing.expectEqual(@as(u32, 0), s2);
}

test "MultiStreamPool exhaustion returns error" {
    var pool = MultiStreamPool.init(2);

    _ = try pool.acquire();
    _ = try pool.acquire();

    try std.testing.expectError(error.NoStreamsAvailable, pool.acquire());
}

test "MultiStreamPool release out-of-range is safe" {
    var pool = MultiStreamPool.init(2);
    // Releasing an index beyond num_streams should be a no-op
    pool.release(99);
    try std.testing.expectEqual(@as(u32, 0), pool.activeCount());
}

test "MultiStreamPool releaseAll" {
    var pool = MultiStreamPool.init(4);

    _ = try pool.acquire();
    _ = try pool.acquire();
    _ = try pool.acquire();
    try std.testing.expectEqual(@as(u32, 3), pool.activeCount());

    pool.releaseAll();
    try std.testing.expectEqual(@as(u32, 0), pool.activeCount());
}

test "MultiStreamPool clamped to MAX_STREAMS" {
    const pool = MultiStreamPool.init(100);
    try std.testing.expectEqual(@as(u32, MultiStreamPool.MAX_STREAMS), pool.num_streams);
}
