//! Unified GPU Buffer
//!
//! Smart buffer system with automatic and explicit memory modes.
//! Tracks dirty state for efficient host-device synchronization.

const std = @import("std");
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");
const stream_mod = @import("stream.zig");
const sync_event_mod = @import("sync_event.zig");

// Zig 0.16 compatibility: Simple spinlock Mutex
const Mutex = struct {
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    pub fn lock(self: *Mutex) void {
        while (self.locked.swap(true, .acquire)) std.atomic.spinLoopHint();
    }
    pub fn unlock(self: *Mutex) void {
        self.locked.store(false, .release);
    }
};

pub const Backend = backend_mod.Backend;
pub const Device = device_mod.Device;
pub const Stream = stream_mod.Stream;
pub const SyncEvent = sync_event_mod.SyncEvent;

/// Memory management mode.
pub const MemoryMode = enum {
    /// API handles all transfers automatically.
    automatic,
    /// User controls transfers explicitly.
    explicit,
    /// Use unified memory where available.
    unified,
};

/// Memory location preference.
pub const MemoryLocation = enum {
    /// Prefer device memory.
    device_preferred,
    /// Prefer host memory.
    host_preferred,
    /// Use host memory only.
    host_only,
    /// Use device memory only.
    device_only,
};

/// Access hint for optimization.
pub const AccessHint = enum {
    /// Buffer is read-only.
    read_only,
    /// Buffer is write-only.
    write_only,
    /// Buffer is read-write.
    read_write,
};

/// Map access mode.
pub const MapAccess = enum {
    read,
    write,
    read_write,
};

/// Buffer element type.
pub const ElementType = enum {
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f16,
    f32,
    f64,

    pub fn size(self: ElementType) usize {
        return switch (self) {
            .u8, .i8 => 1,
            .u16, .i16, .f16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
        };
    }

    pub fn fromType(comptime T: type) ElementType {
        return switch (T) {
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            else => @compileError("Unsupported element type"),
        };
    }
};

/// Buffer creation options.
pub const BufferOptions = struct {
    /// Memory management mode.
    mode: MemoryMode = .automatic,
    /// Access pattern hint.
    access: AccessHint = .read_write,
    /// Memory location preference.
    location: MemoryLocation = .device_preferred,
    /// Use zero-copy memory if available.
    zero_copy: bool = false,
    /// Target device (null = use active device).
    device: ?*const Device = null,
    /// Element type (for typed operations).
    element_type: ElementType = .u8,
    /// Initial data to copy (if any).
    initial_data: ?[]const u8 = null,
};

/// Dirty state tracking.
const DirtyState = packed struct {
    host_dirty: bool = false,
    device_dirty: bool = false,
};

/// Unified GPU buffer with smart memory management.
pub const Buffer = struct {
    // Core state
    allocator: std.mem.Allocator,
    thread_safe_allocator: std.heap.ThreadSafeAllocator,
    size: usize,
    element_type: ElementType,
    options: BufferOptions,

    // Memory storage
    host_data: ?[]u8,
    device_handle: ?*anyopaque,

    // State tracking
    dirty: std.atomic.Value(u8), // Packed DirtyState
    mode: MemoryMode,
    device_id: u32,
    backend: Backend,

    // Thread safety
    mutex: Mutex,

    // Statistics
    host_to_device_transfers: u64,
    device_to_host_transfers: u64,
    bytes_transferred: u64,

    // Synchronization event for non-blocking sync operations
    sync_event: SyncEvent,

    /// Create a new buffer.
    pub fn init(
        allocator: std.mem.Allocator,
        size: usize,
        device: *const Device,
        options: BufferOptions,
    ) !Buffer {
        var thread_safe_allocator = std.heap.ThreadSafeAllocator{ .child_allocator = allocator };

        // Allocate host memory if needed
        const host_data: ?[]u8 = switch (options.location) {
            .device_only => null,
            else => blk: {
                const data = try thread_safe_allocator.allocator().alloc(u8, size);
                @memset(data, 0);
                break :blk data;
            },
        };
        errdefer if (host_data) |data| thread_safe_allocator.allocator().free(data);

        // Copy initial data if provided
        if (options.initial_data) |initial| {
            if (host_data) |data| {
                const copy_size = @min(initial.len, size);
                @memcpy(data[0..copy_size], initial[0..copy_size]);
            }
        }

        var buffer = Buffer{
            .allocator = allocator,
            .thread_safe_allocator = thread_safe_allocator,
            .size = size,
            .element_type = options.element_type,
            .options = options,
            .host_data = host_data,
            .device_handle = null, // Allocated lazily or by backend
            .dirty = std.atomic.Value(u8).init(if (options.initial_data != null) 0b01 else 0), // host_dirty if has data
            .mode = options.mode,
            .device_id = device.id,
            .backend = device.backend,
            .mutex = .{},
            .host_to_device_transfers = 0,
            .device_to_host_transfers = 0,
            .bytes_transferred = 0,
            .sync_event = SyncEvent.init(),
        };

        // In automatic mode with initial data, mark for upload
        if (options.mode == .automatic and options.initial_data != null) {
            buffer.markHostDirty();
        }

        return buffer;
    }

    /// Destroy the buffer.
    pub fn deinit(self: *Buffer) void {
        if (self.host_data) |data| {
            self.thread_safe_allocator.allocator().free(data);
        }
        self.sync_event.deinit();
        // Backend would free device_handle here
        self.* = undefined;
    }

    /// Get buffer size in bytes.
    pub fn getSize(self: *const Buffer) usize {
        return self.size;
    }

    /// Get element count.
    pub fn elementCount(self: *const Buffer) usize {
        return self.size / self.element_type.size();
    }

    /// Write data to the buffer from host.
    pub fn write(self: *Buffer, comptime T: type, data: []const T) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const bytes = std.mem.sliceAsBytes(data);
        if (bytes.len > self.size) {
            return error.BufferOverflow;
        }

        if (self.host_data) |host| {
            @memcpy(host[0..bytes.len], bytes);
        } else {
            return error.NoHostMemory;
        }

        self.markHostDirty();

        // In automatic mode, sync to device immediately
        if (self.mode == .automatic) {
            try self.toDevice();
        }
    }

    /// Write raw bytes to the buffer.
    pub fn writeBytes(self: *Buffer, data: []const u8) !void {
        if (data.len > self.size) {
            return error.BufferOverflow;
        }

        if (self.host_data) |host| {
            @memcpy(host[0..data.len], data);
        } else {
            return error.NoHostMemory;
        }

        self.markHostDirty();

        if (self.mode == .automatic) {
            try self.toDevice();
        }
    }

    /// Read data from the buffer to host.
    pub fn read(self: *Buffer, comptime T: type, output: []T) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const needed_bytes = output.len * @sizeOf(T);
        if (needed_bytes > self.size) {
            return error.BufferOverflow;
        }

        // In automatic mode, sync from device first
        if (self.mode == .automatic and self.isDeviceDirty()) {
            try self.toHost();
        }

        if (self.host_data) |host| {
            const bytes = std.mem.sliceAsBytes(output);
            @memcpy(bytes, host[0..needed_bytes]);
        } else {
            return error.NoHostMemory;
        }
    }

    /// Read raw bytes from the buffer.
    pub fn readBytes(self: *Buffer, output: []u8) !void {
        if (output.len > self.size) {
            return error.BufferOverflow;
        }

        if (self.mode == .automatic and self.isDeviceDirty()) {
            try self.toHost();
        }

        if (self.host_data) |host| {
            @memcpy(output, host[0..output.len]);
        } else {
            return error.NoHostMemory;
        }
    }

    /// Transfer data from host to device (explicit mode).
    /// Signals the sync event upon completion.
    pub fn toDevice(self: *Buffer) !void {
        if (!self.isHostDirty()) {
            self.sync_event.record(); // Already synced
            return;
        }

        // Use synchronous copy via direct memory copy
        // (Backend vtable handles real GPU transfers; for unified buffer
        // tracking we simulate with memcpy to device-mapped memory)
        if (self.host_data) |src| {
            if (self.device_handle) |dst| {
                const dst_ptr: [*]u8 = @ptrCast(dst);
                @memcpy(dst_ptr[0..src.len], src);
            } else {
                return error.NoDeviceMemory;
            }
        } else {
            return error.NoHostMemory;
        }

        self.clearHostDirty();
        self.markDeviceDirty();

        self.host_to_device_transfers += 1;
        self.bytes_transferred += self.size;

        // Signal that transfer is complete
        self.sync_event.record();
    }

    /// Transfer data from host to device asynchronously.
    /// Does not block host. Synchronization should be handled via the stream.
    pub fn toDeviceAsync(self: *Buffer, stream: ?*anyopaque) !void {
        _ = stream; // Stream-based async not yet implemented at buffer level
        if (!self.isHostDirty()) return;

        // Use direct memory copy (async would require backend integration)
        if (self.host_data) |src| {
            if (self.device_handle) |dst| {
                const dst_ptr: [*]u8 = @ptrCast(dst);
                @memcpy(dst_ptr[0..src.len], src);
            } else {
                return error.NoDeviceMemory;
            }
        } else {
            return error.NoHostMemory;
        }

        self.clearHostDirty();
        self.markDeviceDirty();

        self.host_to_device_transfers += 1;
        self.bytes_transferred += self.size;
    }

    /// Transfer data from device to host (explicit mode).
    pub fn toHost(self: *Buffer) !void {
        if (!self.isDeviceDirty()) {
            return; // No need to transfer
        }

        // Use synchronous copy via direct memory copy
        // (Backend vtable handles real GPU transfers; for unified buffer
        // tracking we simulate with memcpy from device-mapped memory)
        if (self.host_data) |dst| {
            if (self.device_handle) |src| {
                const src_ptr: [*]const u8 = @ptrCast(src);
                @memcpy(dst[0..self.size], src_ptr[0..self.size]);
            } else {
                return error.NoDeviceMemory;
            }
        } else {
            return error.NoHostMemory;
        }

        self.clearDeviceDirty();

        self.device_to_host_transfers += 1;
        self.bytes_transferred += self.size;
    }

    /// Transfer data from device to host asynchronously.
    /// Does not block host. Synchronization should be handled via the stream.
    pub fn toHostAsync(self: *Buffer, stream: ?*anyopaque) !void {
        _ = stream; // Stream-based async not yet implemented at buffer level
        if (!self.isDeviceDirty()) return;

        // Use direct memory copy (async would require backend integration)
        if (self.host_data) |dst| {
            if (self.device_handle) |src| {
                const src_ptr: [*]const u8 = @ptrCast(src);
                @memcpy(dst[0..self.size], src_ptr[0..self.size]);
            } else {
                return error.NoDeviceMemory;
            }
        } else {
            return error.NoHostMemory;
        }

        self.clearDeviceDirty();

        self.device_to_host_transfers += 1;
        self.bytes_transferred += self.size;
    }

    /// Get a device pointer for kernel execution.
    pub fn getDevicePtr(self: *Buffer) !*anyopaque {
        // Ensure device has latest data
        if (self.mode == .automatic and self.isHostDirty()) {
            try self.toDevice();
        }

        if (self.device_handle) |handle| {
            return handle;
        }

        return error.NoDeviceMemory;
    }

    /// Check if buffer has a device handle allocated.
    pub fn hasDeviceHandle(self: *const Buffer) bool {
        return self.device_handle != null;
    }

    /// Get raw host bytes (for backend transfers).
    pub fn getHostBytes(self: *const Buffer) ?[]u8 {
        return self.host_data;
    }

    /// Set the device handle (called by backend after allocation).
    pub fn setDeviceHandle(self: *Buffer, handle: *anyopaque) void {
        self.device_handle = handle;
    }

    /// Map the buffer for host access.
    pub fn map(self: *Buffer, access: MapAccess) !MappedBuffer {
        // Sync from device if needed
        if (access != .write and self.isDeviceDirty()) {
            try self.toHost();
        }

        const data = self.host_data orelse return error.NoHostMemory;

        return MappedBuffer{
            .buffer = self,
            .data = data,
            .access = access,
        };
    }

    /// Create a view into a portion of this buffer.
    pub fn slice(self: *Buffer, offset: usize, len: usize) !BufferView {
        if (offset + len > self.size) {
            return error.BufferOverflow;
        }

        return BufferView{
            .buffer = self,
            .offset = offset,
            .len = len,
        };
    }

    /// Mark host data as modified.
    /// Resets the sync event to indicate pending transfer.
    pub fn markHostDirty(self: *Buffer) void {
        _ = self.dirty.fetchOr(0b01, .release);
        self.sync_event.reset();
    }

    /// Mark device data as modified.
    pub fn markDeviceDirty(self: *Buffer) void {
        _ = self.dirty.fetchOr(0b10, .release);
    }

    /// Clear host dirty flag.
    pub fn clearHostDirty(self: *Buffer) void {
        _ = self.dirty.fetchAnd(0b10, .release);
    }

    /// Clear device dirty flag.
    pub fn clearDeviceDirty(self: *Buffer) void {
        _ = self.dirty.fetchAnd(0b01, .release);
    }

    /// Check if host data is dirty.
    pub fn isHostDirty(self: *const Buffer) bool {
        return (self.dirty.load(.acquire) & 0b01) != 0;
    }

    /// Check if device data is dirty.
    pub fn isDeviceDirty(self: *const Buffer) bool {
        return (self.dirty.load(.acquire) & 0b10) != 0;
    }

    /// Check if data is synchronized.
    pub fn isSynchronized(self: *const Buffer) bool {
        return self.dirty.load(.acquire) == 0;
    }

    /// Get the sync event for non-blocking synchronization checks.
    /// Use isComplete() to check if transfer is done without blocking,
    /// or wait() to block until transfer completes.
    pub fn getSyncEvent(self: *Buffer) *SyncEvent {
        return &self.sync_event;
    }

    /// Check if the last transfer operation has completed.
    /// Non-blocking - returns immediately.
    pub fn isSyncComplete(self: *const Buffer) bool {
        return self.sync_event.isComplete();
    }

    /// Wait for the last transfer operation to complete.
    /// Blocking - returns when transfer is done.
    pub fn waitForSync(self: *Buffer) void {
        self.sync_event.wait();
    }

    /// Wait for sync with timeout.
    /// Returns true if sync completed, false if timeout expired.
    pub fn waitForSyncTimeout(self: *Buffer, timeout_ns: u64) bool {
        return self.sync_event.waitTimeout(timeout_ns);
    }

    /// Get transfer statistics.
    pub fn getStats(self: *const Buffer) BufferStats {
        return .{
            .host_to_device_transfers = self.host_to_device_transfers,
            .device_to_host_transfers = self.device_to_host_transfers,
            .bytes_transferred = self.bytes_transferred,
        };
    }

    /// Fill the buffer with a value.
    pub fn fill(self: *Buffer, comptime T: type, value: T) !void {
        if (self.host_data) |host| {
            const typed: []T = @alignCast(std.mem.bytesAsSlice(T, host));
            @memset(typed, value);
            self.markHostDirty();

            if (self.mode == .automatic) {
                try self.toDevice();
            }
        } else {
            return error.NoHostMemory;
        }
    }

    /// Copy data from another buffer.
    pub fn copyFrom(self: *Buffer, source: *Buffer) !void {
        if (source.size > self.size) {
            return error.BufferOverflow;
        }

        // Ensure source is on host
        if (source.isDeviceDirty()) {
            try source.toHost();
        }

        if (self.host_data) |dest| {
            if (source.host_data) |src| {
                @memcpy(dest[0..source.size], src);
                self.markHostDirty();

                if (self.mode == .automatic) {
                    try self.toDevice();
                }
            } else {
                return error.NoHostMemory;
            }
        } else {
            return error.NoHostMemory;
        }
    }
};

/// Mapped buffer for direct host access.
pub const MappedBuffer = struct {
    buffer: *Buffer,
    data: []u8,
    access: MapAccess,

    /// Get typed slice.
    pub fn asSlice(self: *MappedBuffer, comptime T: type) []T {
        return @alignCast(std.mem.bytesAsSlice(T, self.data));
    }

    /// Unmap and commit changes.
    pub fn unmap(self: *MappedBuffer) !void {
        if (self.access != .read) {
            self.buffer.markHostDirty();

            if (self.buffer.mode == .automatic) {
                try self.buffer.toDevice();
            }
        }
    }
};

/// View into a portion of a buffer.
pub const BufferView = struct {
    buffer: *Buffer,
    offset: usize,
    len: usize,

    /// Get the offset in bytes.
    pub fn getOffset(self: *const BufferView) usize {
        return self.offset;
    }

    /// Get the length in bytes.
    pub fn getLen(self: *const BufferView) usize {
        return self.len;
    }

    /// Get underlying buffer.
    pub fn getBuffer(self: *const BufferView) *Buffer {
        return self.buffer;
    }
};

/// Buffer statistics.
pub const BufferStats = struct {
    host_to_device_transfers: u64,
    device_to_host_transfers: u64,
    bytes_transferred: u64,
};

/// Create a buffer from a typed slice.
pub fn createFromSlice(
    allocator: std.mem.Allocator,
    comptime T: type,
    data: []const T,
    device: *const Device,
    options: BufferOptions,
) !Buffer {
    const bytes = std.mem.sliceAsBytes(data);
    var opts = options;
    opts.element_type = ElementType.fromType(T);
    opts.initial_data = bytes;

    return Buffer.init(allocator, bytes.len, device, opts);
}

// ============================================================================
// Tests
// ============================================================================

test "Buffer basic operations" {
    const device = Device{
        .id = 0,
        .backend = .vulkan,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var buffer = try Buffer.init(std.testing.allocator, 1024, &device, .{});
    defer buffer.deinit();

    try std.testing.expect(buffer.getSize() == 1024);
    try std.testing.expect(!buffer.isHostDirty());
    try std.testing.expect(!buffer.isDeviceDirty());
}

test "Buffer write and read" {
    const device = Device{
        .id = 0,
        .backend = .cuda,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var buffer = try Buffer.init(std.testing.allocator, 16 * @sizeOf(f32), &device, .{
        .mode = .explicit,
        .element_type = .f32,
    });
    defer buffer.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try buffer.write(f32, &input);

    try std.testing.expect(buffer.isHostDirty());

    var output: [4]f32 = undefined;
    try buffer.read(f32, &output);

    try std.testing.expectEqualSlices(f32, &input, &output);
}

test "Buffer dirty state tracking" {
    const device = Device{
        .id = 0,
        .backend = .metal,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var buffer = try Buffer.init(std.testing.allocator, 256, &device, .{
        .mode = .explicit,
    });
    defer buffer.deinit();

    // Initially clean
    try std.testing.expect(!buffer.isHostDirty());
    try std.testing.expect(!buffer.isDeviceDirty());

    // Write makes host dirty
    try buffer.writeBytes(&[_]u8{ 1, 2, 3, 4 });
    try std.testing.expect(buffer.isHostDirty());

    // toDevice clears host dirty, sets device "current"
    try buffer.toDevice();
    try std.testing.expect(!buffer.isHostDirty());

    // Simulate device modification
    buffer.markDeviceDirty();
    try std.testing.expect(buffer.isDeviceDirty());

    // toHost clears device dirty
    try buffer.toHost();
    try std.testing.expect(!buffer.isDeviceDirty());
}

test "Buffer createFromSlice" {
    const device = Device{
        .id = 0,
        .backend = .webgpu,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var buffer = try createFromSlice(std.testing.allocator, i32, &data, &device, .{
        .mode = .explicit,
    });
    defer buffer.deinit();

    try std.testing.expect(buffer.getSize() == 5 * @sizeOf(i32));
    try std.testing.expect(buffer.element_type == .i32);

    var output: [5]i32 = undefined;
    try buffer.read(i32, &output);
    try std.testing.expectEqualSlices(i32, &data, &output);
}

test "Buffer map and unmap" {
    const device = Device{
        .id = 0,
        .backend = .vulkan,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var buffer = try Buffer.init(std.testing.allocator, 64, &device, .{
        .mode = .explicit,
    });
    defer buffer.deinit();

    var mapped = try buffer.map(.write);
    mapped.data[0] = 42;
    mapped.data[1] = 43;
    try mapped.unmap();

    try std.testing.expect(buffer.isHostDirty());
}

test "Buffer slice" {
    const device = Device{
        .id = 0,
        .backend = .cuda,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var buffer = try Buffer.init(std.testing.allocator, 1024, &device, .{});
    defer buffer.deinit();

    const view = try buffer.slice(100, 200);
    try std.testing.expect(view.getOffset() == 100);
    try std.testing.expect(view.getLen() == 200);
}

test "ElementType size" {
    try std.testing.expect(ElementType.u8.size() == 1);
    try std.testing.expect(ElementType.f32.size() == 4);
    try std.testing.expect(ElementType.f64.size() == 8);
}
