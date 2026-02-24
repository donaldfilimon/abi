//! Apple Silicon Unified Memory Manager
//!
//! Provides efficient memory management for Apple's unified memory architecture,
//! enabling zero-copy data sharing between CPU and GPU for neural network training.
//!
//! ## Features
//! - Zero-copy buffer sharing between CPU and GPU
//! - Automatic memory migration tracking
//! - Page-aligned allocations for optimal performance
//! - Integration with Metal's resource storage modes
//!
//! ## Apple Silicon Memory Architecture
//! On Apple Silicon, CPU and GPU share the same physical memory pool. This means:
//! - No explicit memory copies needed between CPU and GPU
//! - Both processors can access the same data simultaneously
//! - Synchronization is still required for coherency
//!
//! ## Usage
//! ```zig
//! const unified = @import("unified_memory");
//!
//! var manager = try unified.UnifiedMemoryManager.init(allocator, .{});
//! defer manager.deinit();
//!
//! // Allocate unified buffer
//! const buffer = try manager.alloc(f32, 1024);
//! defer manager.free(buffer);
//!
//! // CPU writes are visible to GPU (after sync point)
//! buffer[0] = 1.0;
//! manager.syncForGpu(buffer);
//!
//! // GPU writes are visible to CPU (after sync point)
//! manager.syncForCpu(buffer);
//! const value = buffer[0];
//! ```

const std = @import("std");
const builtin = @import("builtin");
const time = @import("../../../../services/shared/time.zig");
const accelerate = @import("accelerate.zig");

/// Configuration for the unified memory manager
pub const UnifiedMemoryConfig = struct {
    /// Initial pool size in bytes (0 = no pre-allocation)
    initial_pool_size: usize = 0,
    /// Maximum memory that can be allocated (0 = unlimited)
    max_memory: usize = 0,
    /// Alignment for allocations (default: 16KB for Apple Silicon)
    alignment: usize = 16 * 1024,
    /// Enable allocation tracking for debugging
    track_allocations: bool = false,
    /// Prefer shared storage mode (visible to both CPU and GPU)
    prefer_shared: bool = true,
};

/// Storage mode for unified memory buffers
pub const StorageMode = enum {
    /// Shared: CPU and GPU can both access (default for unified memory)
    shared,
    /// Managed: System manages CPU/GPU coherency automatically
    managed,
    /// Private: GPU-only access (fastest for GPU-bound data)
    private,

    /// Get the optimal storage mode for a given access pattern
    pub fn forAccessPattern(cpu_read: bool, cpu_write: bool, gpu_read: bool, gpu_write: bool) StorageMode {
        if (!cpu_read and !cpu_write) {
            // GPU-only access
            return .private;
        }
        if (cpu_read and cpu_write and gpu_read and gpu_write) {
            // Full bidirectional access - shared is optimal on Apple Silicon
            return .shared;
        }
        // Mixed access - let system manage
        return .managed;
    }
};

/// Allocation metadata for tracking
const AllocationInfo = struct {
    ptr: [*]u8,
    size: usize,
    alignment: usize,
    storage_mode: StorageMode,
    timestamp: i64,
};

/// Unified memory manager for Apple Silicon
pub const UnifiedMemoryManager = struct {
    allocator: std.mem.Allocator,
    config: UnifiedMemoryConfig,
    allocations: std.AutoHashMapUnmanaged(*anyopaque, AllocationInfo),
    total_allocated: usize,
    peak_allocated: usize,
    allocation_count: usize,

    const Self = @This();

    fn resolvedAlignmentBytes(self: *const Self, comptime T: type) usize {
        const requested = @max(self.config.alignment, @alignOf(T));
        std.debug.assert(requested > 0);
        return std.math.ceilPowerOfTwoAssert(usize, requested);
    }

    fn resolvedAlignment(self: *const Self, comptime T: type) std.mem.Alignment {
        return std.mem.Alignment.fromByteUnits(self.resolvedAlignmentBytes(T));
    }

    /// Initialize the unified memory manager
    pub fn init(allocator: std.mem.Allocator, config: UnifiedMemoryConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
            .allocations = .empty,
            .total_allocated = 0,
            .peak_allocated = 0,
            .allocation_count = 0,
        };

        // Pre-allocate tracking capacity if tracking is enabled
        if (config.track_allocations) {
            try self.allocations.ensureTotalCapacity(allocator, 256);
        }

        return self;
    }

    /// Deinitialize and free all tracked allocations
    pub fn deinit(self: *Self) void {
        // Free all tracked allocations
        var iter = self.allocations.iterator();
        while (iter.next()) |entry| {
            const info = entry.value_ptr.*;
            const slice = info.ptr[0..info.size];
            const alignment = std.mem.Alignment.fromByteUnits(info.alignment);
            self.allocator.rawFree(slice, alignment, @returnAddress());
        }
        self.allocations.deinit(self.allocator);
    }

    /// Allocate unified memory buffer
    pub fn alloc(self: *Self, comptime T: type, count: usize) ![]T {
        return self.allocWithMode(T, count, .shared);
    }

    /// Allocate unified memory buffer with specific storage mode
    pub fn allocWithMode(self: *Self, comptime T: type, count: usize, mode: StorageMode) ![]T {
        if (count == 0) {
            return self.allocator.alloc(T, 0);
        }

        const size = count * @sizeOf(T);
        const alignment_bytes = self.resolvedAlignmentBytes(T);
        const alignment = std.mem.Alignment.fromByteUnits(alignment_bytes);

        // Check max memory limit
        if (self.config.max_memory > 0 and self.total_allocated + size > self.config.max_memory) {
            return error.OutOfMemory;
        }

        // Allocate with proper alignment
        const byte_ptr = self.allocator.rawAlloc(size, alignment, @returnAddress()) orelse return error.OutOfMemory;
        const aligned_ptr: [*]align(@alignOf(T)) u8 = @alignCast(byte_ptr);
        const ptr: [*]T = @ptrCast(aligned_ptr);
        const slice = ptr[0..count];

        // Track allocation if enabled
        if (self.config.track_allocations) {
            try self.allocations.put(self.allocator, @ptrCast(byte_ptr), .{
                .ptr = byte_ptr,
                .size = size,
                .alignment = alignment_bytes,
                .storage_mode = mode,
                .timestamp = time.nowMs(),
            });
        }

        self.total_allocated += size;
        self.peak_allocated = @max(self.peak_allocated, self.total_allocated);
        self.allocation_count += 1;

        return slice;
    }

    /// Free unified memory buffer
    pub fn free(self: *Self, ptr: anytype) void {
        const T = @TypeOf(ptr);
        const info = @typeInfo(T);

        if (info != .pointer) {
            @compileError("free expects a slice type");
        }

        if (ptr.len == 0) return;

        const size = ptr.len * @sizeOf(info.pointer.child);
        var alignment = self.resolvedAlignment(info.pointer.child);

        // Remove from tracking
        if (self.config.track_allocations) {
            if (self.allocations.fetchRemove(@ptrCast(ptr.ptr))) |entry| {
                alignment = std.mem.Alignment.fromByteUnits(entry.value.alignment);
            }
        }

        self.total_allocated -= size;
        const byte_ptr: [*]u8 = @ptrCast(ptr.ptr);
        self.allocator.rawFree(byte_ptr[0..size], alignment, @returnAddress());
    }

    /// Synchronization point for CPU to GPU data transfer.
    /// On unified memory, this is a memory barrier rather than a copy.
    /// NOTE: With real Metal device integration, this would use
    /// MTLBlitCommandEncoder.synchronizeResource to flush CPU caches
    /// and ensure GPU coherency. The atomic fence serves as the
    /// software-only fallback for the current implementation.
    pub fn syncForGpu(self: *Self, ptr: anytype) void {
        _ = self;
        // On Apple Silicon unified memory, we need a memory barrier
        // to ensure CPU writes are visible to GPU
        std.atomic.fence(.release);
        _ = ptr;
    }

    /// Synchronization point for GPU to CPU data transfer.
    /// On unified memory, this is a memory barrier rather than a copy.
    /// NOTE: With real Metal device integration, this would use
    /// MTLBlitCommandEncoder.synchronizeResource to invalidate GPU caches
    /// and ensure CPU coherency. The atomic fence serves as the
    /// software-only fallback for the current implementation.
    pub fn syncForCpu(self: *Self, ptr: anytype) void {
        _ = self;
        // On Apple Silicon unified memory, we need a memory barrier
        // to ensure GPU writes are visible to CPU
        std.atomic.fence(.acquire);
        _ = ptr;
    }

    /// Get statistics about memory usage
    pub fn getStats(self: *const Self) MemoryStats {
        return .{
            .total_allocated = self.total_allocated,
            .peak_allocated = self.peak_allocated,
            .allocation_count = self.allocation_count,
            .has_unified_memory = accelerate.hasUnifiedMemory(),
        };
    }

    /// Check if this system has unified memory (Apple Silicon)
    pub fn hasUnifiedMemory() bool {
        return accelerate.hasUnifiedMemory();
    }

    /// Create a MetalBufferHandle backed by page-aligned memory.
    /// On macOS with Metal available, this would create an MTLBuffer via
    /// [device newBufferWithLength:options:] and map its contents pointer.
    /// Falls back to a page-aligned CPU allocation on other platforms.
    pub fn createMetalBuffer(self: *Self, size: usize, mode: StorageMode) !MetalBufferHandle {
        if (size == 0) {
            return MetalBufferHandle{ .storage_mode = mode };
        }

        // Allocate page-aligned backing memory as fallback / software path.
        // In a full Metal integration, this would call [device newBufferWithLength:options:]
        // and use [buffer contents] for the CPU-mapped pointer.
        const alignment_bytes = self.resolvedAlignmentBytes(u8);
        const alignment = std.mem.Alignment.fromByteUnits(alignment_bytes);
        const byte_ptr = self.allocator.rawAlloc(size, alignment, @returnAddress()) orelse return error.OutOfMemory;

        // Zero-initialize
        @memset(byte_ptr[0..size], 0);

        // Track allocation
        if (self.config.track_allocations) {
            try self.allocations.put(self.allocator, @ptrCast(byte_ptr), .{
                .ptr = byte_ptr,
                .size = size,
                .alignment = alignment_bytes,
                .storage_mode = mode,
                .timestamp = time.nowMs(),
            });
        }

        self.total_allocated += size;
        self.peak_allocated = @max(self.peak_allocated, self.total_allocated);
        self.allocation_count += 1;

        return MetalBufferHandle{
            .buffer_id = null, // Would be MTLBuffer ID with real Metal device
            .contents = byte_ptr,
            .length = size,
            .storage_mode = mode,
        };
    }

    /// Release a MetalBufferHandle created by createMetalBuffer.
    /// Frees the backing allocation or releases the MTLBuffer.
    pub fn destroyMetalBuffer(self: *Self, handle: *MetalBufferHandle) void {
        if (handle.contents == null or handle.length == 0) {
            handle.* = .{};
            return;
        }

        const ptr = handle.contents.?;
        var alignment_bytes = self.resolvedAlignmentBytes(u8);

        // Remove from tracking
        if (self.config.track_allocations) {
            if (self.allocations.fetchRemove(@ptrCast(ptr))) |entry| {
                alignment_bytes = entry.value.alignment;
            }
        }

        const alignment = std.mem.Alignment.fromByteUnits(alignment_bytes);
        self.total_allocated -= handle.length;
        self.allocator.rawFree(ptr[0..handle.length], alignment, @returnAddress());
        handle.* = .{};
    }

    /// Wrap an existing typed slice as a MetalBufferHandle without allocating.
    /// The handle provides a view over the slice's memory; the caller retains ownership.
    pub fn wrapSliceAsBuffer(self: *Self, comptime T: type, data: []T) MetalBufferHandle {
        _ = self;
        if (data.len == 0) return .{};
        const byte_ptr: [*]u8 = @ptrCast(data.ptr);
        return MetalBufferHandle{
            .buffer_id = null,
            .contents = byte_ptr,
            .length = data.len * @sizeOf(T),
            .storage_mode = .shared,
        };
    }
};

/// Handle to a Metal buffer for GPU memory operations
pub const MetalBufferHandle = struct {
    /// MTLBuffer Obj-C ID
    buffer_id: ?*anyopaque = null,
    /// Mapped CPU pointer from [buffer contents]
    contents: ?[*]u8 = null,
    /// Buffer size in bytes
    length: usize = 0,
    /// Metal storage mode
    storage_mode: StorageMode = .shared,

    /// Cast buffer contents to a typed slice
    pub fn asSlice(self: MetalBufferHandle, comptime T: type) ?[]T {
        const ptr = self.contents orelse return null;
        if (self.length < @sizeOf(T)) return null;
        const count = self.length / @sizeOf(T);
        const typed: [*]T = @ptrCast(@alignCast(ptr));
        return typed[0..count];
    }
};

/// Memory statistics
pub const MemoryStats = struct {
    total_allocated: usize,
    peak_allocated: usize,
    allocation_count: usize,
    has_unified_memory: bool,
};

/// Zero-copy tensor view for unified memory
/// Allows both CPU and GPU to access the same data without copies
pub fn UnifiedTensor(comptime T: type) type {
    return struct {
        data: []T,
        shape: []const usize,
        strides: []const usize,
        manager: *UnifiedMemoryManager,

        const Self = @This();

        /// Create a new unified tensor
        pub fn init(manager: *UnifiedMemoryManager, shape: []const usize) !Self {
            // Calculate total elements
            var total: usize = 1;
            for (shape) |dim| {
                total *= dim;
            }

            // Calculate strides (row-major)
            const strides = try manager.allocator.alloc(usize, shape.len);
            var stride: usize = 1;
            var i = shape.len;
            while (i > 0) {
                i -= 1;
                strides[i] = stride;
                stride *= shape[i];
            }

            // Allocate data
            const data = try manager.alloc(T, total);

            // Copy shape
            const shape_copy = try manager.allocator.dupe(usize, shape);

            return Self{
                .data = data,
                .shape = shape_copy,
                .strides = strides,
                .manager = manager,
            };
        }

        /// Free the tensor
        pub fn deinit(self: *Self) void {
            self.manager.free(self.data);
            self.manager.allocator.free(self.strides);
            self.manager.allocator.free(@constCast(self.shape));
        }

        /// Get element at indices
        pub fn get(self: *const Self, indices: []const usize) T {
            var offset: usize = 0;
            for (indices, self.strides) |idx, stride| {
                offset += idx * stride;
            }
            return self.data[offset];
        }

        /// Set element at indices
        pub fn set(self: *Self, indices: []const usize, value: T) void {
            var offset: usize = 0;
            for (indices, self.strides) |idx, stride| {
                offset += idx * stride;
            }
            self.data[offset] = value;
        }

        /// Sync for GPU access
        pub fn syncForGpu(self: *Self) void {
            self.manager.syncForGpu(self.data);
        }

        /// Sync for CPU access
        pub fn syncForCpu(self: *Self) void {
            self.manager.syncForCpu(self.data);
        }

        /// Get total number of elements
        pub fn numel(self: *const Self) usize {
            return self.data.len;
        }

        /// Get the number of dimensions
        pub fn ndim(self: *const Self) usize {
            return self.shape.len;
        }

        /// Return a MetalBufferHandle pointing at this tensor's data buffer.
        /// The handle provides typed access via asSlice() and can be passed to
        /// Metal APIs that accept buffer pointers.
        pub fn asMetalBuffer(self: *const Self) ?MetalBufferHandle {
            if (self.data.len == 0) return null;
            const byte_ptr: [*]u8 = @ptrCast(self.data.ptr);
            return MetalBufferHandle{
                .buffer_id = null,
                .contents = byte_ptr,
                .length = self.data.len * @sizeOf(T),
                .storage_mode = .shared,
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "unified memory manager basic operations" {
    const allocator = std.testing.allocator;

    var manager = try UnifiedMemoryManager.init(allocator, .{
        .track_allocations = true,
    });
    defer manager.deinit();

    // Allocate buffer
    const buffer = try manager.alloc(f32, 1024);
    defer manager.free(buffer);

    // Write and read
    buffer[0] = 42.0;
    try std.testing.expectEqual(@as(f32, 42.0), buffer[0]);

    // Check stats
    const stats = manager.getStats();
    try std.testing.expect(stats.total_allocated > 0);
    try std.testing.expect(stats.allocation_count == 1);
}

test "unified tensor operations" {
    const allocator = std.testing.allocator;

    var manager = try UnifiedMemoryManager.init(allocator, .{});
    defer manager.deinit();

    // Create 2D tensor (3x4)
    var tensor = try UnifiedTensor(f32).init(&manager, &.{ 3, 4 });
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 12), tensor.numel());
    try std.testing.expectEqual(@as(usize, 2), tensor.ndim());

    // Set and get values
    tensor.set(&.{ 1, 2 }, 3.14);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), tensor.get(&.{ 1, 2 }), 0.001);
}

test "storage mode selection" {
    // GPU only
    try std.testing.expectEqual(StorageMode.private, StorageMode.forAccessPattern(false, false, true, true));

    // Full bidirectional
    try std.testing.expectEqual(StorageMode.shared, StorageMode.forAccessPattern(true, true, true, true));

    // Mixed
    try std.testing.expectEqual(StorageMode.managed, StorageMode.forAccessPattern(true, false, true, true));
}

test "MetalBufferHandle asSlice type casting" {
    // Test with f32 data
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const byte_ptr: [*]u8 = @ptrCast(&data);
    const handle = MetalBufferHandle{
        .contents = byte_ptr,
        .length = @sizeOf(f32) * data.len,
        .storage_mode = .shared,
    };

    const slice = handle.asSlice(f32);
    try std.testing.expect(slice != null);
    try std.testing.expectEqual(@as(usize, 4), slice.?.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), slice.?[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), slice.?[3], 0.001);

    // Null contents should return null
    const empty_handle = MetalBufferHandle{};
    try std.testing.expect(empty_handle.asSlice(f32) == null);

    // Buffer too small for type should return null
    const tiny_handle = MetalBufferHandle{
        .contents = byte_ptr,
        .length = 2, // less than @sizeOf(f32)
    };
    try std.testing.expect(tiny_handle.asSlice(f32) == null);
}

test "wrapSliceAsBuffer round-trip" {
    const allocator = std.testing.allocator;

    var manager = try UnifiedMemoryManager.init(allocator, .{});
    defer manager.deinit();

    // Create a slice and wrap it
    var data = [_]f32{ 10.0, 20.0, 30.0 };
    const handle = manager.wrapSliceAsBuffer(f32, &data);

    // Verify handle properties
    try std.testing.expect(handle.contents != null);
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), handle.length);
    try std.testing.expectEqual(StorageMode.shared, handle.storage_mode);

    // Round-trip: get the data back as a typed slice
    const recovered = handle.asSlice(f32);
    try std.testing.expect(recovered != null);
    try std.testing.expectEqual(@as(usize, 3), recovered.?.len);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), recovered.?[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), recovered.?[2], 0.001);

    // Modify through original, verify through handle
    data[1] = 99.0;
    try std.testing.expectApproxEqAbs(@as(f32, 99.0), recovered.?[1], 0.001);

    // Empty slice wrap
    const empty: []f32 = &.{};
    const empty_handle = manager.wrapSliceAsBuffer(f32, empty);
    try std.testing.expect(empty_handle.contents == null);
    try std.testing.expectEqual(@as(usize, 0), empty_handle.length);
}

test "createMetalBuffer and destroyMetalBuffer lifecycle" {
    const allocator = std.testing.allocator;

    var manager = try UnifiedMemoryManager.init(allocator, .{
        .track_allocations = true,
    });
    defer manager.deinit();

    // Create a buffer
    var handle = try manager.createMetalBuffer(256, .shared);
    try std.testing.expect(handle.contents != null);
    try std.testing.expectEqual(@as(usize, 256), handle.length);
    try std.testing.expectEqual(StorageMode.shared, handle.storage_mode);
    try std.testing.expect(manager.total_allocated >= 256);
    try std.testing.expectEqual(@as(usize, 1), manager.allocation_count);

    // Write through contents pointer
    const f32_slice = handle.asSlice(f32);
    try std.testing.expect(f32_slice != null);
    f32_slice.?[0] = 42.0;
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), f32_slice.?[0], 0.001);

    // Create a second buffer with different mode
    var handle2 = try manager.createMetalBuffer(512, .private);
    try std.testing.expectEqual(@as(usize, 512), handle2.length);
    try std.testing.expectEqual(StorageMode.private, handle2.storage_mode);
    try std.testing.expectEqual(@as(usize, 2), manager.allocation_count);

    // Destroy first buffer
    const alloc_before = manager.total_allocated;
    manager.destroyMetalBuffer(&handle);
    try std.testing.expect(handle.contents == null);
    try std.testing.expectEqual(@as(usize, 0), handle.length);
    try std.testing.expect(manager.total_allocated < alloc_before);

    // Destroy second buffer
    manager.destroyMetalBuffer(&handle2);
    try std.testing.expect(handle2.contents == null);

    // Zero-size buffer should not crash
    var zero_handle = try manager.createMetalBuffer(0, .managed);
    try std.testing.expect(zero_handle.contents == null);
    try std.testing.expectEqual(@as(usize, 0), zero_handle.length);
    manager.destroyMetalBuffer(&zero_handle);
}

test {
    std.testing.refAllDecls(@This());
}
