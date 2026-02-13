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
    allocations: std.AutoHashMap(*anyopaque, AllocationInfo),
    total_allocated: usize,
    peak_allocated: usize,
    allocation_count: usize,

    const Self = @This();

    /// Initialize the unified memory manager
    pub fn init(allocator: std.mem.Allocator, config: UnifiedMemoryConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
            .allocations = std.AutoHashMap(*anyopaque, AllocationInfo).init(allocator),
            .total_allocated = 0,
            .peak_allocated = 0,
            .allocation_count = 0,
        };

        // Pre-allocate tracking capacity if tracking is enabled
        if (config.track_allocations) {
            try self.allocations.ensureTotalCapacity(256);
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
            self.allocator.free(@as([]align(1) u8, @alignCast(slice)));
        }
        self.allocations.deinit();
    }

    /// Allocate unified memory buffer
    pub fn alloc(self: *Self, comptime T: type, count: usize) ![]T {
        return self.allocWithMode(T, count, .shared);
    }

    /// Allocate unified memory buffer with specific storage mode
    pub fn allocWithMode(self: *Self, comptime T: type, count: usize, mode: StorageMode) ![]T {
        const size = count * @sizeOf(T);
        const alignment = @max(self.config.alignment, @alignOf(T));

        // Check max memory limit
        if (self.config.max_memory > 0 and self.total_allocated + size > self.config.max_memory) {
            return error.OutOfMemory;
        }

        // Allocate with proper alignment
        const ptr = try self.allocator.alignedAlloc(T, alignment, count);

        // Track allocation if enabled
        if (self.config.track_allocations) {
            try self.allocations.put(@ptrCast(ptr.ptr), .{
                .ptr = @ptrCast(ptr.ptr),
                .size = size,
                .alignment = alignment,
                .storage_mode = mode,
                .timestamp = time.nowMs(),
            });
        }

        self.total_allocated += size;
        self.peak_allocated = @max(self.peak_allocated, self.total_allocated);
        self.allocation_count += 1;

        return ptr;
    }

    /// Free unified memory buffer
    pub fn free(self: *Self, ptr: anytype) void {
        const T = @TypeOf(ptr);
        const info = @typeInfo(T);

        if (info != .pointer) {
            @compileError("free expects a slice type");
        }

        const size = ptr.len * @sizeOf(info.pointer.child);

        // Remove from tracking
        if (self.config.track_allocations) {
            _ = self.allocations.remove(@ptrCast(ptr.ptr));
        }

        self.total_allocated -= size;
        self.allocator.free(ptr);
    }

    /// Synchronization point for CPU to GPU data transfer
    /// On unified memory, this is a memory barrier rather than a copy
    pub fn syncForGpu(self: *Self, ptr: anytype) void {
        _ = self;
        // On Apple Silicon unified memory, we need a memory barrier
        // to ensure CPU writes are visible to GPU
        std.atomic.fence(.release);
        _ = ptr;
    }

    /// Synchronization point for GPU to CPU data transfer
    /// On unified memory, this is a memory barrier rather than a copy
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
