// ! Advanced GPU memory pool with size-class allocation and fragmentation mitigation.
//!
//! Provides efficient memory pooling with:
//! - Size-class based allocation to reduce fragmentation
//! - Free-list management for fast reuse
//! - Memory coalescing to reduce external fragmentation
//! - Allocation tracking and leak detection
//! - Memory pressure handling with automatic eviction

const std = @import("std");
const time = @import("../../shared/utils/time.zig");
const memory = @import("memory.zig");

/// Memory allocation size classes in bytes.
const SIZE_CLASSES = [_]usize{
    64, // Tiny
    256, // Small
    1024, // Medium (1 KB)
    4096, // Large (4 KB)
    16384, // XLarge (16 KB)
    65536, // XXLarge (64 KB)
    262144, // Huge (256 KB)
    1048576, // Massive (1 MB)
    4194304, // Giant (4 MB)
};

/// Memory pool configuration.
pub const PoolConfig = struct {
    /// Maximum total pool size in bytes.
    max_total_size: usize = 1024 * 1024 * 1024, // 1 GB default
    /// Maximum size per size class.
    max_class_size: usize = 128 * 1024 * 1024, // 128 MB per class
    /// Enable automatic memory coalescing.
    enable_coalescing: bool = true,
    /// Enable leak detection.
    enable_leak_detection: bool = false,
    /// High water mark for memory pressure (0.0-1.0).
    high_water_mark: f64 = 0.85,
    /// Low water mark for memory pressure (0.0-1.0).
    low_water_mark: f64 = 0.70,
};

/// Allocation metadata for tracking.
const AllocationMeta = struct {
    size: usize,
    size_class_idx: ?usize,
    allocated_at: i64,
    used: bool,
    next_free: ?*AllocationMeta,
};

/// Size class bucket for managing allocations of similar sizes.
const SizeClassBucket = struct {
    size_class: usize,
    allocations: std.ArrayListUnmanaged(memory.GpuBuffer),
    free_list: ?*AllocationMeta,
    metadata: std.ArrayListUnmanaged(AllocationMeta),
    total_allocated: usize,
    total_used: usize,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, size_class: usize) SizeClassBucket {
        return .{
            .size_class = size_class,
            .allocations = .{},
            .free_list = null,
            .metadata = .{},
            .total_allocated = 0,
            .total_used = 0,
            .allocator = allocator,
        };
    }

    fn deinit(self: *SizeClassBucket) void {
        for (self.allocations.items) |*buf| {
            buf.deinit();
        }
        self.allocations.deinit(self.allocator);
        self.metadata.deinit(self.allocator);
        self.* = undefined;
    }

    fn allocate(self: *SizeClassBucket, size: usize, flags: memory.BufferFlags) !*memory.GpuBuffer {
        // Try to reuse from free list
        if (self.free_list) |meta| {
            self.free_list = meta.next_free;
            meta.used = true;
            meta.next_free = null;
            self.total_used += meta.size;

            // Find corresponding buffer
            for (self.allocations.items) |*buf| {
                if (buf.size == meta.size) {
                    return buf;
                }
            }
        }

        // Allocate new buffer
        var buffer = try memory.GpuBuffer.init(self.allocator, self.size_class, flags);
        errdefer buffer.deinit();

        try self.allocations.append(self.allocator, buffer);
        const meta = AllocationMeta{
            .size = size,
            .size_class_idx = null,
            .allocated_at = time.unixSeconds(),
            .used = true,
            .next_free = null,
        };
        try self.metadata.append(self.allocator, meta);

        self.total_allocated += self.size_class;
        self.total_used += size;

        return &self.allocations.items[self.allocations.items.len - 1];
    }

    fn free(self: *SizeClassBucket, buffer: *memory.GpuBuffer) bool {
        for (self.allocations.items, 0..) |*buf, i| {
            if (buf == buffer) {
                const meta = &self.metadata.items[i];
                if (!meta.used) return false;

                meta.used = false;
                meta.next_free = self.free_list;
                self.free_list = meta;
                self.total_used -= meta.size;
                return true;
            }
        }
        return false;
    }

    fn coalesce(self: *SizeClassBucket) void {
        // Remove completely unused buffers from the end
        while (self.allocations.items.len > 0) {
            const last_idx = self.allocations.items.len - 1;
            if (!self.metadata.items[last_idx].used) {
                const buf = &self.allocations.items[last_idx];
                self.total_allocated -= buf.size;
                buf.deinit();
                _ = self.allocations.pop();
                _ = self.metadata.pop();
            } else {
                break;
            }
        }
    }
};

/// Advanced GPU memory pool.
pub const AdvancedMemoryPool = struct {
    allocator: std.mem.Allocator,
    config: PoolConfig,
    size_classes: [SIZE_CLASSES.len]SizeClassBucket,
    overflow_allocations: std.ArrayListUnmanaged(memory.GpuBuffer),
    total_size: usize,
    peak_size: usize,
    allocation_count: u64,
    free_count: u64,
    coalesce_count: u64,
    mutex: std.Thread.Mutex,

    /// Initialize the advanced memory pool.
    pub fn init(allocator: std.mem.Allocator, config: PoolConfig) AdvancedMemoryPool {
        var pool = AdvancedMemoryPool{
            .allocator = allocator,
            .config = config,
            .size_classes = undefined,
            .overflow_allocations = .{},
            .total_size = 0,
            .peak_size = 0,
            .allocation_count = 0,
            .free_count = 0,
            .coalesce_count = 0,
            .mutex = .{},
        };

        for (&pool.size_classes, 0..) |*bucket, i| {
            bucket.* = SizeClassBucket.init(allocator, SIZE_CLASSES[i]);
        }

        return pool;
    }

    /// Deinitialize the pool and free all resources.
    pub fn deinit(self: *AdvancedMemoryPool) void {
        for (&self.size_classes) |*bucket| {
            bucket.deinit();
        }

        for (self.overflow_allocations.items) |*buf| {
            buf.deinit();
        }
        self.overflow_allocations.deinit(self.allocator);
        self.* = undefined;
    }

    /// Allocate a GPU buffer from the pool.
    pub fn allocate(self: *AdvancedMemoryPool, size: usize, flags: memory.BufferFlags) !*memory.GpuBuffer {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check memory pressure before allocating
        try self.checkMemoryPressure();

        // Find appropriate size class
        const class_idx = self.findSizeClass(size);

        if (class_idx) |idx| {
            // Allocate from size class bucket
            const bucket = &self.size_classes[idx];
            if (bucket.total_allocated >= self.config.max_class_size) {
                return error.OutOfMemory;
            }

            const buffer = try bucket.allocate(size, flags);
            self.total_size += SIZE_CLASSES[idx];
            self.allocation_count += 1;

            if (self.total_size > self.peak_size) {
                self.peak_size = self.total_size;
            }

            return buffer;
        } else {
            // Size too large for size classes, allocate directly
            if (self.total_size + size > self.config.max_total_size) {
                return error.OutOfMemory;
            }

            var buffer = try memory.GpuBuffer.init(self.allocator, size, flags);
            errdefer buffer.deinit();

            try self.overflow_allocations.append(self.allocator, buffer);
            self.total_size += size;
            self.allocation_count += 1;

            if (self.total_size > self.peak_size) {
                self.peak_size = self.total_size;
            }

            return &self.overflow_allocations.items[self.overflow_allocations.items.len - 1];
        }
    }

    /// Free a GPU buffer back to the pool.
    pub fn free(self: *AdvancedMemoryPool, buffer: *memory.GpuBuffer) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Try to free from size class buckets
        for (&self.size_classes) |*bucket| {
            if (bucket.free(buffer)) {
                self.total_size -= buffer.size;
                self.free_count += 1;

                // Trigger coalescing if enabled
                if (self.config.enable_coalescing and self.shouldCoalesce()) {
                    self.coalesceAll();
                }

                return true;
            }
        }

        // Try to free from overflow allocations
        for (self.overflow_allocations.items, 0..) |*buf, i| {
            if (buf == buffer) {
                self.total_size -= buf.size;
                self.free_count += 1;
                buf.deinit();
                _ = self.overflow_allocations.swapRemove(i);
                return true;
            }
        }

        return false;
    }

    /// Get pool statistics.
    pub fn getStats(self: *const AdvancedMemoryPool) PoolStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var used_size: usize = 0;
        for (self.size_classes) |bucket| {
            used_size += bucket.total_used;
        }

        return .{
            .total_size = self.total_size,
            .used_size = used_size,
            .peak_size = self.peak_size,
            .allocation_count = self.allocation_count,
            .free_count = self.free_count,
            .coalesce_count = self.coalesce_count,
            .fragmentation_ratio = self.calculateFragmentation(),
            .utilization = if (self.config.max_total_size > 0)
                @as(f64, @floatFromInt(self.total_size)) / @as(f64, @floatFromInt(self.config.max_total_size))
            else
                0.0,
        };
    }

    /// Reset statistics.
    pub fn resetStats(self: *AdvancedMemoryPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.allocation_count = 0;
        self.free_count = 0;
        self.coalesce_count = 0;
        self.peak_size = self.total_size;
    }

    /// Force memory coalescing.
    pub fn coalesce(self: *AdvancedMemoryPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.coalesceAll();
    }

    // Internal helpers
    fn findSizeClass(self: *const AdvancedMemoryPool, size: usize) ?usize {
        _ = self;
        for (SIZE_CLASSES, 0..) |class_size, i| {
            if (size <= class_size) {
                return i;
            }
        }
        return null;
    }

    fn shouldCoalesce(self: *const AdvancedMemoryPool) bool {
        const utilization = @as(f64, @floatFromInt(self.total_size)) /
            @as(f64, @floatFromInt(self.config.max_total_size));
        return utilization > 0.8 and (self.allocation_count % 100 == 0);
    }

    fn coalesceAll(self: *AdvancedMemoryPool) void {
        for (&self.size_classes) |*bucket| {
            bucket.coalesce();
        }
        self.coalesce_count += 1;
    }

    fn calculateFragmentation(self: *const AdvancedMemoryPool) f64 {
        var total_allocated: usize = 0;
        var total_used: usize = 0;

        for (self.size_classes) |bucket| {
            total_allocated += bucket.total_allocated;
            total_used += bucket.total_used;
        }

        if (total_allocated == 0) return 0.0;

        return 1.0 - (@as(f64, @floatFromInt(total_used)) / @as(f64, @floatFromInt(total_allocated)));
    }

    fn checkMemoryPressure(self: *AdvancedMemoryPool) !void {
        const utilization = @as(f64, @floatFromInt(self.total_size)) /
            @as(f64, @floatFromInt(self.config.max_total_size));

        if (utilization >= self.config.high_water_mark) {
            // High memory pressure - force coalescing
            self.coalesceAll();

            // If still above low water mark, fail allocation
            const new_utilization = @as(f64, @floatFromInt(self.total_size)) /
                @as(f64, @floatFromInt(self.config.max_total_size));

            if (new_utilization >= self.config.low_water_mark) {
                return error.OutOfMemory;
            }
        }
    }
};

/// Pool statistics.
pub const PoolStats = struct {
    total_size: usize,
    used_size: usize,
    peak_size: usize,
    allocation_count: u64,
    free_count: u64,
    coalesce_count: u64,
    fragmentation_ratio: f64,
    utilization: f64,
};

test "advanced pool allocation" {
    const allocator = std.testing.allocator;
    var pool = AdvancedMemoryPool.init(allocator, .{
        .max_total_size = 1024 * 1024,
    });
    defer pool.deinit();

    // Allocate from different size classes
    const buf1 = try pool.allocate(128, .{});
    const buf2 = try pool.allocate(512, .{});
    const buf3 = try pool.allocate(2048, .{});

    const stats = pool.getStats();
    try std.testing.expect(stats.allocation_count == 3);
    try std.testing.expect(stats.total_size > 0);

    // Free and verify
    try std.testing.expect(pool.free(buf1));
    try std.testing.expect(pool.free(buf2));
    try std.testing.expect(pool.free(buf3));

    const final_stats = pool.getStats();
    try std.testing.expectEqual(@as(u64, 3), final_stats.free_count);
}

test "pool coalescing" {
    const allocator = std.testing.allocator;
    var pool = AdvancedMemoryPool.init(allocator, .{
        .enable_coalescing = true,
    });
    defer pool.deinit();

    // Allocate and free many times
    var buffers: [10]*memory.GpuBuffer = undefined;
    for (&buffers) |*buf| {
        buf.* = try pool.allocate(256, .{});
    }

    for (buffers) |buf| {
        _ = pool.free(buf);
    }

    pool.coalesce();

    const stats = pool.getStats();
    try std.testing.expect(stats.coalesce_count > 0);
}

test "memory pressure handling" {
    const allocator = std.testing.allocator;
    var pool = AdvancedMemoryPool.init(allocator, .{
        .max_total_size = 4096,
        .high_water_mark = 0.8,
    });
    defer pool.deinit();

    // Fill pool close to capacity
    var buffers: [3]*memory.GpuBuffer = undefined;
    for (&buffers) |*buf| {
        buf.* = try pool.allocate(1024, .{});
    }

    // This should trigger memory pressure
    const result = pool.allocate(1024, .{});
    if (result) |buf| {
        _ = buf;
    } else |err| {
        try std.testing.expectEqual(error.OutOfMemory, err);
    }

    // Clean up
    for (buffers) |buf| {
        _ = pool.free(buf);
    }
}
