//! GPU Memory Pool for Efficient Memory Management
//!
//! This module provides a memory pool system for GPU resources to:
//! - Reduce memory allocation overhead
//! - Minimize GPU memory fragmentation
//! - Provide automatic memory reuse
//! - Track memory usage and statistics
//! - Support different buffer types and sizes

const std = @import("std");
const gpu_renderer = @import("../core/gpu_renderer.zig");

/// GPU Memory Pool Configuration
pub const MemoryPoolConfig = struct {
    /// Maximum number of buffers to keep in free pool
    max_free_buffers: usize = 100,
    /// Maximum age of free buffers before cleanup (milliseconds)
    max_buffer_age_ms: i64 = 30000, // 30 seconds
    /// Minimum buffer size to pool (bytes)
    min_pool_size: usize = 1024, // 1KB
    /// Maximum buffer size to pool (bytes)
    max_pool_size: usize = 128 * 1024 * 1024, // 128MB
    /// Enable memory defragmentation
    enable_defragmentation: bool = true,
    /// Memory cleanup interval (milliseconds)
    cleanup_interval_ms: i64 = 10000, // 10 seconds
};

/// Memory pool statistics
pub const MemoryStats = struct {
    total_buffers_allocated: usize = 0,
    total_buffers_free: usize = 0,
    total_memory_allocated: usize = 0,
    total_memory_free: usize = 0,
    peak_memory_usage: usize = 0,
    average_buffer_size: usize = 0,
    fragmentation_ratio: f32 = 0.0,
    cache_hit_rate: f32 = 0.0,
    last_cleanup_time: i64 = 0,
    total_allocations: usize = 0,
    total_frees: usize = 0,
    total_cache_hits: usize = 0,
    total_cache_misses: usize = 0,
};

/// Buffer metadata for pool management
pub const BufferMetadata = struct {
    handle: u32,
    size: usize,
    usage: gpu_renderer.BufferUsage,
    allocation_time: i64,
    last_access_time: i64,
    access_count: usize,
    generation: u32, // For LRU tracking
    is_pooled: bool,
};

/// GPU Memory Pool Manager
pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,
    config: MemoryPoolConfig,

    // Buffer tracking
    allocated_buffers: std.AutoHashMap(u32, BufferMetadata),
    free_buffers: std.ArrayList(BufferMetadata),

    // Size-based organization for faster lookup
    size_buckets: std.AutoHashMap(usize, std.ArrayList(BufferMetadata)),

    // Statistics
    stats: MemoryStats,

    // LRU tracking
    lru_generation: u32 = 0,

    // Cleanup timer
    last_cleanup: i64,

    pub fn init(
        allocator: std.mem.Allocator,
        renderer: *gpu_renderer.GPURenderer,
        config: MemoryPoolConfig,
    ) !*MemoryPool {
        const self = try allocator.create(MemoryPool);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .renderer = renderer,
            .config = config,
            .allocated_buffers = std.AutoHashMap(u32, BufferMetadata).init(allocator),
            .free_buffers = try std.ArrayList(BufferMetadata).initCapacity(allocator, 0),
            .size_buckets = std.AutoHashMap(usize, std.ArrayList(BufferMetadata)).init(allocator),
            .stats = .{},
            .last_cleanup = std.time.milliTimestamp(),
        };

        return self;
    }

    pub fn deinit(self: *MemoryPool) void {
        // Clean up all allocated buffers
        var it = self.allocated_buffers.iterator();
        while (it.next()) |entry| {
            self.renderer.destroyBuffer(entry.key_ptr.*) catch {};
        }

        // Clean up free buffers
        for (self.free_buffers.items) |buffer| {
            self.renderer.destroyBuffer(buffer.handle) catch {};
        }

        // Clean up size buckets
        var bucket_it = self.size_buckets.iterator();
        while (bucket_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }

        self.allocated_buffers.deinit();
        self.free_buffers.deinit(self.allocator);
        self.size_buckets.deinit();
        self.allocator.destroy(self);
    }

    /// Allocate a buffer from the pool or create a new one
    pub fn allocBuffer(
        self: *MemoryPool,
        size: usize,
        usage: gpu_renderer.BufferUsage,
    ) !u32 {
        self.stats.total_allocations += 1;

        // Check if size is within pooling range
        if (size < self.config.min_pool_size or size > self.config.max_pool_size) {
            return self.createNewBuffer(size, usage);
        }

        // Try to find a suitable buffer in the free pool
        if (self.findFreeBuffer(size, usage)) |handle| {
            self.stats.total_cache_hits += 1;
            return handle;
        }

        // No suitable buffer found, create new one
        self.stats.total_cache_misses += 1;
        return self.createNewBuffer(size, usage);
    }

    /// Create a new buffer and track it
    fn createNewBuffer(self: *MemoryPool, size: usize, usage: gpu_renderer.BufferUsage) !u32 {
        const handle = try self.renderer.createBuffer(size, usage);

        const metadata = BufferMetadata{
            .handle = handle,
            .size = size,
            .usage = usage,
            .allocation_time = std.time.milliTimestamp(),
            .last_access_time = std.time.milliTimestamp(),
            .access_count = 1,
            .generation = self.lru_generation,
            .is_pooled = (size >= self.config.min_pool_size and size <= self.config.max_pool_size),
        };

        try self.allocated_buffers.put(handle, metadata);

        // Update statistics
        self.stats.total_buffers_allocated += 1;
        self.stats.total_memory_allocated += size;
        self.stats.peak_memory_usage = @max(self.stats.peak_memory_usage, self.stats.total_memory_allocated);

        return handle;
    }

    /// Find a suitable free buffer for reuse
    fn findFreeBuffer(self: *MemoryPool, size: usize, usage: gpu_renderer.BufferUsage) ?u32 {
        // First, try exact size match
        if (self.size_buckets.getPtr(size)) |bucket| {
            for (bucket.items, 0..) |*buffer, i| {
                if (self.bufferUsageCompatible(buffer.usage, usage)) {
                    const handle = buffer.handle;
                    _ = bucket.swapRemove(i);
                    return handle;
                }
            }
        }

        // Then try larger buffers (with some waste tolerance)
        var it = self.size_buckets.iterator();
        while (it.next()) |entry| {
            const bucket_size = entry.key_ptr.*;
            if (bucket_size >= size and bucket_size <= size * 2) { // Allow up to 2x waste
                const bucket = entry.value_ptr;
                for (bucket.items, 0..) |*buffer, i| {
                    if (self.bufferUsageCompatible(buffer.usage, usage)) {
                        const handle = buffer.handle;
                        _ = bucket.swapRemove(i);
                        return handle;
                    }
                }
            }
        }

        return null;
    }

    /// Check if buffer usage is compatible for reuse
    fn bufferUsageCompatible(self: *MemoryPool, existing: gpu_renderer.BufferUsage, requested: gpu_renderer.BufferUsage) bool {
        _ = self; // Not used in this implementation

        // Storage buffers can be reused for storage operations
        if (existing.storage and requested.storage) return true;

        // Copy source/destination compatibility
        if (existing.copy_src == requested.copy_src and existing.copy_dst == requested.copy_dst) {
            return true;
        }

        return false;
    }

    /// Return a buffer to the pool for reuse
    pub fn freeBuffer(self: *MemoryPool, handle: u32) !void {
        self.stats.total_frees += 1;

        // Find the buffer in allocated list
        if (self.allocated_buffers.fetchRemove(handle)) |kv| {
            var metadata = kv.value;
            metadata.last_access_time = std.time.milliTimestamp();

            // Only pool buffers within size limits
            if (metadata.is_pooled and self.stats.total_buffers_free < self.config.max_free_buffers) {
                try self.free_buffers.append(self.allocator, metadata);

                // Add to size bucket for faster lookup
                const bucket = try self.size_buckets.getOrPut(metadata.size);
                if (!bucket.found_existing) {
                    bucket.value_ptr.* = try std.ArrayList(BufferMetadata).initCapacity(self.allocator, 0);
                }
                try bucket.value_ptr.append(self.allocator, metadata);

                self.stats.total_buffers_free += 1;
                self.stats.total_memory_free += metadata.size;
            } else {
                // Destroy buffer if not pooling or pool is full
                self.renderer.destroyBuffer(handle) catch {};
                self.stats.total_memory_allocated -= metadata.size;
            }
        }
    }

    /// Periodic cleanup of old buffers
    pub fn cleanup(self: *MemoryPool) !void {
        const current_time = std.time.milliTimestamp();

        if (current_time - self.last_cleanup < self.config.cleanup_interval_ms) {
            return;
        }

        self.last_cleanup = current_time;
        self.stats.last_cleanup_time = current_time;

        // Remove old buffers from free pool
        var i = self.free_buffers.items.len;
        while (i > 0) {
            i -= 1;
            const buffer = &self.free_buffers.items[i];
            if (current_time - buffer.last_access_time > self.config.max_buffer_age_ms) {
                self.renderer.destroyBuffer(buffer.handle) catch {};
                _ = self.free_buffers.swapRemove(i);
                self.stats.total_buffers_free -= 1;
                self.stats.total_memory_free -= buffer.size;
            }
        }

        // Clean up empty size buckets
        var buckets_to_remove = try std.ArrayList(usize).initCapacity(self.allocator, 0);
        defer buckets_to_remove.deinit(self.allocator);

        var it = self.size_buckets.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.items.len == 0) {
                try buckets_to_remove.append(self.allocator, entry.key_ptr.*);
            }
        }

        for (buckets_to_remove.items) |size| {
            if (self.size_buckets.fetchRemove(size)) |kv| {
                @constCast(&kv.value).deinit(self.allocator);
            }
        }
    }

    /// Get comprehensive memory statistics
    pub fn getStats(self: *MemoryPool) MemoryStats {
        var stats = self.stats;

        // Calculate additional metrics
        const total_buffers = stats.total_buffers_allocated + stats.total_buffers_free;
        if (total_buffers > 0) {
            stats.average_buffer_size = (stats.total_memory_allocated + stats.total_memory_free) / total_buffers;
        }

        // Calculate cache hit rate
        const total_requests = stats.total_allocations;
        if (total_requests > 0) {
            stats.cache_hit_rate = @as(f32, @floatFromInt(stats.total_cache_hits)) / @as(f32, @floatFromInt(total_requests));
        }

        // Estimate fragmentation (simplified)
        if (stats.total_memory_allocated > 0) {
            stats.fragmentation_ratio = 1.0 - (@as(f32, @floatFromInt(stats.total_memory_free)) / @as(f32, @floatFromInt(stats.total_memory_allocated)));
        }

        return stats;
    }

    /// Defragment memory pool (advanced feature)
    pub fn defragment(self: *MemoryPool) !void {
        if (!self.config.enable_defragmentation) return;

        // This is a simplified defragmentation strategy
        // In a real implementation, this would involve:
        // 1. Analyzing free buffer patterns
        // 2. Coalescing adjacent free buffers
        // 3. Relocating buffers to reduce fragmentation

        std.log.info("Running memory pool defragmentation", .{});

        // Sort free buffers by size for better reuse patterns
        std.mem.sort(BufferMetadata, self.free_buffers.items, {}, struct {
            fn lessThan(_: void, a: BufferMetadata, b: BufferMetadata) bool {
                return a.size < b.size;
            }
        }.lessThan);

        // Clean up very small buffers that cause fragmentation
        var i = self.free_buffers.items.len;
        while (i > 0) {
            i -= 1;
            const buffer = &self.free_buffers.items[i];
            if (buffer.size < self.config.min_pool_size / 4) {
                self.renderer.destroyBuffer(buffer.handle) catch {};
                _ = self.free_buffers.swapRemove(i);
                self.stats.total_buffers_free -= 1;
                self.stats.total_memory_free -= buffer.size;
            }
        }
    }

    /// Prefetch buffers for anticipated usage patterns
    pub fn prefetchBuffers(self: *MemoryPool, sizes: []const usize, usage: gpu_renderer.BufferUsage) !void {
        for (sizes) |size| {
            if (size >= self.config.min_pool_size and size <= self.config.max_pool_size) {
                const handle = try self.createNewBuffer(size, usage);
                try self.freeBuffer(handle); // Immediately return to pool
            }
        }
    }

    /// Resize a buffer while maintaining pool efficiency
    pub fn resizeBuffer(self: *MemoryPool, old_handle: u32, new_size: usize) !u32 {
        // Get metadata for old buffer
        if (self.allocated_buffers.get(old_handle)) |old_metadata| {
            // Create new buffer with requested size
            const new_handle = try self.createNewBuffer(new_size, old_metadata.usage);

            // Copy data from old buffer to new buffer (if needed)
            // This is a simplified implementation - real version would use GPU copy operations

            // Free old buffer
            try self.freeBuffer(old_handle);

            return new_handle;
        }

        return gpu_renderer.GpuError.HandleNotFound;
    }

    /// Get memory usage report
    pub fn getMemoryReport(self: *MemoryPool, allocator: std.mem.Allocator) ![]const u8 {
        const stats = self.getStats();

        var report = std.ArrayList(u8).init(allocator);
        errdefer report.deinit();

        try std.fmt.format(report.writer(),
            \\GPU Memory Pool Report
            \\======================
            \\Total Buffers Allocated: {}
            \\Total Buffers Free: {}
            \\Total Memory Allocated: {} MB
            \\Total Memory Free: {} MB
            \\Peak Memory Usage: {} MB
            \\Average Buffer Size: {} KB
            \\Cache Hit Rate: {d:.2}%
            \\Fragmentation Ratio: {d:.2}%
            \\Total Allocations: {}
            \\Total Frees: {}
            \\Last Cleanup: {}ms ago
            \\
        , .{
            stats.total_buffers_allocated,
            stats.total_buffers_free,
            stats.total_memory_allocated / (1024 * 1024),
            stats.total_memory_free / (1024 * 1024),
            stats.peak_memory_usage / (1024 * 1024),
            stats.average_buffer_size / 1024,
            stats.cache_hit_rate * 100.0,
            stats.fragmentation_ratio,
            stats.total_allocations,
            stats.total_frees,
            std.time.milliTimestamp() - stats.last_cleanup_time,
        });

        return report.toOwnedSlice();
    }
};
