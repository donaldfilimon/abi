//! Cache-Aligned Storage and Vector Pool
//!
//! Provides cache-optimized data structures for vector storage:
//! - HotVectorData: Cache-aligned contiguous storage for vectors and norms
//! - ColdVectorData: Infrequently accessed metadata storage
//! - VectorPool: Size-class based memory pooling for vectors

const std = @import("std");

/// Cache line size for alignment (64 bytes on most modern CPUs)
pub const CACHE_LINE_SIZE = 64;

/// Hot data structure optimized for cache access patterns.
/// Separates frequently-accessed data (vectors, norms) from cold data (metadata).
pub const HotVectorData = struct {
    /// Contiguous vector storage - cache-aligned
    vectors: []align(CACHE_LINE_SIZE) f32,
    /// Parallel array of norms - cache-aligned
    norms: []align(CACHE_LINE_SIZE) f32,
    /// Number of vectors stored
    count: usize,
    /// Dimension of each vector
    dimension: usize,
    /// Total capacity
    capacity: usize,

    pub fn init(allocator: std.mem.Allocator, dimension: usize, capacity: usize) !HotVectorData {
        const vectors = try allocator.alignedAlloc(f32, CACHE_LINE_SIZE, capacity * dimension);
        const norms = try allocator.alignedAlloc(f32, CACHE_LINE_SIZE, capacity);
        return .{
            .vectors = vectors,
            .norms = norms,
            .count = 0,
            .dimension = dimension,
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *HotVectorData, allocator: std.mem.Allocator) void {
        allocator.free(self.vectors);
        allocator.free(self.norms);
    }

    /// Get vector at index as a slice
    pub fn getVector(self: *const HotVectorData, index: usize) []const f32 {
        const start = index * self.dimension;
        return self.vectors[start..][0..self.dimension];
    }

    /// Get mutable vector at index
    pub fn getVectorMut(self: *HotVectorData, index: usize) []f32 {
        const start = index * self.dimension;
        return self.vectors[start..][0..self.dimension];
    }

    /// Get norm at index
    pub fn getNorm(self: *const HotVectorData, index: usize) f32 {
        return self.norms[index];
    }

    /// Add a vector and its norm
    pub fn append(self: *HotVectorData, vector: []const f32, norm: f32) !void {
        if (self.count >= self.capacity) return error.PoolExhausted;
        const dest = self.getVectorMut(self.count);
        @memcpy(dest, vector);
        self.norms[self.count] = norm;
        self.count += 1;
    }

    /// Prefetch vector at index for upcoming access
    pub fn prefetch(self: *const HotVectorData, index: usize) void {
        if (index < self.count) {
            const start = index * self.dimension;
            const ptr: [*]const f32 = @ptrCast(&self.vectors[start]);
            @prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data });
        }
    }
};

/// Cold data structure for infrequently accessed data.
pub const ColdVectorData = struct {
    /// Vector IDs
    ids: std.ArrayListUnmanaged(u64),
    /// Metadata (optional per vector)
    metadata: std.ArrayListUnmanaged(?[]const u8),

    pub fn init() ColdVectorData {
        return .{
            .ids = .{},
            .metadata = .{},
        };
    }

    pub fn deinit(self: *ColdVectorData, allocator: std.mem.Allocator) void {
        for (self.metadata.items) |meta| {
            if (meta) |m| allocator.free(m);
        }
        self.ids.deinit(allocator);
        self.metadata.deinit(allocator);
    }

    pub fn append(self: *ColdVectorData, allocator: std.mem.Allocator, id: u64, metadata: ?[]const u8) !void {
        try self.ids.append(allocator, id);
        const meta_copy: ?[]const u8 = if (metadata) |m| try allocator.dupe(u8, m) else null;
        try self.metadata.append(allocator, meta_copy);
    }
};

// ============================================================================
// Vector Pool - Allocator wrapper for vector memory
// ============================================================================

/// Vector allocator wrapper.
/// Currently a thin wrapper around the allocator. Future versions may implement
/// size-class based pooling for common dimensions (128, 256, 384, 512, 768, 1024, 1536, 4096).
///
/// Usage is optional - Database works with or without VectorPool.
pub const VectorPool = struct {
    allocator: std.mem.Allocator,
    /// Statistics for monitoring
    alloc_count: usize,
    free_count: usize,
    total_bytes: usize,

    pub fn init(allocator: std.mem.Allocator) VectorPool {
        return .{
            .allocator = allocator,
            .alloc_count = 0,
            .free_count = 0,
            .total_bytes = 0,
        };
    }

    pub fn deinit(self: *VectorPool) void {
        // No cleanup needed - allocator owns all memory
        self.* = undefined;
    }

    /// Allocate a vector of the given dimension.
    pub fn alloc(self: *VectorPool, dimension: usize) ![]f32 {
        const vec = try self.allocator.alloc(f32, dimension);
        self.alloc_count += 1;
        self.total_bytes += dimension * @sizeOf(f32);
        return vec;
    }

    /// Free a vector.
    pub fn free(self: *VectorPool, vector: []f32) void {
        self.free_count += 1;
        self.total_bytes -|= vector.len * @sizeOf(f32);
        self.allocator.free(vector);
    }

    /// Get pool statistics.
    pub fn getStats(self: *const VectorPool) PoolStats {
        return .{
            .alloc_count = self.alloc_count,
            .free_count = self.free_count,
            .active_count = self.alloc_count -| self.free_count,
            .total_bytes = self.total_bytes,
        };
    }

    pub const PoolStats = struct {
        alloc_count: usize,
        free_count: usize,
        active_count: usize,
        total_bytes: usize,
    };
};
