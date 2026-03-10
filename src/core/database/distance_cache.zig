//! Hash-based distance cache for HNSW vector index.
//! Provides O(1) lookups of previously computed similarity distances with FIFO eviction.

const std = @import("std");

// ============================================================================
// Distance Cache - Hash-based cache for frequently computed distances
// ============================================================================

/// Hash-based distance cache for O(1) lookups of similarity computations.
/// Uses packed u64 keys combining two u32 node IDs with FIFO eviction.
pub const DistanceCache = struct {
    /// Hash map for O(1) lookups
    map: std.AutoHashMapUnmanaged(u64, f32),
    /// FIFO queue for eviction order
    eviction_queue: []u64,
    /// Current position in eviction queue
    queue_head: usize,
    /// Number of items in cache
    size: usize,
    /// Maximum capacity
    capacity: usize,
    /// Hit counter for statistics
    hits: u64,
    /// Miss counter for statistics
    misses: u64,
    /// Allocator for internal structures
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !DistanceCache {
        const cap = @max(capacity, 16);
        const eviction_queue = try allocator.alloc(u64, cap);
        @memset(eviction_queue, 0);

        return .{
            .map = .empty,
            .eviction_queue = eviction_queue,
            .queue_head = 0,
            .size = 0,
            .capacity = cap,
            .hits = 0,
            .misses = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DistanceCache, allocator: std.mem.Allocator) void {
        self.map.deinit(allocator);
        allocator.free(self.eviction_queue);
    }

    /// Create a cache key from two node IDs (order-independent).
    inline fn makeKey(a: u32, b: u32) u64 {
        const lo = @min(a, b);
        const hi = @max(a, b);
        return (@as(u64, hi) << 32) | @as(u64, lo);
    }

    /// Get cached distance between two nodes. O(1) average case.
    pub fn get(self: *DistanceCache, a: u32, b: u32) ?f32 {
        const key = makeKey(a, b);

        if (self.map.get(key)) |distance| {
            self.hits += 1;
            return distance;
        }
        self.misses += 1;
        return null;
    }

    /// Store distance in cache with FIFO eviction. O(1) average case.
    pub fn put(self: *DistanceCache, a: u32, b: u32, distance: f32) void {
        const key = makeKey(a, b);

        // Check if key already exists
        if (self.map.contains(key)) {
            // Update existing entry
            self.map.put(self.allocator, key, distance) catch return;
            return;
        }

        // Evict if at capacity
        if (self.size >= self.capacity) {
            const evict_key = self.eviction_queue[self.queue_head];
            _ = self.map.remove(evict_key);
            self.size -= 1;
        }

        // Insert new entry
        self.map.put(self.allocator, key, distance) catch return;
        self.eviction_queue[self.queue_head] = key;
        self.queue_head = (self.queue_head + 1) % self.capacity;
        self.size += 1;
    }

    /// Clear all cached entries while retaining allocated memory.
    /// Hit/miss statistics are not reset by this operation.
    pub fn clear(self: *DistanceCache) void {
        self.map.clearRetainingCapacity();
        @memset(self.eviction_queue, 0);
        self.size = 0;
        self.queue_head = 0;
    }

    /// Get cache statistics.
    pub fn getStats(self: *const DistanceCache) struct { hits: u64, misses: u64, hit_rate: f32 } {
        const total = self.hits + self.misses;
        const hit_rate = if (total > 0) @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total)) else 0.0;
        return .{ .hits = self.hits, .misses = self.misses, .hit_rate = hit_rate };
    }
};

test {
    std.testing.refAllDecls(@This());
}
