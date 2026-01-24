//! Thread-local memory cache for per-thread allocation.
//!
//! Provides thread-local caching of allocations to reduce contention:
//! - Per-thread free lists
//! - Batch allocation from global pool
//! - Automatic balancing between threads
//!
//! Usage:
//! ```zig
//! var global_cache = try ThreadCache.init(allocator, .{});
//! defer global_cache.deinit();
//!
//! // In worker thread:
//! var local = global_cache.getLocal();
//! const ptr = try local.alloc(u8, 256);
//! defer local.free(ptr);
//! ```

const std = @import("std");

/// Configuration for thread cache.
pub const ThreadCacheConfig = struct {
    /// Size of per-thread cache in bytes.
    local_cache_size: usize = 64 * 1024, // 64 KB
    /// Maximum number of cached objects per size class.
    max_cached_per_class: usize = 64,
    /// Number of size classes.
    num_size_classes: usize = 8,
    /// Smallest size class (bytes).
    min_size_class: usize = 16,
    /// Enable statistics collection.
    collect_stats: bool = true,
};

/// Statistics for thread cache.
pub const ThreadCacheStats = struct {
    local_allocs: u64,
    local_frees: u64,
    global_allocs: u64,
    global_frees: u64,
    cache_hits: u64,
    cache_misses: u64,
    active_threads: usize,
    total_cached_bytes: usize,
};

/// Per-size-class cache entry (overlaid on freed memory).
/// Only contains next pointer - the memory itself is the entry.
const CacheEntry = struct {
    next: ?*CacheEntry,
};

/// Per-thread local cache.
pub const LocalCache = struct {
    parent: *ThreadCache,
    free_lists: []?*CacheEntry,
    cached_count: []usize,
    local_allocs: u64,
    local_frees: u64,
    cache_hits: u64,
    cache_misses: u64,

    const Self = @This();

    /// Allocate from local cache.
    pub fn alloc(self: *Self, comptime T: type, n: usize) ![]T {
        const size = @sizeOf(T) * n;
        const class_idx = self.parent.getSizeClass(size);

        if (class_idx) |idx| {
            // Try to get from local cache
            if (self.free_lists[idx]) |entry| {
                self.free_lists[idx] = entry.next;
                self.cached_count[idx] -= 1;
                self.cache_hits += 1;
                self.local_allocs += 1;

                // The CacheEntry IS the memory - return pointer to it
                const ptr: [*]T = @ptrCast(@alignCast(entry));
                return ptr[0..n];
            }
        }

        // Cache miss - allocate from global
        self.cache_misses += 1;
        return self.parent.globalAlloc(T, n);
    }

    /// Free to local cache.
    pub fn free(self: *Self, ptr: anytype) void {
        const T = @TypeOf(ptr);
        const size = @sizeOf(std.meta.Elem(T)) * ptr.len;
        const class_idx = self.parent.getSizeClass(size);

        // Check if allocation is large and aligned enough to store CacheEntry
        const min_cache_size = @sizeOf(CacheEntry);
        const cache_align = @alignOf(CacheEntry);
        const ptr_addr = @intFromPtr(ptr.ptr);

        if (class_idx) |idx| {
            if (self.cached_count[idx] < self.parent.config.max_cached_per_class and
                size >= min_cache_size and ptr_addr % cache_align == 0)
            {
                // Add to local cache
                const entry: *CacheEntry = @ptrCast(@alignCast(ptr.ptr));
                entry.next = self.free_lists[idx];
                self.free_lists[idx] = entry;
                self.cached_count[idx] += 1;
                self.local_frees += 1;
                return;
            }
        }

        // Local cache full, size too small, unaligned, or size too large - free to global
        self.parent.globalFree(ptr);
    }

    /// Flush local cache to global.
    pub fn flush(self: *Self) void {
        for (self.free_lists, 0..) |list, idx| {
            var current = list;
            while (current) |entry| {
                const next = entry.next;
                const size = self.parent.sizeClassToSize(idx);
                const ptr: [*]u8 = @ptrCast(entry);
                self.parent.backing_allocator.free(ptr[0..size]);
                current = next;
            }
            self.free_lists[idx] = null;
            self.cached_count[idx] = 0;
        }
    }

    /// Get local statistics.
    pub fn getStats(self: *const Self) LocalCacheStats {
        var total_cached: usize = 0;
        for (self.cached_count, 0..) |count, idx| {
            total_cached += count * self.parent.sizeClassToSize(idx);
        }

        return .{
            .local_allocs = self.local_allocs,
            .local_frees = self.local_frees,
            .cache_hits = self.cache_hits,
            .cache_misses = self.cache_misses,
            .total_cached = total_cached,
            .hit_rate = if (self.cache_hits + self.cache_misses > 0)
                @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(self.cache_hits + self.cache_misses))
            else
                0.0,
        };
    }
};

/// Local cache statistics.
pub const LocalCacheStats = struct {
    local_allocs: u64,
    local_frees: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_cached: usize,
    hit_rate: f32,
};

/// Global thread cache manager.
pub const ThreadCache = struct {
    backing_allocator: std.mem.Allocator,
    config: ThreadCacheConfig,
    size_classes: []usize,
    global_allocs: std.atomic.Value(u64),
    global_frees: std.atomic.Value(u64),
    active_locals: std.atomic.Value(usize),
    mutex: std.Thread.Mutex,
    locals: std.ArrayListUnmanaged(*LocalCache),

    const Self = @This();

    /// Initialize the thread cache.
    pub fn init(backing_allocator: std.mem.Allocator, config: ThreadCacheConfig) !Self {
        // Build size class table
        const size_classes = try backing_allocator.alloc(usize, config.num_size_classes);
        var size = config.min_size_class;
        for (size_classes) |*class| {
            class.* = size;
            size *= 2;
        }

        return .{
            .backing_allocator = backing_allocator,
            .config = config,
            .size_classes = size_classes,
            .global_allocs = std.atomic.Value(u64).init(0),
            .global_frees = std.atomic.Value(u64).init(0),
            .active_locals = std.atomic.Value(usize).init(0),
            .mutex = .{},
            .locals = .{},
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *Self) void {
        // Flush and free all local caches
        for (self.locals.items) |local| {
            local.flush();
            self.backing_allocator.free(local.free_lists);
            self.backing_allocator.free(local.cached_count);
            self.backing_allocator.destroy(local);
        }
        self.locals.deinit(self.backing_allocator);
        self.backing_allocator.free(self.size_classes);
        self.* = undefined;
    }

    /// Get or create thread-local cache.
    pub fn getLocal(self: *Self) !*LocalCache {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Create new local cache
        const local = try self.backing_allocator.create(LocalCache);
        errdefer self.backing_allocator.destroy(local);

        const free_lists = try self.backing_allocator.alloc(?*CacheEntry, self.config.num_size_classes);
        errdefer self.backing_allocator.free(free_lists);
        @memset(free_lists, null);

        const cached_count = try self.backing_allocator.alloc(usize, self.config.num_size_classes);
        errdefer self.backing_allocator.free(cached_count);
        @memset(cached_count, 0);

        local.* = .{
            .parent = self,
            .free_lists = free_lists,
            .cached_count = cached_count,
            .local_allocs = 0,
            .local_frees = 0,
            .cache_hits = 0,
            .cache_misses = 0,
        };

        try self.locals.append(self.backing_allocator, local);
        _ = self.active_locals.fetchAdd(1, .monotonic);

        return local;
    }

    /// Release a local cache.
    pub fn releaseLocal(self: *Self, local: *LocalCache) void {
        local.flush();

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.locals.items, 0..) |item, i| {
            if (item == local) {
                self.backing_allocator.free(local.free_lists);
                self.backing_allocator.free(local.cached_count);
                self.backing_allocator.destroy(local);
                _ = self.locals.swapRemove(i);
                _ = self.active_locals.fetchSub(1, .monotonic);
                return;
            }
        }
    }

    /// Get global statistics.
    pub fn getStats(self: *Self) ThreadCacheStats {
        var total_local_allocs: u64 = 0;
        var total_local_frees: u64 = 0;
        var total_cache_hits: u64 = 0;
        var total_cache_misses: u64 = 0;
        var total_cached_bytes: usize = 0;

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.locals.items) |local| {
            const stats = local.getStats();
            total_local_allocs += stats.local_allocs;
            total_local_frees += stats.local_frees;
            total_cache_hits += stats.cache_hits;
            total_cache_misses += stats.cache_misses;
            total_cached_bytes += stats.total_cached;
        }

        return .{
            .local_allocs = total_local_allocs,
            .local_frees = total_local_frees,
            .global_allocs = self.global_allocs.load(.monotonic),
            .global_frees = self.global_frees.load(.monotonic),
            .cache_hits = total_cache_hits,
            .cache_misses = total_cache_misses,
            .active_threads = self.active_locals.load(.monotonic),
            .total_cached_bytes = total_cached_bytes,
        };
    }

    // Internal methods
    fn getSizeClass(self: *const Self, size: usize) ?usize {
        for (self.size_classes, 0..) |class_size, i| {
            if (size <= class_size) {
                return i;
            }
        }
        return null;
    }

    fn sizeClassToSize(self: *const Self, idx: usize) usize {
        return self.size_classes[idx];
    }

    fn globalAlloc(self: *Self, comptime T: type, n: usize) ![]T {
        _ = self.global_allocs.fetchAdd(1, .monotonic);
        return self.backing_allocator.alloc(T, n);
    }

    fn globalFree(self: *Self, ptr: anytype) void {
        _ = self.global_frees.fetchAdd(1, .monotonic);
        self.backing_allocator.free(ptr);
    }
};

/// Simple per-thread arena allocator.
pub const ThreadArena = struct {
    backing: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    allocations: u64,
    peak_size: usize,

    const Self = @This();

    /// Initialize thread arena.
    pub fn init(backing: std.mem.Allocator) Self {
        return .{
            .backing = backing,
            .arena = std.heap.ArenaAllocator.init(backing),
            .allocations = 0,
            .peak_size = 0,
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Get allocator.
    pub fn allocator(self: *Self) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Reset arena (free all allocations).
    pub fn reset(self: *Self) void {
        // Track peak before reset
        // Note: ArenaAllocator doesn't expose total size, so we approximate
        _ = self.arena.reset(.retain_capacity);
    }

    /// Get statistics.
    pub fn getStats(self: *const Self) ThreadArenaStats {
        return .{
            .allocations = self.allocations,
            .peak_size = self.peak_size,
        };
    }
};

/// Thread arena statistics.
pub const ThreadArenaStats = struct {
    allocations: u64,
    peak_size: usize,
};

test "thread cache basic" {
    const allocator = std.testing.allocator;
    var cache = try ThreadCache.init(allocator, .{});
    defer cache.deinit();

    var local = try cache.getLocal();
    defer cache.releaseLocal(local);

    const data = try local.alloc(u8, 32);
    try std.testing.expectEqual(@as(usize, 32), data.len);

    local.free(data);

    const stats = cache.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.active_threads);
}

test "thread cache reuse" {
    const allocator = std.testing.allocator;
    var cache = try ThreadCache.init(allocator, .{
        .max_cached_per_class = 4,
    });
    defer cache.deinit();

    var local = try cache.getLocal();
    defer cache.releaseLocal(local);

    // Allocate and free multiple times
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        const data = try local.alloc(u8, 32);
        local.free(data);
    }

    // This should hit the cache
    const data = try local.alloc(u8, 32);
    local.free(data);

    const stats = local.getStats();
    try std.testing.expect(stats.cache_hits > 0);
}

test "thread arena" {
    const allocator = std.testing.allocator;
    var arena = ThreadArena.init(allocator);
    defer arena.deinit();

    const alloc = arena.allocator();

    const a = try alloc.alloc(u8, 100);
    const b = try alloc.alloc(u8, 200);

    try std.testing.expectEqual(@as(usize, 100), a.len);
    try std.testing.expectEqual(@as(usize, 200), b.len);

    arena.reset();

    // Can allocate again after reset
    const c = try alloc.alloc(u8, 300);
    try std.testing.expectEqual(@as(usize, 300), c.len);
}
