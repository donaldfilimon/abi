//! Embedding cache for avoiding redundant computation.
//!
//! Provides LRU-based caching with configurable size limits
//! and automatic eviction of least recently used entries.

const std = @import("std");

/// Cache configuration.
pub const CacheConfig = struct {
    /// Maximum number of entries in cache.
    max_entries: usize = 10000,
    /// Embedding dimension (for memory estimation).
    dimension: u32 = 384,
    /// Enable statistics collection.
    collect_stats: bool = true,
};

/// Cache statistics.
pub const CacheStats = struct {
    /// Total number of cache lookups.
    total_lookups: u64,
    /// Number of cache hits.
    hits: u64,
    /// Number of cache misses.
    misses: u64,
    /// Current number of entries.
    current_entries: usize,
    /// Total number of evictions.
    evictions: u64,
    /// Estimated memory usage in bytes.
    memory_bytes: usize,

    /// Cache hit ratio (0.0 to 1.0).
    pub fn hitRatio(self: CacheStats) f64 {
        if (self.total_lookups == 0) return 0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(self.total_lookups));
    }
};

/// LRU cache entry.
const CacheEntry = struct {
    key: u64,
    value: []f32,
    prev: ?*CacheEntry,
    next: ?*CacheEntry,
};

/// LRU embedding cache.
pub const EmbeddingCache = struct {
    allocator: std.mem.Allocator,
    config: CacheConfig,
    entries: std.AutoHashMapUnmanaged(u64, *CacheEntry),
    head: ?*CacheEntry,
    tail: ?*CacheEntry,
    stats: CacheStats,

    pub fn init(allocator: std.mem.Allocator, config: CacheConfig) EmbeddingCache {
        return .{
            .allocator = allocator,
            .config = config,
            .entries = std.AutoHashMapUnmanaged(u64, *CacheEntry){},
            .head = null,
            .tail = null,
            .stats = .{
                .total_lookups = 0,
                .hits = 0,
                .misses = 0,
                .current_entries = 0,
                .evictions = 0,
                .memory_bytes = 0,
            },
        };
    }

    pub fn deinit(self: *EmbeddingCache) void {
        // Free all entries
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*.value);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Get an embedding from cache by hash key.
    pub fn get(self: *EmbeddingCache, key: u64) ?[]const f32 {
        if (self.config.collect_stats) {
            self.stats.total_lookups += 1;
        }

        if (self.entries.get(key)) |entry| {
            if (self.config.collect_stats) {
                self.stats.hits += 1;
            }
            // Move to front (most recently used)
            self.moveToFront(entry);
            return entry.value;
        }

        if (self.config.collect_stats) {
            self.stats.misses += 1;
        }
        return null;
    }

    /// Put an embedding into cache.
    pub fn put(self: *EmbeddingCache, key: u64, value: []const f32) !void {
        // Check if already exists
        if (self.entries.get(key)) |existing| {
            // Update value
            self.allocator.free(existing.value);
            existing.value = try self.allocator.dupe(f32, value);
            self.moveToFront(existing);
            return;
        }

        // Evict if at capacity
        while (self.stats.current_entries >= self.config.max_entries) {
            self.evictOldest();
        }

        // Create new entry
        const entry = try self.allocator.create(CacheEntry);
        errdefer self.allocator.destroy(entry);

        entry.* = .{
            .key = key,
            .value = try self.allocator.dupe(f32, value),
            .prev = null,
            .next = self.head,
        };

        if (self.head) |h| {
            h.prev = entry;
        }
        self.head = entry;

        if (self.tail == null) {
            self.tail = entry;
        }

        try self.entries.put(self.allocator, key, entry);
        self.stats.current_entries += 1;
        self.stats.memory_bytes += value.len * @sizeOf(f32) + @sizeOf(CacheEntry);
    }

    /// Remove an entry from cache.
    pub fn remove(self: *EmbeddingCache, key: u64) bool {
        if (self.entries.fetchRemove(key)) |kv| {
            const entry = kv.value;
            self.removeFromList(entry);
            self.stats.memory_bytes -= entry.value.len * @sizeOf(f32) + @sizeOf(CacheEntry);
            self.allocator.free(entry.value);
            self.allocator.destroy(entry);
            self.stats.current_entries -= 1;
            return true;
        }
        return false;
    }

    /// Clear all entries.
    pub fn clear(self: *EmbeddingCache) void {
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*.value);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.entries.clearAndFree(self.allocator);
        self.head = null;
        self.tail = null;
        self.stats.current_entries = 0;
        self.stats.memory_bytes = 0;
    }

    /// Get cache statistics.
    pub fn getStats(self: *const EmbeddingCache) CacheStats {
        return self.stats;
    }

    /// Check if key exists in cache.
    pub fn contains(self: *const EmbeddingCache, key: u64) bool {
        return self.entries.contains(key);
    }

    /// Get current number of entries.
    pub fn count(self: *const EmbeddingCache) usize {
        return self.stats.current_entries;
    }

    fn moveToFront(self: *EmbeddingCache, entry: *CacheEntry) void {
        if (entry == self.head) return;

        // Remove from current position
        self.removeFromList(entry);

        // Add to front
        entry.prev = null;
        entry.next = self.head;

        if (self.head) |h| {
            h.prev = entry;
        }
        self.head = entry;

        if (self.tail == null) {
            self.tail = entry;
        }
    }

    fn removeFromList(self: *EmbeddingCache, entry: *CacheEntry) void {
        if (entry.prev) |p| {
            p.next = entry.next;
        } else {
            self.head = entry.next;
        }

        if (entry.next) |n| {
            n.prev = entry.prev;
        } else {
            self.tail = entry.prev;
        }

        entry.prev = null;
        entry.next = null;
    }

    fn evictOldest(self: *EmbeddingCache) void {
        const oldest = self.tail orelse return;
        _ = self.remove(oldest.key);
        self.stats.evictions += 1;
    }
};

/// Pre-computed embedding store for static corpora.
pub const EmbeddingStore = struct {
    allocator: std.mem.Allocator,
    embeddings: std.ArrayListUnmanaged(StoredEmbedding),
    index: std.StringHashMapUnmanaged(usize),

    const StoredEmbedding = struct {
        text: []const u8,
        vector: []f32,
    };

    pub fn init(allocator: std.mem.Allocator) EmbeddingStore {
        return .{
            .allocator = allocator,
            .embeddings = std.ArrayListUnmanaged(StoredEmbedding){},
            .index = std.StringHashMapUnmanaged(usize){},
        };
    }

    pub fn deinit(self: *EmbeddingStore) void {
        for (self.embeddings.items) |*emb| {
            self.allocator.free(emb.text);
            self.allocator.free(emb.vector);
        }
        self.embeddings.deinit(self.allocator);
        self.index.deinit(self.allocator);
    }

    /// Add an embedding to the store.
    pub fn add(self: *EmbeddingStore, text: []const u8, vector: []const f32) !void {
        const idx = self.embeddings.items.len;
        const text_copy = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(text_copy);

        const vector_copy = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(vector_copy);

        try self.embeddings.append(self.allocator, .{
            .text = text_copy,
            .vector = vector_copy,
        });

        try self.index.put(self.allocator, text_copy, idx);
    }

    /// Get embedding by text.
    pub fn getByText(self: *const EmbeddingStore, text: []const u8) ?[]const f32 {
        if (self.index.get(text)) |idx| {
            return self.embeddings.items[idx].vector;
        }
        return null;
    }

    /// Get embedding by index.
    pub fn getByIndex(self: *const EmbeddingStore, idx: usize) ?[]const f32 {
        if (idx < self.embeddings.items.len) {
            return self.embeddings.items[idx].vector;
        }
        return null;
    }

    /// Get all embeddings as matrix (for batch operations).
    pub fn getAllVectors(self: *const EmbeddingStore) []const []const f32 {
        var vectors = std.ArrayListUnmanaged([]const f32){};
        for (self.embeddings.items) |emb| {
            vectors.append(self.allocator, emb.vector) catch continue;
        }
        return vectors.toOwnedSlice(self.allocator) catch &.{};
    }

    /// Number of stored embeddings.
    pub fn count(self: *const EmbeddingStore) usize {
        return self.embeddings.items.len;
    }
};

test "cache initialization" {
    const allocator = std.testing.allocator;
    var cache = EmbeddingCache.init(allocator, .{});
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 0), cache.count());
}

test "cache put and get" {
    const allocator = std.testing.allocator;
    var cache = EmbeddingCache.init(allocator, .{});
    defer cache.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try cache.put(123, &vector);

    const result = cache.get(123);
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.?[0], 0.0001);
}

test "cache eviction" {
    const allocator = std.testing.allocator;
    var cache = EmbeddingCache.init(allocator, .{ .max_entries = 2 });
    defer cache.deinit();

    const v1 = [_]f32{1.0};
    const v2 = [_]f32{2.0};
    const v3 = [_]f32{3.0};

    try cache.put(1, &v1);
    try cache.put(2, &v2);
    try cache.put(3, &v3); // Should evict key 1

    try std.testing.expect(cache.get(1) == null);
    try std.testing.expect(cache.get(2) != null);
    try std.testing.expect(cache.get(3) != null);
}

test "cache statistics" {
    const allocator = std.testing.allocator;
    var cache = EmbeddingCache.init(allocator, .{});
    defer cache.deinit();

    const vector = [_]f32{1.0};
    try cache.put(1, &vector);

    _ = cache.get(1); // Hit
    _ = cache.get(2); // Miss

    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.hits);
    try std.testing.expectEqual(@as(u64, 1), stats.misses);
    try std.testing.expectEqual(@as(u64, 2), stats.total_lookups);
}

test "cache LRU order" {
    const allocator = std.testing.allocator;
    var cache = EmbeddingCache.init(allocator, .{ .max_entries = 2 });
    defer cache.deinit();

    const v1 = [_]f32{1.0};
    const v2 = [_]f32{2.0};
    const v3 = [_]f32{3.0};

    try cache.put(1, &v1);
    try cache.put(2, &v2);

    // Access key 1 to make it most recently used
    _ = cache.get(1);

    // Add key 3, should evict key 2 (least recently used)
    try cache.put(3, &v3);

    try std.testing.expect(cache.get(1) != null);
    try std.testing.expect(cache.get(2) == null);
    try std.testing.expect(cache.get(3) != null);
}
