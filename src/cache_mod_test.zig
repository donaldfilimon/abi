//! Focused cache unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const cache = @import("features/cache/mod.zig");

test {
    std.testing.refAllDecls(cache);
}

// ── Module lifecycle tests ─────────────────────────────────────────────

test "cache isEnabled returns true" {
    try std.testing.expect(cache.isEnabled());
}

test "cache not initialized before init" {
    // State should be null before explicit init
    // (Only valid if no other test has left state behind)
    try std.testing.expectEqual(@as(u32, 0), cache.size());
}

test "cache init deinit cycle" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    try std.testing.expect(cache.isInitialized());
    cache.deinit();
    try std.testing.expect(!cache.isInitialized());
}

// ── Error when not initialized ─────────────────────────────────────────

test "cache get returns FeatureDisabled when not initialized" {
    // Ensure cache is not initialized
    cache.deinit();
    try std.testing.expectError(error.FeatureDisabled, cache.get("any"));
}

test "cache put returns FeatureDisabled when not initialized" {
    cache.deinit();
    try std.testing.expectError(error.FeatureDisabled, cache.put("key", "value"));
}

test "cache delete returns FeatureDisabled when not initialized" {
    cache.deinit();
    try std.testing.expectError(error.FeatureDisabled, cache.delete("key"));
}

// ── Eviction policy tests ──────────────────────────────────────────────

test "cache LRU evicts least recently used" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, .{ .max_entries = 3, .eviction_policy = .lru });
    defer cache.deinit();

    try cache.put("a", "1");
    try cache.put("b", "2");
    try cache.put("c", "3");
    // Access 'a' to make it recently used
    _ = try cache.get("a");
    // Insert 'd' to trigger eviction of LRU item ('b')
    try cache.put("d", "4");

    try std.testing.expect(cache.contains("a"));
    try std.testing.expect(!cache.contains("b")); // evicted
    try std.testing.expect(cache.contains("c"));
    try std.testing.expect(cache.contains("d"));
}

test "cache FIFO evicts oldest entry" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, .{ .max_entries = 2, .eviction_policy = .fifo });
    defer cache.deinit();

    try cache.put("first", "1");
    try cache.put("second", "2");
    try cache.put("third", "3"); // evicts "first"

    try std.testing.expect(!cache.contains("first"));
    try std.testing.expect(cache.contains("second"));
    try std.testing.expect(cache.contains("third"));
}

// ── Stats accuracy ─────────────────────────────────────────────────────

test "cache stats track hits misses and evictions" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, .{ .max_entries = 2, .eviction_policy = .lru });
    defer cache.deinit();

    try cache.put("x", "1");
    _ = try cache.get("x"); // hit
    _ = try cache.get("nope"); // miss

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 1), s.hits);
    try std.testing.expectEqual(@as(u64, 1), s.misses);
    try std.testing.expectEqual(@as(u32, 1), s.entries);
}

test "cache stats eviction count" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, .{ .max_entries = 1, .eviction_policy = .lru });
    defer cache.deinit();

    try cache.put("a", "1");
    try cache.put("b", "2"); // evicts "a"

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 1), s.evictions);
}

// ── Update in-place ────────────────────────────────────────────────────

test "cache put overwrites existing value" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try cache.put("key", "original");
    try cache.put("key", "updated");

    const val = try cache.get("key");
    try std.testing.expect(val != null);
    try std.testing.expectEqualStrings("updated", val.?);
    try std.testing.expectEqual(@as(u32, 1), cache.size());
}

// ── Context tests ──────────────────────────────────────────────────────

test "cache Context init and deinit" {
    const allocator = std.testing.allocator;
    const ctx = try cache.Context.init(allocator, cache.CacheConfig.defaults());
    defer ctx.deinit();
    try std.testing.expectEqual(allocator, ctx.allocator);
}

// ── Type tests ─────────────────────────────────────────────────────────

test "cache CacheStats default values are zero" {
    const s = cache.CacheStats{};
    try std.testing.expectEqual(@as(u64, 0), s.hits);
    try std.testing.expectEqual(@as(u64, 0), s.misses);
    try std.testing.expectEqual(@as(u32, 0), s.entries);
    try std.testing.expectEqual(@as(u64, 0), s.memory_used);
    try std.testing.expectEqual(@as(u64, 0), s.evictions);
    try std.testing.expectEqual(@as(u64, 0), s.expired);
}

test "cache CacheEntry default values" {
    const e = cache.CacheEntry{};
    try std.testing.expectEqualStrings("", e.key);
    try std.testing.expectEqualStrings("", e.value);
    try std.testing.expectEqual(@as(u64, 0), e.ttl_ms);
}
