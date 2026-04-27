//! Integration Tests: Cache Module
//!
//! Tests the in-memory cache module exports, eviction policy types,
//! config defaults, context lifecycle, and basic put/get/delete contracts.

const std = @import("std");
const abi = @import("abi");

const cache = abi.cache;

// ── Type Export Tests ──────────────────────────────────────────────────

test "cache: module exports expected types" {
    const _config = cache.CacheConfig{};
    try std.testing.expectEqual(@as(u32, 10_000), _config.max_entries);
    try std.testing.expectEqual(@as(u32, 256), _config.max_memory_mb);
    try std.testing.expectEqual(@as(u64, 300_000), _config.default_ttl_ms);
    try std.testing.expectEqual(cache.EvictionPolicy.lru, _config.eviction_policy);

    const _err: cache.CacheError = error.CacheFull;
    _ = _err;

    const _entry = cache.CacheEntry{};
    try std.testing.expectEqualStrings("", _entry.key);
    try std.testing.expectEqualStrings("", _entry.value);
    try std.testing.expectEqual(@as(u64, 0), _entry.ttl_ms);

    const _stats = cache.CacheStats{};
    try std.testing.expectEqual(@as(u64, 0), _stats.hits);
    try std.testing.expectEqual(@as(u64, 0), _stats.misses);
    try std.testing.expectEqual(@as(u32, 0), _stats.entries);
    try std.testing.expectEqual(@as(u64, 0), _stats.memory_used);
    try std.testing.expectEqual(@as(u64, 0), _stats.evictions);
    try std.testing.expectEqual(@as(u64, 0), _stats.expired);
}

test "cache: eviction policy enum variants" {
    const lru = cache.EvictionPolicy.lru;
    const lfu = cache.EvictionPolicy.lfu;
    const fifo = cache.EvictionPolicy.fifo;
    const random = cache.EvictionPolicy.random;

    try std.testing.expect(lru != lfu);
    try std.testing.expect(fifo != random);
    try std.testing.expect(lru != fifo);
}

test "cache: config defaults factory" {
    const config = cache.CacheConfig.defaults();
    try std.testing.expectEqual(@as(u32, 10_000), config.max_entries);
    try std.testing.expectEqual(cache.EvictionPolicy.lru, config.eviction_policy);
}

test "cache: error type is aliased as Error" {
    const err1: cache.Error = error.CacheFull;
    const err2: cache.CacheError = err1;
    _ = err2;
}

// ── Context Lifecycle Tests ────────────────────────────────────────────

test "cache: context init and deinit" {
    const allocator = std.testing.allocator;
    const ctx = try cache.Context.init(allocator, cache.CacheConfig.defaults());
    defer ctx.deinit();

    try std.testing.expectEqual(cache.EvictionPolicy.lru, ctx.config.eviction_policy);
}

test "cache: context with custom config" {
    const allocator = std.testing.allocator;
    const ctx = try cache.Context.init(allocator, .{
        .max_entries = 500,
        .max_memory_mb = 64,
        .default_ttl_ms = 60_000,
        .eviction_policy = .fifo,
    });
    defer ctx.deinit();

    try std.testing.expectEqual(@as(u32, 500), ctx.config.max_entries);
    try std.testing.expectEqual(@as(u32, 64), ctx.config.max_memory_mb);
    try std.testing.expectEqual(cache.EvictionPolicy.fifo, ctx.config.eviction_policy);
}

// ── Module API Tests ───────────────────────────────────────────────────

test "cache: isEnabled returns true" {
    try std.testing.expect(cache.isEnabled());
}

test "cache: init deinit lifecycle" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try std.testing.expect(cache.isInitialized());
}

test "cache: put and get round-trip" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try cache.put("integ-key", "integ-value");
    const result = try cache.get("integ-key");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("integ-value", result.?);
}

test "cache: delete returns correct boolean" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try cache.put("to-remove", "data");
    const deleted = try cache.delete("to-remove");
    try std.testing.expect(deleted);

    const not_deleted = try cache.delete("nonexistent");
    try std.testing.expect(!not_deleted);
}

test "cache: contains contract" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try std.testing.expect(!cache.contains("missing"));
    try cache.put("present", "value");
    try std.testing.expect(cache.contains("present"));
}

test "cache: size tracks entries" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try std.testing.expectEqual(@as(u32, 0), cache.size());
    try cache.put("k1", "v1");
    try cache.put("k2", "v2");
    try std.testing.expectEqual(@as(u32, 2), cache.size());
}

test "cache: clear removes all entries" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try cache.put("a", "1");
    try cache.put("b", "2");
    try cache.put("c", "3");
    try std.testing.expectEqual(@as(u32, 3), cache.size());

    cache.clear();
    try std.testing.expectEqual(@as(u32, 0), cache.size());
    try std.testing.expect(!cache.contains("a"));
}

test "cache: stats track hits and misses" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try cache.put("key", "val");
    _ = try cache.get("key"); // hit
    _ = try cache.get("missing"); // miss

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 1), s.hits);
    try std.testing.expectEqual(@as(u64, 1), s.misses);
    try std.testing.expectEqual(@as(u32, 1), s.entries);
    try std.testing.expect(s.memory_used > 0);
}

test "cache: LRU eviction on capacity" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, .{
        .max_entries = 3,
        .eviction_policy = .lru,
    });
    defer cache.deinit();

    try cache.put("a", "1");
    try cache.put("b", "2");
    try cache.put("c", "3");

    // Access "a" to make it recently used
    _ = try cache.get("a");

    // Insert "d" — should evict "b" (least recently used)
    try cache.put("d", "4");

    try std.testing.expect(cache.contains("a"));
    try std.testing.expect(!cache.contains("b"));
    try std.testing.expect(cache.contains("c"));
    try std.testing.expect(cache.contains("d"));
}

test "cache: FIFO eviction on capacity" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, .{
        .max_entries = 3,
        .eviction_policy = .fifo,
    });
    defer cache.deinit();

    try cache.put("a", "1");
    try cache.put("b", "2");
    try cache.put("c", "3");

    // Insert "d" — should evict "a" (first in)
    try cache.put("d", "4");

    try std.testing.expect(!cache.contains("a"));
    try std.testing.expect(cache.contains("b"));
    try std.testing.expect(cache.contains("d"));
}

test "cache: update existing key preserves size" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try cache.put("key", "value1");
    try cache.put("key", "value2");

    const result = try cache.get("key");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("value2", result.?);
    try std.testing.expectEqual(@as(u32, 1), cache.size());
}

test "cache: putWithTtl stores value" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, cache.CacheConfig.defaults());
    defer cache.deinit();

    try cache.putWithTtl("ttl-key", "ttl-value", 60_000);
    const result = try cache.get("ttl-key");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("ttl-value", result.?);
}

test "cache: stats after eviction tracks eviction count" {
    const allocator = std.testing.allocator;
    try cache.init(allocator, .{
        .max_entries = 2,
        .eviction_policy = .lru,
    });
    defer cache.deinit();

    try cache.put("a", "1");
    try cache.put("b", "2");
    try cache.put("c", "3"); // evicts one

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 1), s.evictions);
    try std.testing.expectEqual(@as(u32, 2), s.entries);
}

test "cache: sub-module types accessible" {
    // Verify types sub-module is reachable through abi.cache
    _ = cache.types;
}

test {
    std.testing.refAllDecls(@This());
}
