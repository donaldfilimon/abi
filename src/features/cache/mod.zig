//! Cache Module
//!
//! In-memory caching with LRU/LFU/FIFO/Random eviction, TTL support,
//! and thread-safe concurrent access via RwLock.
//!
//! Architecture:
//! - SwissMap-backed storage for O(1) key lookup
//! - 4 eviction strategies: LRU (doubly-linked list), LFU (frequency buckets),
//!   FIFO (queue), Random (RNG selection)
//! - Lazy TTL expiration on get() + size-triggered sweep
//! - RwLock for read-heavy concurrency (multiple readers, single writer)
//! - Cache owns all keys/values (copies on put, caller borrows on get)

const std = @import("std");
const time = @import("../../foundation/mod.zig").time;
pub const types = @import("types.zig");
const cache_state = @import("state.zig");
const eviction = @import("eviction.zig");
const ttl = @import("ttl.zig");

const CacheState = cache_state.CacheState;

pub const CacheConfig = types.CacheConfig;
pub const EvictionPolicy = types.EvictionPolicy;
pub const CacheError = types.CacheError;
pub const Error = CacheError;
pub const CacheEntry = types.CacheEntry;
pub const CacheStats = types.CacheStats;
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: CacheConfig,

    pub fn init(allocator: std.mem.Allocator, config: CacheConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ── Module State ───────────────────────────────────────────────────────

var state: ?*CacheState = null;

// ── Public API ─────────────────────────────────────────────────────────

pub fn init(allocator: std.mem.Allocator, config: CacheConfig) CacheError!void {
    if (state != null) return;
    state = CacheState.create(allocator, config) catch return error.OutOfMemory;
}

pub fn deinit() void {
    if (state) |s| {
        s.destroy();
        state = null;
    }
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return state != null;
}

pub fn get(key: []const u8) CacheError!?[]const u8 {
    const s = state orelse return error.FeatureDisabled;

    const needs_promotion = switch (s.config.eviction_policy) {
        .lru, .lfu => true,
        .fifo, .random => false,
    };

    if (needs_promotion) {
        s.rw_lock.lock();
        defer s.rw_lock.unlock();
        return getLockedImpl(key);
    } else {
        s.rw_lock.lockShared();
        defer s.rw_lock.unlockShared();

        const result = s.getInternal(key);
        if (result) |value| {
            _ = s.stat_hits.fetchAdd(1, .monotonic);
            return value;
        }
        _ = s.stat_misses.fetchAdd(1, .monotonic);
        return null;
    }
}

fn getLockedImpl(key: []const u8) ?[]const u8 {
    const s = state orelse return null;
    const result = s.getInternal(key);
    if (result) |value| {
        _ = s.stat_hits.fetchAdd(1, .monotonic);
        s.promoteOnHit(key);
        return value;
    }

    _ = s.stat_misses.fetchAdd(1, .monotonic);

    if (s.key_map.get(key)) |idx| {
        if (ttl.isExpired(s.slab.getEntry(idx))) {
            s.listRemove(idx);
            s.removeEntry(idx);
            _ = s.stat_expired.fetchAdd(1, .monotonic);
        }
    }

    return null;
}

pub fn put(key: []const u8, value: []const u8) CacheError!void {
    return putWithTtl(key, value, 0);
}

pub fn putWithTtl(key: []const u8, value: []const u8, ttl_ms: u64) CacheError!void {
    const s = state orelse return error.FeatureDisabled;

    const now_ns = (time.Instant.now() catch return error.OutOfMemory).nanos;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const effective_ttl = if (ttl_ms > 0) ttl_ms else s.config.default_ttl_ms;

    // Update existing key in place
    if (s.key_map.get(key)) |existing_idx| {
        const entry = s.slab.getEntry(existing_idx);
        if (entry.active) {
            const old_mem = CacheState.entryMemory(entry);
            if (s.memory_used >= old_mem) s.memory_used -= old_mem else s.memory_used = 0;

            if (entry.value_buf.len != value.len) {
                if (entry.value_buf.len > 0) s.allocator.free(entry.value_buf);
                entry.value_buf = s.allocator.dupe(u8, value) catch return error.OutOfMemory;
            } else {
                @memcpy(entry.value_buf, value);
            }
            entry.ttl_ms = effective_ttl;
            entry.created_at_ns = now_ns;
            entry.frequency = 1;

            s.memory_used += CacheState.entryMemory(entry);

            s.listRemove(existing_idx);
            s.listPushFront(existing_idx);
            return;
        }
    }

    // Evict if at capacity
    while (s.key_map.count() >= s.config.max_entries) {
        eviction.evictOne(s);
    }

    // Evict if over memory budget
    const new_entry_mem = @as(u64, @sizeOf(cache_state.InternalEntry)) +
        @as(u64, key.len) + @as(u64, value.len);
    while (s.memory_used + new_entry_mem > s.maxMemoryBytes() and s.key_map.count() > 0) {
        eviction.evictOne(s);
    }

    // Periodically sweep expired entries
    if (s.key_map.count() % 64 == 0) {
        _ = ttl.expireSweep(s, 32);
    }

    // Allocate new entry
    const idx = s.slab.alloc(s.allocator) catch return error.OutOfMemory;
    errdefer s.slab.release(s.allocator, idx);

    const entry = s.slab.getEntry(idx);
    entry.key_buf = s.allocator.dupe(u8, key) catch return error.OutOfMemory;
    errdefer s.allocator.free(entry.key_buf);
    entry.value_buf = s.allocator.dupe(u8, value) catch return error.OutOfMemory;
    errdefer s.allocator.free(entry.value_buf);
    entry.ttl_ms = effective_ttl;
    entry.created_at_ns = now_ns;
    entry.frequency = 1;
    entry.active = true;

    s.key_map.put(s.allocator, entry.key_buf, idx) catch return error.OutOfMemory;

    s.listPushFront(idx);
    s.memory_used += CacheState.entryMemory(entry);
}

pub fn delete(key: []const u8) CacheError!bool {
    const s = state orelse return error.FeatureDisabled;

    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const idx = s.key_map.get(key) orelse return false;
    s.listRemove(idx);
    s.removeEntry(idx);
    return true;
}

pub fn contains(key: []const u8) bool {
    const s = state orelse return false;
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();
    const result = s.getInternal(key);
    return result != null;
}

pub fn clear() void {
    const s = state orelse return;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    var iter = s.key_map.iterator();
    while (iter.next()) |entry| {
        const idx = entry.value_ptr.*;
        s.slab.release(s.allocator, idx);
    }
    s.key_map.clearRetainingCapacity();
    s.list_head = cache_state.SENTINEL;
    s.list_tail = cache_state.SENTINEL;
    s.memory_used = 0;
}

pub fn size() u32 {
    const s = state orelse return 0;
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();
    return @intCast(s.key_map.count());
}

pub fn stats() CacheStats {
    const s = state orelse return .{};
    return .{
        .hits = s.stat_hits.load(.monotonic),
        .misses = s.stat_misses.load(.monotonic),
        .entries = @intCast(s.key_map.count()),
        .memory_used = s.memory_used,
        .evictions = s.stat_evictions.load(.monotonic),
        .expired = s.stat_expired.load(.monotonic),
    };
}

// ── Tests ──────────────────────────────────────────────────────────────

test "cache basic put/get/delete" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try put("hello", "world");
    const result = try get("hello");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("world", result.?);

    const deleted = try delete("hello");
    try std.testing.expect(deleted);

    const after_delete = try get("hello");
    try std.testing.expect(after_delete == null);
}

test "cache contains and size" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try std.testing.expect(!contains("key1"));
    try std.testing.expectEqual(@as(u32, 0), size());

    try put("key1", "val1");
    try put("key2", "val2");

    try std.testing.expect(contains("key1"));
    try std.testing.expect(contains("key2"));
    try std.testing.expectEqual(@as(u32, 2), size());
}

test "cache clear" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try put("a", "1");
    try put("b", "2");
    try put("c", "3");
    try std.testing.expectEqual(@as(u32, 3), size());

    clear();
    try std.testing.expectEqual(@as(u32, 0), size());
    try std.testing.expect(!contains("a"));
}

test "cache LRU eviction" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .max_entries = 3,
        .eviction_policy = .lru,
    });
    defer deinit();

    try put("a", "1");
    try put("b", "2");
    try put("c", "3");

    _ = try get("a");

    try put("d", "4");

    try std.testing.expect(contains("a"));
    try std.testing.expect(!contains("b"));
    try std.testing.expect(contains("c"));
    try std.testing.expect(contains("d"));
}

test "cache FIFO eviction" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .max_entries = 3,
        .eviction_policy = .fifo,
    });
    defer deinit();

    try put("a", "1");
    try put("b", "2");
    try put("c", "3");

    _ = try get("a");

    try put("d", "4");

    try std.testing.expect(!contains("a"));
    try std.testing.expect(contains("b"));
    try std.testing.expect(contains("c"));
    try std.testing.expect(contains("d"));
}

test "cache stats tracking" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try put("key", "value");
    _ = try get("key");
    _ = try get("missing");

    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.hits);
    try std.testing.expectEqual(@as(u64, 1), s.misses);
    try std.testing.expectEqual(@as(u32, 1), s.entries);
    try std.testing.expect(s.memory_used > 0);
}

test "cache update existing key" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try put("key", "value1");
    try put("key", "value2");

    const result = try get("key");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("value2", result.?);
    try std.testing.expectEqual(@as(u32, 1), size());
}

test "cache random eviction" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .max_entries = 3,
        .eviction_policy = .random,
    });
    defer deinit();

    try put("a", "1");
    try put("b", "2");
    try put("c", "3");

    try put("d", "4");

    try std.testing.expectEqual(@as(u32, 3), size());
    try std.testing.expect(contains("d"));
}

test "cache LFU eviction by frequency" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .max_entries = 3,
        .eviction_policy = .lfu,
    });
    defer deinit();

    try put("a", "1");
    try put("b", "2");
    try put("c", "3");

    _ = try get("a");
    _ = try get("a");
    _ = try get("c");

    try put("d", "4");

    try std.testing.expect(contains("a"));
    try std.testing.expect(!contains("b"));
    try std.testing.expect(contains("c"));
    try std.testing.expect(contains("d"));
}

test "cache putWithTtl basic operation" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try putWithTtl("ttl_key", "ttl_value", 60_000);
    const result = try get("ttl_key");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("ttl_value", result.?);
}

test "cache update value with different length" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try put("key", "short");
    const s1 = stats();
    const mem1 = s1.memory_used;

    try put("key", "a much longer value than before");
    const s2 = stats();

    try std.testing.expect(s2.memory_used != mem1);
    try std.testing.expectEqual(@as(u32, 1), s2.entries);

    const result = try get("key");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("a much longer value than before", result.?);
}

test "cache stats after eviction" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .max_entries = 2,
        .eviction_policy = .lru,
    });
    defer deinit();

    try put("a", "1");
    try put("b", "2");
    try put("c", "3");

    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.evictions);
    try std.testing.expectEqual(@as(u32, 2), s.entries);
}

test "cache memory budget enforcement" {
    const allocator = std.testing.allocator;
    try init(allocator, .{
        .max_entries = 10000,
        .max_memory_mb = 1,
        .eviction_policy = .lru,
    });
    defer deinit();

    const big_value = "x" ** 8192;
    try put("mem0", big_value);
    try put("mem1", big_value);
    try put("mem2", big_value);
    try put("mem3", big_value);
    try put("mem4", big_value);

    const s = stats();
    try std.testing.expect(s.memory_used <= 1024 * 1024);
}

test "cache re-initialization guard" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_entries = 100 });
    defer deinit();

    try put("key", "value");

    try init(allocator, .{ .max_entries = 1 });

    try put("key2", "value2");
    try std.testing.expectEqual(@as(u32, 2), size());
}

test "cache delete non-existent key is no-op" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_entries = 10 });
    defer deinit();

    try put("exists", "value");
    const removed = try delete("nonexistent");
    try std.testing.expect(!removed);
    try std.testing.expectEqual(@as(u32, 1), size());
}

test "cache get after clear returns null" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_entries = 10 });
    defer deinit();

    try put("k1", "v1");
    try put("k2", "v2");
    clear();

    const v1 = try get("k1");
    const v2 = try get("k2");
    try std.testing.expect(v1 == null);
    try std.testing.expect(v2 == null);
    try std.testing.expectEqual(@as(u32, 0), size());
}

test "cache exact capacity fill" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_entries = 3, .eviction_policy = .fifo });
    defer deinit();

    try put("a", "1");
    try put("b", "2");
    try put("c", "3");
    try std.testing.expectEqual(@as(u32, 3), size());

    try std.testing.expect((try get("a")) != null);
    try std.testing.expect((try get("b")) != null);
    try std.testing.expect((try get("c")) != null);
}

test "cache stats hit and miss accuracy" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_entries = 10 });
    defer deinit();

    try put("key", "val");

    _ = try get("key");
    _ = try get("key");
    _ = try get("key");
    _ = try get("missing1");
    _ = try get("missing2");

    const s = stats();
    try std.testing.expectEqual(@as(u64, 3), s.hits);
    try std.testing.expectEqual(@as(u64, 2), s.misses);
}

test "cache FIFO shared read lock path" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_entries = 10, .eviction_policy = .fifo });
    defer deinit();

    try put("k1", "hello");
    const val = try get("k1");
    try std.testing.expect(val != null);
    try std.testing.expectEqualSlices(u8, "hello", val.?);
}

test {
    std.testing.refAllDecls(@This());
}
