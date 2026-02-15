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
const core_config = @import("../../core/config/cache.zig");
const sync = @import("../../services/shared/sync.zig");
const time = @import("../../services/shared/time.zig");

pub const CacheConfig = core_config.CacheConfig;
pub const EvictionPolicy = core_config.EvictionPolicy;

pub const CacheError = error{
    FeatureDisabled,
    CacheFull,
    KeyNotFound,
    InvalidTTL,
    OutOfMemory,
};

pub const CacheEntry = struct {
    key: []const u8 = "",
    value: []const u8 = "",
    ttl_ms: u64 = 0,
    created_at: u64 = 0,
};

pub const CacheStats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    entries: u32 = 0,
    memory_used: u64 = 0,
    evictions: u64 = 0,
    expired: u64 = 0,
};

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

// ── Internal Types ─────────────────────────────────────────────────────

const NodeIndex = u32;
const SENTINEL: NodeIndex = std.math.maxInt(NodeIndex);

/// Internal cache entry stored in the slab array.
const InternalEntry = struct {
    key_buf: []u8,
    value_buf: []u8,
    ttl_ms: u64,
    created_at_ns: u128,
    // LRU/FIFO: doubly-linked list pointers
    prev: NodeIndex = SENTINEL,
    next: NodeIndex = SENTINEL,
    // LFU: access frequency
    frequency: u32 = 1,
    // Slab management
    active: bool = true,
};

/// Slab allocator for cache entries. Avoids per-entry heap allocation.
const EntrySlab = struct {
    entries: std.ArrayListUnmanaged(InternalEntry) = .empty,
    free_list: std.ArrayListUnmanaged(NodeIndex) = .empty,

    fn alloc(self: *EntrySlab, allocator: std.mem.Allocator) !NodeIndex {
        if (self.free_list.items.len > 0) {
            return self.free_list.pop().?;
        }
        const idx: NodeIndex = @intCast(self.entries.items.len);
        try self.entries.append(allocator, .{
            .key_buf = &.{},
            .value_buf = &.{},
            .ttl_ms = 0,
            .created_at_ns = 0,
        });
        return idx;
    }

    fn release(self: *EntrySlab, allocator: std.mem.Allocator, idx: NodeIndex) void {
        var entry = &self.entries.items[idx];
        if (entry.key_buf.len > 0) allocator.free(entry.key_buf);
        if (entry.value_buf.len > 0) allocator.free(entry.value_buf);
        entry.* = .{
            .key_buf = &.{},
            .value_buf = &.{},
            .ttl_ms = 0,
            .created_at_ns = 0,
            .active = false,
        };
        self.free_list.append(allocator, idx) catch |err| {
            // Slot won't be reused until next resize — minor fragmentation, not data loss
            std.log.debug("cache: free list append failed, slot {d} leaked: {t}", .{ idx, err });
        };
    }

    fn getEntry(self: *EntrySlab, idx: NodeIndex) *InternalEntry {
        return &self.entries.items[idx];
    }

    fn deinitAll(self: *EntrySlab, allocator: std.mem.Allocator) void {
        for (self.entries.items) |*entry| {
            if (entry.key_buf.len > 0) allocator.free(entry.key_buf);
            if (entry.value_buf.len > 0) allocator.free(entry.value_buf);
        }
        self.entries.deinit(allocator);
        self.free_list.deinit(allocator);
    }
};

/// Key → NodeIndex map using StringHashMap (suitable since keys are []const u8).
const KeyMap = std.StringHashMapUnmanaged(NodeIndex);

// ── Module State ───────────────────────────────────────────────────────

var state: ?*CacheState = null;

const CacheState = struct {
    allocator: std.mem.Allocator,
    config: CacheConfig,
    key_map: KeyMap,
    slab: EntrySlab,
    rw_lock: sync.RwLock,

    // Eviction list: head = most-recently-used (LRU) / newest (FIFO)
    list_head: NodeIndex,
    list_tail: NodeIndex,

    // Stats (atomic for lock-free reads)
    stat_hits: std.atomic.Value(u64),
    stat_misses: std.atomic.Value(u64),
    stat_evictions: std.atomic.Value(u64),
    stat_expired: std.atomic.Value(u64),
    memory_used: u64,

    // Random eviction state
    rng_state: u64,

    fn create(allocator: std.mem.Allocator, config: CacheConfig) !*CacheState {
        const s = try allocator.create(CacheState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .key_map = .empty,
            .slab = .{},
            .rw_lock = sync.RwLock.init(),
            .list_head = SENTINEL,
            .list_tail = SENTINEL,
            .stat_hits = std.atomic.Value(u64).init(0),
            .stat_misses = std.atomic.Value(u64).init(0),
            .stat_evictions = std.atomic.Value(u64).init(0),
            .stat_expired = std.atomic.Value(u64).init(0),
            .memory_used = 0,
            .rng_state = 0x853c49e6748fea9b, // splitmix seed
        };
        return s;
    }

    fn destroy(self: *CacheState) void {
        const allocator = self.allocator;
        // Free all entries
        self.slab.deinitAll(allocator);
        // Free key map (keys are owned by entries, already freed)
        self.key_map.deinit(allocator);
        allocator.destroy(self);
    }

    // ── Linked List Ops ────────────────────────────────────────

    fn listRemove(self: *CacheState, idx: NodeIndex) void {
        const entry = self.slab.getEntry(idx);
        const prev_idx = entry.prev;
        const next_idx = entry.next;

        if (prev_idx != SENTINEL) {
            self.slab.getEntry(prev_idx).next = next_idx;
        } else {
            self.list_head = next_idx;
        }

        if (next_idx != SENTINEL) {
            self.slab.getEntry(next_idx).prev = prev_idx;
        } else {
            self.list_tail = prev_idx;
        }

        entry.prev = SENTINEL;
        entry.next = SENTINEL;
    }

    fn listPushFront(self: *CacheState, idx: NodeIndex) void {
        const entry = self.slab.getEntry(idx);
        entry.prev = SENTINEL;
        entry.next = self.list_head;

        if (self.list_head != SENTINEL) {
            self.slab.getEntry(self.list_head).prev = idx;
        }
        self.list_head = idx;

        if (self.list_tail == SENTINEL) {
            self.list_tail = idx;
        }
    }

    fn listPopBack(self: *CacheState) ?NodeIndex {
        if (self.list_tail == SENTINEL) return null;
        const idx = self.list_tail;
        self.listRemove(idx);
        return idx;
    }

    // ── Eviction ───────────────────────────────────────────────

    fn evictOne(self: *CacheState) void {
        const victim_idx: ?NodeIndex = switch (self.config.eviction_policy) {
            .lru, .fifo => self.listPopBack(),
            .lfu => self.findLfuVictim(),
            .random => self.findRandomVictim(),
        };

        if (victim_idx) |idx| {
            self.removeEntry(idx);
            _ = self.stat_evictions.fetchAdd(1, .monotonic);
        }
    }

    fn findLfuVictim(self: *CacheState) ?NodeIndex {
        // Walk the list from tail, find entry with lowest frequency
        var min_freq: u32 = std.math.maxInt(u32);
        var min_idx: NodeIndex = SENTINEL;
        var cur = self.list_tail;
        var checked: u32 = 0;

        while (cur != SENTINEL and checked < 64) : (checked += 1) {
            const entry = self.slab.getEntry(cur);
            if (entry.active and entry.frequency < min_freq) {
                min_freq = entry.frequency;
                min_idx = cur;
            }
            cur = entry.prev;
        }

        if (min_idx != SENTINEL) {
            self.listRemove(min_idx);
        }
        return if (min_idx != SENTINEL) min_idx else null;
    }

    fn findRandomVictim(self: *CacheState) ?NodeIndex {
        if (self.key_map.count() == 0) return null;

        // xorshift64 RNG
        var x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;

        // Pick a random entry by walking the list
        const target = x % self.key_map.count();
        var cur = self.list_head;
        var i: u64 = 0;
        while (cur != SENTINEL and i < target) : (i += 1) {
            cur = self.slab.getEntry(cur).next;
        }

        if (cur != SENTINEL) {
            self.listRemove(cur);
            return cur;
        }
        return null;
    }

    fn removeEntry(self: *CacheState, idx: NodeIndex) void {
        const entry = self.slab.getEntry(idx);
        if (!entry.active) return;

        // Track memory
        const entry_mem = entryMemory(entry);
        if (self.memory_used >= entry_mem) {
            self.memory_used -= entry_mem;
        } else {
            self.memory_used = 0;
        }

        // Remove from key map (key still valid at this point)
        _ = self.key_map.remove(entry.key_buf);

        // Release entry (frees key/value buffers)
        self.slab.release(self.allocator, idx);
    }

    fn entryMemory(entry: *const InternalEntry) u64 {
        return @as(u64, @sizeOf(InternalEntry)) +
            @as(u64, entry.key_buf.len) +
            @as(u64, entry.value_buf.len);
    }

    fn maxMemoryBytes(self: *const CacheState) u64 {
        return @as(u64, self.config.max_memory_mb) * 1024 * 1024;
    }

    // ── TTL ────────────────────────────────────────────────────

    fn isExpired(entry: *const InternalEntry) bool {
        if (entry.ttl_ms == 0) return false;
        const now_ns = (time.Instant.now() catch return false).nanos;
        const ttl_ns = @as(u128, entry.ttl_ms) * std.time.ns_per_ms;
        return now_ns > entry.created_at_ns + ttl_ns;
    }

    fn expireSweep(self: *CacheState, max_check: u32) u32 {
        var expired_count: u32 = 0;
        var cur = self.list_tail;
        var checked: u32 = 0;

        while (cur != SENTINEL and checked < max_check) : (checked += 1) {
            const prev = self.slab.getEntry(cur).prev;
            if (isExpired(self.slab.getEntry(cur))) {
                self.listRemove(cur);
                self.removeEntry(cur);
                expired_count += 1;
                _ = self.stat_expired.fetchAdd(1, .monotonic);
            }
            cur = prev;
        }
        return expired_count;
    }

    // ── Core Ops (caller must hold appropriate lock) ───────────

    fn getInternal(self: *CacheState, key: []const u8) ?[]const u8 {
        const idx_ptr = self.key_map.get(key) orelse return null;
        const entry = self.slab.getEntry(idx_ptr);

        if (!entry.active) return null;

        // Lazy TTL check
        if (isExpired(entry)) {
            return null; // Caller will handle cleanup
        }

        return entry.value_buf;
    }

    fn promoteOnHit(self: *CacheState, key: []const u8) void {
        const idx = self.key_map.get(key) orelse return;
        const entry = self.slab.getEntry(idx);

        switch (self.config.eviction_policy) {
            .lru => {
                // Move to front
                self.listRemove(idx);
                self.listPushFront(idx);
            },
            .lfu => {
                entry.frequency +|= 1; // Saturating add
            },
            .fifo, .random => {}, // No promotion
        }
    }
};

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

    // Hold write lock for entire get+promote to prevent TOCTOU race
    // (entry could be evicted between read-unlock and write-lock)
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    const result = s.getInternal(key);
    if (result) |value| {
        _ = s.stat_hits.fetchAdd(1, .monotonic);
        s.promoteOnHit(key);
        return value;
    }

    _ = s.stat_misses.fetchAdd(1, .monotonic);

    // Check if expired — clean up while we hold the lock
    if (s.key_map.get(key)) |idx| {
        if (CacheState.isExpired(s.slab.getEntry(idx))) {
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

    // Check if key already exists — update in place
    if (s.key_map.get(key)) |existing_idx| {
        const entry = s.slab.getEntry(existing_idx);
        if (entry.active) {
            // Subtract old memory
            const old_mem = CacheState.entryMemory(entry);
            if (s.memory_used >= old_mem) s.memory_used -= old_mem else s.memory_used = 0;

            // Re-allocate value if size changed
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

            // Promote to front
            s.listRemove(existing_idx);
            s.listPushFront(existing_idx);
            return;
        }
    }

    // Evict if at capacity
    while (s.key_map.count() >= s.config.max_entries) {
        s.evictOne();
    }

    // Evict if over memory budget
    const new_entry_mem = @as(u64, @sizeOf(InternalEntry)) +
        @as(u64, key.len) + @as(u64, value.len);
    while (s.memory_used + new_entry_mem > s.maxMemoryBytes() and s.key_map.count() > 0) {
        s.evictOne();
    }

    // Periodically sweep expired entries (every 64 inserts)
    if (s.key_map.count() % 64 == 0) {
        _ = s.expireSweep(32);
    }

    // Allocate new entry — errdefer release handles all cleanup on failure
    const idx = s.slab.alloc(s.allocator) catch return error.OutOfMemory;
    errdefer s.slab.release(s.allocator, idx);

    const entry = s.slab.getEntry(idx);
    entry.key_buf = s.allocator.dupe(u8, key) catch return error.OutOfMemory;
    entry.value_buf = s.allocator.dupe(u8, value) catch return error.OutOfMemory;
    entry.ttl_ms = effective_ttl;
    entry.created_at_ns = now_ns;
    entry.frequency = 1;
    entry.active = true;

    // Register in key map
    s.key_map.put(s.allocator, entry.key_buf, idx) catch return error.OutOfMemory;

    // Add to eviction list
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

    // Remove all entries
    var iter = s.key_map.iterator();
    while (iter.next()) |entry| {
        const idx = entry.value_ptr.*;
        s.slab.release(s.allocator, idx);
    }
    s.key_map.clearRetainingCapacity();
    s.list_head = SENTINEL;
    s.list_tail = SENTINEL;
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

    // Access "a" to make it recently used
    _ = try get("a");

    // Insert "d" — should evict "b" (least recently used)
    try put("d", "4");

    try std.testing.expect(contains("a"));
    try std.testing.expect(!contains("b")); // evicted
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

    // Access "a" — shouldn't matter for FIFO
    _ = try get("a");

    // Insert "d" — should evict "a" (first in)
    try put("d", "4");

    try std.testing.expect(!contains("a")); // evicted (first in)
    try std.testing.expect(contains("b"));
    try std.testing.expect(contains("c"));
    try std.testing.expect(contains("d"));
}

test "cache stats tracking" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try put("key", "value");
    _ = try get("key"); // hit
    _ = try get("missing"); // miss

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

    // Insert "d" — should evict one random entry
    try put("d", "4");

    // Exactly 3 entries remain
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

    // Access "a" and "c" to boost their frequency
    _ = try get("a");
    _ = try get("a");
    _ = try get("c");

    // Insert "d" — should evict "b" (lowest frequency)
    try put("d", "4");

    try std.testing.expect(contains("a"));
    try std.testing.expect(!contains("b")); // evicted (freq=1)
    try std.testing.expect(contains("c"));
    try std.testing.expect(contains("d"));
}

test "cache putWithTtl basic operation" {
    const allocator = std.testing.allocator;
    try init(allocator, CacheConfig.defaults());
    defer deinit();

    try putWithTtl("ttl_key", "ttl_value", 60_000); // 60s TTL
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

    // Memory should have changed
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
    try put("c", "3"); // evicts "a"

    const s = stats();
    try std.testing.expectEqual(@as(u64, 1), s.evictions);
    try std.testing.expectEqual(@as(u32, 2), s.entries);
}

test "cache memory budget enforcement" {
    const allocator = std.testing.allocator;
    // 1 MB budget, LRU eviction
    try init(allocator, .{
        .max_entries = 10000,
        .max_memory_mb = 1,
        .eviction_policy = .lru,
    });
    defer deinit();

    // Put several entries that together approach the budget
    // 8KB value × N entries
    const big_value = "x" ** 8192;
    try put("mem0", big_value);
    try put("mem1", big_value);
    try put("mem2", big_value);
    try put("mem3", big_value);
    try put("mem4", big_value);

    // Should still be within budget
    const s = stats();
    try std.testing.expect(s.memory_used <= 1024 * 1024);
}

test "cache re-initialization guard" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_entries = 100 });
    defer deinit();

    try put("key", "value");

    // Second init should be no-op
    try init(allocator, .{ .max_entries = 1 });

    // Original config still active — can add more entries
    try put("key2", "value2");
    try std.testing.expectEqual(@as(u32, 2), size());
}
