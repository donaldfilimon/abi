const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;
pub const types = @import("types.zig");

pub const NodeIndex = u32;
pub const SENTINEL: NodeIndex = std.math.maxInt(NodeIndex);

/// Internal cache entry stored in the slab array.
pub const InternalEntry = struct {
    key_buf: []u8,
    value_buf: []u8,
    ttl_ms: u64,
    created_at_ns: u128,
    prev: NodeIndex = SENTINEL,
    next: NodeIndex = SENTINEL,
    frequency: u32 = 1,
    active: bool = true,
};

/// Slab allocator for cache entries. Avoids per-entry heap allocation.
pub const EntrySlab = struct {
    entries: std.ArrayListUnmanaged(InternalEntry) = .empty,
    free_list: std.ArrayListUnmanaged(NodeIndex) = .empty,

    pub fn alloc(self: *EntrySlab, allocator: std.mem.Allocator) !NodeIndex {
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

    pub fn release(self: *EntrySlab, allocator: std.mem.Allocator, idx: NodeIndex) void {
        const entry = &self.entries.items[idx];
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
            std.log.debug("cache: free list append failed, slot {d} leaked: {t}", .{ idx, err });
        };
    }

    pub fn getEntry(self: *EntrySlab, idx: NodeIndex) *InternalEntry {
        return &self.entries.items[idx];
    }

    pub fn deinitAll(self: *EntrySlab, allocator: std.mem.Allocator) void {
        for (self.entries.items) |*entry| {
            if (entry.key_buf.len > 0) allocator.free(entry.key_buf);
            if (entry.value_buf.len > 0) allocator.free(entry.value_buf);
        }
        self.entries.deinit(allocator);
        self.free_list.deinit(allocator);
    }
};

const KeyMap = std.StringHashMapUnmanaged(NodeIndex);

pub const CacheState = struct {
    allocator: std.mem.Allocator,
    config: types.CacheConfig,
    key_map: KeyMap,
    slab: EntrySlab,
    rw_lock: sync.RwLock,

    list_head: NodeIndex,
    list_tail: NodeIndex,

    stat_hits: std.atomic.Value(u64),
    stat_misses: std.atomic.Value(u64),
    stat_evictions: std.atomic.Value(u64),
    stat_expired: std.atomic.Value(u64),
    memory_used: u64,

    rng_state: u64,

    pub fn create(allocator: std.mem.Allocator, config: types.CacheConfig) !*CacheState {
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
            .rng_state = 0x853c49e6748fea9b,
        };
        return s;
    }

    pub fn destroy(self: *CacheState) void {
        const allocator = self.allocator;
        self.slab.deinitAll(allocator);
        self.key_map.deinit(allocator);
        allocator.destroy(self);
    }

    // ── Linked List Ops ────────────────────────────────────────

    pub fn listRemove(self: *CacheState, idx: NodeIndex) void {
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

    pub fn listPushFront(self: *CacheState, idx: NodeIndex) void {
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

    pub fn listPopBack(self: *CacheState) ?NodeIndex {
        if (self.list_tail == SENTINEL) return null;
        const idx = self.list_tail;
        self.listRemove(idx);
        return idx;
    }

    // ── Entry Management ───────────────────────────────────────

    pub fn removeEntry(self: *CacheState, idx: NodeIndex) void {
        const entry = self.slab.getEntry(idx);
        if (!entry.active) return;

        const entry_mem = entryMemory(entry);
        if (self.memory_used >= entry_mem) {
            self.memory_used -= entry_mem;
        } else {
            self.memory_used = 0;
        }

        _ = self.key_map.remove(entry.key_buf);
        self.slab.release(self.allocator, idx);
    }

    pub fn entryMemory(entry: *const InternalEntry) u64 {
        return @as(u64, @sizeOf(InternalEntry)) +
            @as(u64, entry.key_buf.len) +
            @as(u64, entry.value_buf.len);
    }

    pub fn maxMemoryBytes(self: *const CacheState) u64 {
        return @as(u64, self.config.max_memory_mb) * 1024 * 1024;
    }

    // ── Core Internal Ops ──────────────────────────────────────

    pub fn getInternal(self: *CacheState, key: []const u8) ?[]const u8 {
        const idx_ptr = self.key_map.get(key) orelse return null;
        const entry = self.slab.getEntry(idx_ptr);

        if (!entry.active) return null;

        const ttl = @import("ttl.zig");
        if (ttl.isExpired(entry)) {
            return null;
        }

        return entry.value_buf;
    }

    pub fn promoteOnHit(self: *CacheState, key: []const u8) void {
        const idx = self.key_map.get(key) orelse return;
        const entry = self.slab.getEntry(idx);

        switch (self.config.eviction_policy) {
            .lru => {
                self.listRemove(idx);
                self.listPushFront(idx);
            },
            .lfu => {
                entry.frequency +|= 1;
            },
            .fifo, .random => {},
        }
    }
};
