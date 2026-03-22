//! ═══════════════════════════════════════════════════════════════════════════
//! SwissMap: SIMD-Probed Hash Table
//! ═══════════════════════════════════════════════════════════════════════════
//!
//! Adapted from abi-system-v2.0 hashmap.zig for the ABI framework.
//! Self-contained module — no external utility dependencies.
//!
//! Open-addressing hash map with 1-byte control metadata per slot,
//! inspired by Google's SwissTable / Abseil flat_hash_map. The control
//! byte encodes:
//!   - Empty   (0x80) — slot never occupied
//!   - Deleted (0xFE) — tombstone
//!   - Full    (H2)   — 7-bit secondary hash for SIMD-parallel matching
//!
//! Probe strategy: triangular probing (probe[i] = i*(i+1)/2 mod capacity).
//! Group width: 16 slots — matches SSE register width for bulk comparison.
//!
//! Design rationale:
//!   The WDBX distributed exchange needs fast key->shard mapping and
//!   embedding vector lookup tables. This map provides O(1) amortized
//!   get/put with excellent cache behavior: control bytes fit in a single
//!   cache line per 64-slot group, and the separate key/value arrays
//!   avoid polluting the probe path with large values.
//!
//! Performance: ~30ns lookup, ~50ns insert (hot cache, 50% load factor)
//!
//! ## Security: Hash Flooding (HashDoS)
//!
//! The default constructor uses a deterministic hash seed (wyhash/splitmix64).
//! This is fine for internal/trusted keys. For untrusted input (user-supplied
//! keys, network data, external file content), use `initWithSeed()` with a
//! random seed to prevent hash-collision denial-of-service attacks.
//!
//! Iteration order is NOT stable across insertions/deletions and must not be
//! relied upon. Modifying the map during iteration is undefined behavior
//! (matches `std.HashMap` semantics).
//! ═══════════════════════════════════════════════════════════════════════════

const std = @import("std");
const Math = @import("primitives.zig").Math;

// ─── Inline Helpers ──────────────────────────────────────────────────────────

fn nextPowerOfTwo(x: usize) usize {
    return Math.nextPowerOfTwo(usize, x);
}

// ─── Control Byte Constants ──────────────────────────────────────────────────

const CTRL_EMPTY: u8 = 0x80;
const CTRL_DELETED: u8 = 0xFE;

/// Extract the 7-bit H2 hash (must have high bit = 0 to distinguish from EMPTY)
inline fn h2(hash: u64) u8 {
    return @truncate((hash >> 57) & 0x7F);
}

inline fn isEmptyOrDeleted(ctrl: u8) bool {
    return ctrl >= 0x80;
}

inline fn isEmpty(ctrl: u8) bool {
    return ctrl == CTRL_EMPTY;
}

/// SIMD group matching — compares 16 control bytes at once.
/// Returns a bitmask where bit i is set if ctrl[i] == target.
inline fn groupMatch(ctrl_ptr: [*]const u8, target: u8) u16 {
    const Group = @Vector(16, u8);
    const group: Group = ctrl_ptr[0..16].*;
    const needle: Group = @splat(target);
    const mask: u16 = @bitCast(group == needle);
    return mask;
}

/// Returns a bitmask of empty slots in a group of 16 control bytes.
inline fn groupMatchEmpty(ctrl_ptr: [*]const u8) u16 {
    return groupMatch(ctrl_ptr, CTRL_EMPTY);
}

/// Returns a bitmask of empty or deleted slots in a group of 16 control bytes.
inline fn groupMatchEmptyOrDeleted(ctrl_ptr: [*]const u8) u16 {
    const Group = @Vector(16, u8);
    const group: Group = ctrl_ptr[0..16].*;
    const threshold: Group = @splat(0x80);
    // Empty (0x80) and Deleted (0xFE) both have the high bit set
    const mask: u16 = @bitCast(@as(@Vector(16, bool), (group & threshold) == threshold));
    return mask;
}

// ─── SwissMap ────────────────────────────────────────────────────────────────

pub fn SwissMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();
        const group_width = 16;
        const default_capacity = 16;

        ctrl: []u8,
        keys: []K,
        values: []V,
        capacity: usize,
        size: usize,
        allocator: std.mem.Allocator,
        growth_left: usize,
        /// Per-instance hash seed for HashDoS resistance.
        /// Default (0) uses deterministic hashing for backward compatibility.
        hash_seed: u64,

        // ── Lifecycle ────────────────────────────────────────────

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .ctrl = &.{},
                .keys = &.{},
                .values = &.{},
                .capacity = 0,
                .size = 0,
                .allocator = allocator,
                .growth_left = 0,
                .hash_seed = 0,
            };
        }

        /// Create a SwissMap with a per-instance hash seed for
        /// HashDoS resistance when keys come from untrusted sources.
        pub fn initWithSeed(allocator: std.mem.Allocator, seed: u64) Self {
            return Self{
                .ctrl = &.{},
                .keys = &.{},
                .values = &.{},
                .capacity = 0,
                .size = 0,
                .allocator = allocator,
                .growth_left = 0,
                .hash_seed = seed,
            };
        }

        pub fn initCapacity(allocator: std.mem.Allocator, requested: usize) !Self {
            var self = init(allocator);
            try self.ensureCapacity(requested);
            return self;
        }

        pub fn deinit(self: *Self) void {
            const seed = self.hash_seed;
            if (self.capacity > 0) {
                // ctrl has capacity + group_width sentinel bytes
                self.allocator.free(self.ctrl[0 .. self.capacity + group_width]);
                self.allocator.free(self.keys[0..self.capacity]);
                self.allocator.free(self.values[0..self.capacity]);
            }
            self.* = init(self.allocator);
            self.hash_seed = seed;
        }

        // ── Core Operations ──────────────────────────────────────

        pub fn get(self: *const Self, key: K) ?V {
            if (self.capacity == 0) return null;

            const hash = self.hashKey(key);
            const target_h2 = h2(hash);
            var group_idx = @as(usize, @truncate(hash)) & (self.capacity - 1);
            // Align to group boundary
            group_idx = group_idx & ~@as(usize, group_width - 1);
            var probe: usize = 0;

            while (probe <= self.capacity) {
                // SIMD: compare 16 control bytes at once
                var match_mask = groupMatch(self.ctrl.ptr + group_idx, target_h2);
                while (match_mask != 0) {
                    const bit = @ctz(match_mask);
                    const pos = group_idx + bit;
                    if (keysEqual(self.keys[pos], key)) {
                        return self.values[pos];
                    }
                    match_mask &= match_mask - 1; // clear lowest set bit
                }

                // If any slot in this group is empty, key is absent
                if (groupMatchEmpty(self.ctrl.ptr + group_idx) != 0) return null;

                // Next group (quadratic probing by group)
                probe += group_width;
                group_idx = (group_idx + probe) & (self.capacity - 1);
                group_idx = group_idx & ~@as(usize, group_width - 1);
            }
            return null;
        }

        pub fn getPtr(self: *Self, key: K) ?*V {
            if (self.capacity == 0) return null;

            const hash = self.hashKey(key);
            const target_h2 = h2(hash);
            var group_idx = @as(usize, @truncate(hash)) & (self.capacity - 1);
            group_idx = group_idx & ~@as(usize, group_width - 1);
            var probe: usize = 0;

            while (probe <= self.capacity) {
                var match_mask = groupMatch(self.ctrl.ptr + group_idx, target_h2);
                while (match_mask != 0) {
                    const bit = @ctz(match_mask);
                    const pos = group_idx + bit;
                    if (keysEqual(self.keys[pos], key)) {
                        return &self.values[pos];
                    }
                    match_mask &= match_mask - 1;
                }
                if (groupMatchEmpty(self.ctrl.ptr + group_idx) != 0) return null;
                probe += group_width;
                group_idx = (group_idx + probe) & (self.capacity - 1);
                group_idx = group_idx & ~@as(usize, group_width - 1);
            }
            return null;
        }

        pub fn put(self: *Self, key: K, value: V) !void {
            try self.ensureCapacity(self.size + 1);

            const hash = self.hashKey(key);
            const target_h2 = h2(hash);
            var group_idx = @as(usize, @truncate(hash)) & (self.capacity - 1);
            group_idx = group_idx & ~@as(usize, group_width - 1);
            var probe: usize = 0;

            // Track first available insert position (empty or deleted slot)
            var insert_pos: ?usize = null;

            while (probe <= self.capacity) {
                // SIMD: check for existing key (H2 match)
                var match_mask = groupMatch(self.ctrl.ptr + group_idx, target_h2);
                while (match_mask != 0) {
                    const bit = @ctz(match_mask);
                    const pos = group_idx + bit;
                    if (keysEqual(self.keys[pos], key)) {
                        self.values[pos] = value;
                        return;
                    }
                    match_mask &= match_mask - 1;
                }

                // Remember first available slot for insertion
                if (insert_pos == null) {
                    const avail_mask = groupMatchEmptyOrDeleted(self.ctrl.ptr + group_idx);
                    if (avail_mask != 0) {
                        insert_pos = group_idx + @as(usize, @ctz(avail_mask));
                    }
                }

                // An empty slot terminates the probe chain — key not present
                if (groupMatchEmpty(self.ctrl.ptr + group_idx) != 0) break;

                probe += group_width;
                group_idx = (group_idx + probe) & (self.capacity - 1);
                group_idx = group_idx & ~@as(usize, group_width - 1);
            }

            // Insert at the first available position
            if (insert_pos) |pos| {
                const was_empty = isEmpty(self.ctrl[pos]);
                self.ctrl[pos] = target_h2;
                self.keys[pos] = key;
                self.values[pos] = value;
                self.size += 1;
                if (was_empty) {
                    if (self.growth_left > 0) self.growth_left -= 1;
                }
                return;
            }

            // Probing exhausted all slots — capacity invariant violated
            return error.OutOfMemory;
        }

        pub fn remove(self: *Self, key: K) bool {
            if (self.capacity == 0) return false;

            const hash = self.hashKey(key);
            const target_h2 = h2(hash);
            var group_idx = @as(usize, @truncate(hash)) & (self.capacity - 1);
            group_idx = group_idx & ~@as(usize, group_width - 1);
            var probe: usize = 0;

            while (probe <= self.capacity) {
                var match_mask = groupMatch(self.ctrl.ptr + group_idx, target_h2);
                while (match_mask != 0) {
                    const bit = @ctz(match_mask);
                    const pos = group_idx + bit;
                    if (keysEqual(self.keys[pos], key)) {
                        self.ctrl[pos] = CTRL_DELETED;
                        self.size -= 1;
                        return true;
                    }
                    match_mask &= match_mask - 1;
                }
                if (groupMatchEmpty(self.ctrl.ptr + group_idx) != 0) return false;
                probe += group_width;
                group_idx = (group_idx + probe) & (self.capacity - 1);
                group_idx = group_idx & ~@as(usize, group_width - 1);
            }
            return false;
        }

        pub fn contains(self: *const Self, key: K) bool {
            return self.get(key) != null;
        }

        pub fn count(self: *const Self) usize {
            return self.size;
        }

        pub fn clear(self: *Self) void {
            if (self.capacity > 0) {
                @memset(self.ctrl[0..self.capacity], CTRL_EMPTY);
                self.size = 0;
                self.growth_left = capacityToGrowth(self.capacity);
            }
        }

        // ── Iteration ────────────────────────────────────────────

        pub const Entry = struct { key: K, value: V };

        pub const Iterator = struct {
            map: *const Self,
            index: usize,

            pub fn next(self: *Iterator) ?Entry {
                while (self.index < self.map.capacity) {
                    const i = self.index;
                    self.index += 1;
                    if (!isEmptyOrDeleted(self.map.ctrl[i])) {
                        return .{
                            .key = self.map.keys[i],
                            .value = self.map.values[i],
                        };
                    }
                }
                return null;
            }
        };

        pub fn iterator(self: *const Self) Iterator {
            return .{ .map = self, .index = 0 };
        }

        /// Insert without capacity check — only used during rehash
        fn insertUnchecked(self: *Self, key: K, value: V) void {
            const hash = self.hashKey(key);
            const target_h2 = h2(hash);
            var group_idx = @as(usize, @truncate(hash)) & (self.capacity - 1);
            group_idx = group_idx & ~@as(usize, group_width - 1);
            var probe: usize = 0;

            while (probe <= self.capacity) {
                const empty_mask = groupMatchEmptyOrDeleted(self.ctrl.ptr + group_idx);
                if (empty_mask != 0) {
                    const bit = @ctz(empty_mask);
                    const pos = group_idx + bit;
                    const was_empty = isEmpty(self.ctrl[pos]);
                    self.ctrl[pos] = target_h2;
                    self.keys[pos] = key;
                    self.values[pos] = value;
                    self.size += 1;
                    if (was_empty) {
                        if (self.growth_left > 0) self.growth_left -= 1;
                    }
                    return;
                }

                probe += group_width;
                group_idx = (group_idx + probe) & (self.capacity - 1);
                group_idx = group_idx & ~@as(usize, group_width - 1);
            }

            // Safety: insertUnchecked is only called from rehash() with a
            // freshly-allocated table sized to fit all entries. If we exhaust
            // probing, the capacity calculation in ensureCapacity is wrong.
            unreachable;
        }

        // ── Capacity Management ──────────────────────────────────

        fn ensureCapacity(self: *Self, required: usize) !void {
            if (required <= self.size and self.growth_left > 0) return;
            if (self.capacity == 0 or self.growth_left == 0 or required > capacityToGrowth(self.capacity) + self.size) {
                const doubled = std.math.mul(usize, required, 2) catch return error.OutOfMemory;
                try self.rehash(@max(default_capacity, nextPowerOfTwo(doubled)));
            }
        }

        fn rehash(self: *Self, new_capacity: usize) !void {
            const old_ctrl = self.ctrl;
            const old_keys = self.keys;
            const old_values = self.values;
            const old_capacity = self.capacity;

            // Allocate new arrays (errdefer prevents leaks if a later alloc fails)
            const new_ctrl = try self.allocator.alloc(u8, new_capacity + group_width);
            errdefer self.allocator.free(new_ctrl);
            @memset(new_ctrl, CTRL_EMPTY);

            const new_keys = try self.allocator.alloc(K, new_capacity);
            errdefer self.allocator.free(new_keys);
            const new_values = try self.allocator.alloc(V, new_capacity);

            self.ctrl = new_ctrl;
            self.keys = new_keys;
            self.values = new_values;
            self.capacity = new_capacity;
            self.size = 0;
            self.growth_left = capacityToGrowth(new_capacity);

            // Re-insert old entries (direct insert, no capacity check needed)
            if (old_capacity > 0) {
                for (0..old_capacity) |i| {
                    if (!isEmptyOrDeleted(old_ctrl[i])) {
                        self.insertUnchecked(old_keys[i], old_values[i]);
                    }
                }

                self.allocator.free(old_ctrl[0 .. old_capacity + group_width]);
                self.allocator.free(old_keys[0..old_capacity]);
                self.allocator.free(old_values[0..old_capacity]);
            }
        }

        fn capacityToGrowth(cap: usize) usize {
            // 75% load factor
            return cap - cap / 4;
        }

        // ── Hashing ──────────────────────────────────────────────

        fn hashKey(self: *const Self, key: K) u64 {
            if (K == []const u8) {
                return wyhash(key, self.hash_seed);
            } else if (@typeInfo(K) == .int or @typeInfo(K) == .comptime_int) {
                return splitmix64(@as(u64, @intCast(key)) ^ self.hash_seed);
            } else {
                const bytes = std.mem.asBytes(&key);
                return wyhash(bytes, self.hash_seed);
            }
        }

        fn keysEqual(a: K, b: K) bool {
            if (K == []const u8) {
                return std.mem.eql(u8, a, b);
            } else {
                return a == b;
            }
        }

        // ── Hash Functions ───────────────────────────────────────

        fn splitmix64(input: u64) u64 {
            var x = input;
            x +%= 0x9e3779b97f4a7c15;
            x = (x ^ (x >> 30)) *% 0xbf58476d1ce4e5b9;
            x = (x ^ (x >> 27)) *% 0x94d049bb133111eb;
            return x ^ (x >> 31);
        }

        /// 128-bit widening multiply mixer (core wyhash primitive)
        inline fn wymix(a: u64, b: u64) u64 {
            const full = @as(u128, a) *% @as(u128, b);
            return @as(u64, @truncate(full)) ^ @as(u64, @truncate(full >> 64));
        }

        fn wyhash(data: []const u8, seed: u64) u64 {
            const secret = [4]u64{
                0xa076_1d64_78bd_642f,
                0xe703_7ed1_a0b4_28db,
                0x8ebc_6af0_9c88_c6e3,
                0x5899_65cc_7537_4cc3,
            };
            var hash: u64 = seed ^ secret[0];
            var i: usize = 0;

            // Process 8 bytes at a time with wymix
            while (i + 8 <= data.len) : (i += 8) {
                const word = std.mem.readInt(u64, data[i..][0..8], .little);
                hash = wymix(hash ^ word, secret[1]);
            }

            // Process remaining 1-7 bytes
            if (i < data.len) {
                var tail: u64 = 0;
                var shift: u6 = 0;
                for (data[i..]) |byte| {
                    tail |= @as(u64, byte) << shift;
                    shift +%= 8;
                }
                hash = wymix(hash ^ tail, secret[2]);
            }

            return wymix(hash, hash ^ secret[3]);
        }

        // ── Stats ────────────────────────────────────────────────

        pub const Stats = struct {
            size: usize,
            capacity: usize,
            load_factor: f64,
            empty_slots: usize,
            deleted_slots: usize,
        };

        pub fn stats(self: *const Self) Stats {
            var empty: usize = 0;
            var deleted: usize = 0;
            for (0..self.capacity) |i| {
                if (isEmpty(self.ctrl[i])) empty += 1;
                if (self.ctrl[i] == CTRL_DELETED) deleted += 1;
            }
            return .{
                .size = self.size,
                .capacity = self.capacity,
                .load_factor = if (self.capacity > 0)
                    @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(self.capacity))
                else
                    0,
                .empty_slots = empty,
                .deleted_slots = deleted,
            };
        }
    };
}

// ─── Convenience Aliases ─────────────────────────────────────────────────────

pub const StringMap = SwissMap([]const u8, []const u8);
pub const IntMap = SwissMap(u64, u64);

// ─── Tests ──────────────────────────────────────────────────────────────────

test "SIMD groupMatch basic" {
    var ctrl: [32]u8 = undefined;
    @memset(&ctrl, CTRL_EMPTY);
    ctrl[0] = 0x42;
    ctrl[5] = 0x42;
    ctrl[15] = 0x42;
    const mask = groupMatch(&ctrl, 0x42);
    try std.testing.expectEqual(@as(u16, (1 << 0) | (1 << 5) | (1 << 15)), mask);
}

test "SIMD groupMatchEmpty" {
    var ctrl: [32]u8 = undefined;
    @memset(&ctrl, 0x33); // all full
    ctrl[3] = CTRL_EMPTY;
    ctrl[10] = CTRL_EMPTY;
    const mask = groupMatchEmpty(&ctrl);
    try std.testing.expectEqual(@as(u16, (1 << 3) | (1 << 10)), mask);
}

test "SIMD groupMatchEmptyOrDeleted" {
    var ctrl: [32]u8 = undefined;
    @memset(&ctrl, 0x22); // all full
    ctrl[1] = CTRL_EMPTY;
    ctrl[7] = CTRL_DELETED;
    const mask = groupMatchEmptyOrDeleted(&ctrl);
    try std.testing.expectEqual(@as(u16, (1 << 1) | (1 << 7)), mask);
}

test "SwissMap delete then reinsert at tombstone" {
    const Map = SwissMap(u32, u32);
    var map = Map.init(std.testing.allocator);
    defer map.deinit();

    try map.put(1, 100);
    try map.put(2, 200);
    try std.testing.expect(map.remove(1));
    try std.testing.expectEqual(@as(?u32, null), map.get(1));
    try std.testing.expectEqual(@as(usize, 1), map.count());

    // Reinsert at tombstone
    try map.put(1, 999);
    try std.testing.expectEqual(@as(?u32, 999), map.get(1));
    try std.testing.expectEqual(@as(usize, 2), map.count());
}

test "SwissMap high load factor" {
    const Map = SwissMap(u32, u32);
    var map = try Map.initCapacity(std.testing.allocator, 16);
    defer map.deinit();

    // Fill near capacity (75% load factor threshold)
    for (0..200) |i| {
        try map.put(@intCast(i), @intCast(i * 10));
    }
    try std.testing.expectEqual(@as(usize, 200), map.count());

    // Verify all values retrievable
    for (0..200) |i| {
        const val = map.get(@intCast(i));
        try std.testing.expect(val != null);
        try std.testing.expectEqual(@as(u32, @intCast(i * 10)), val.?);
    }
}

test "SwissMap update existing key preserves count" {
    const Map = SwissMap(u32, u32);
    var map = Map.init(std.testing.allocator);
    defer map.deinit();

    try map.put(42, 1);
    try map.put(42, 2);
    try map.put(42, 3);
    try std.testing.expectEqual(@as(usize, 1), map.count());
    try std.testing.expectEqual(@as(?u32, 3), map.get(42));
}

test "SwissMap stats after mixed operations" {
    const Map = SwissMap(u32, u32);
    var map = Map.init(std.testing.allocator);
    defer map.deinit();

    for (0..10) |i| try map.put(@intCast(i), @intCast(i));
    for (0..5) |i| _ = map.remove(@intCast(i));

    const s = map.stats();
    try std.testing.expectEqual(@as(usize, 5), s.size);
    try std.testing.expectEqual(@as(usize, 5), s.deleted_slots);
    try std.testing.expect(s.load_factor < 0.5);
}

test "SwissMap getPtr mutation" {
    const Map = SwissMap(u32, u32);
    var map = Map.init(std.testing.allocator);
    defer map.deinit();

    try map.put(1, 100);
    if (map.getPtr(1)) |ptr| {
        ptr.* = 999;
    }
    try std.testing.expectEqual(@as(?u32, 999), map.get(1));
}

test "SwissMap seeded hash" {
    const Map = SwissMap(u32, u32);
    var map = Map.initWithSeed(std.testing.allocator, 0xDEADBEEF);
    defer map.deinit();

    try map.put(1, 10);
    try map.put(2, 20);
    try std.testing.expectEqual(@as(?u32, 10), map.get(1));
    try std.testing.expectEqual(@as(?u32, 20), map.get(2));
}

test {
    std.testing.refAllDecls(@This());
}
