//! Generic Cache Utility
//!
//! Provides reusable caching with eviction policies and validation.

const std = @import("std");

pub const CacheError = error{
    InvalidKey,
    KeyTooLong,
    OutOfMemory,
    PathTraversal,
};

/// Eviction policy
pub const EvictionPolicy = enum {
    lru,
    fifo,
    none,
};

/// Generic cache with configurable eviction
pub fn Cache(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        entries: std.AutoHashMapUnmanaged(K, Entry),
        order: std.ArrayListUnmanaged(K),
        allocator: std.mem.Allocator,
        max_entries: usize,
        policy: EvictionPolicy,
        hits: u64 = 0,
        misses: u64 = 0,

        const Entry = struct {
            value: V,
            access_count: u32,
        };

        pub fn init(allocator: std.mem.Allocator, max_entries: usize, policy: EvictionPolicy) Self {
            return .{
                .entries = std.AutoHashMapUnmanaged(K, Entry).empty,
                .order = std.ArrayListUnmanaged(K).empty,
                .allocator = allocator,
                .max_entries = max_entries,
                .policy = policy,
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
            self.order.deinit(self.allocator);
        }

        pub fn get(self: *Self, key: K) ?V {
            if (self.entries.getPtr(key)) |entry| {
                self.hits += 1;
                entry.access_count += 1;
                if (self.policy == .lru) {
                    self.moveToEnd(key);
                }
                return entry.value;
            }
            self.misses += 1;
            return null;
        }

        pub fn put(self: *Self, key: K, value: V) !void {
            if (self.entries.contains(key)) {
                self.entries.getPtr(key).?.value = value;
                return;
            }

            if (self.entries.count() >= self.max_entries) {
                try self.evict();
            }

            try self.entries.put(self.allocator, key, .{ .value = value, .access_count = 1 });
            try self.order.append(self.allocator, key);
        }

        pub fn remove(self: *Self, key: K) bool {
            if (self.entries.remove(key)) {
                self.removeFromOrder(key);
                return true;
            }
            return false;
        }

        pub fn clear(self: *Self) void {
            self.entries.clearRetainingCapacity();
            self.order.clearRetainingCapacity();
        }

        pub fn count(self: *const Self) usize {
            return self.entries.count();
        }

        pub fn hitRate(self: *const Self) f64 {
            const total = self.hits + self.misses;
            if (total == 0) return 0.0;
            return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
        }

        fn evict(self: *Self) !void {
            if (self.order.items.len == 0) return;

            const key = switch (self.policy) {
                .fifo, .lru => self.order.items[0],
                .none => return,
            };

            _ = self.entries.remove(key);
            _ = self.order.orderedRemove(0);
        }

        fn moveToEnd(self: *Self, key: K) void {
            for (self.order.items, 0..) |k, i| {
                if (std.meta.eql(k, key)) {
                    _ = self.order.orderedRemove(i);
                    self.order.append(self.allocator, key) catch {};
                    break;
                }
            }
        }

        fn removeFromOrder(self: *Self, key: K) void {
            for (self.order.items, 0..) |k, i| {
                if (std.meta.eql(k, key)) {
                    _ = self.order.orderedRemove(i);
                    break;
                }
            }
        }
    };
}

/// String-keyed cache with validation
pub fn StringCache(comptime V: type) type {
    return struct {
        const Self = @This();

        entries: std.StringHashMapUnmanaged(Entry),
        order: std.ArrayListUnmanaged([]const u8),
        allocator: std.mem.Allocator,
        max_entries: usize,
        max_key_len: usize,
        policy: EvictionPolicy,

        const Entry = struct {
            value: V,
            owned_key: []u8,
        };

        pub fn init(allocator: std.mem.Allocator, max_entries: usize, max_key_len: usize) Self {
            return .{
                .entries = std.StringHashMapUnmanaged(Entry).empty,
                .order = std.ArrayListUnmanaged([]const u8).empty,
                .allocator = allocator,
                .max_entries = max_entries,
                .max_key_len = max_key_len,
                .policy = .lru,
            };
        }

        pub fn deinit(self: *Self) void {
            var it = self.entries.valueIterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.owned_key);
            }
            self.entries.deinit(self.allocator);
            self.order.deinit(self.allocator);
        }

        pub fn get(self: *Self, key: []const u8) ?V {
            if (self.entries.get(key)) |entry| {
                return entry.value;
            }
            return null;
        }

        pub fn put(self: *Self, key: []const u8, value: V) CacheError!void {
            if (!isValidKey(key, self.max_key_len)) {
                return CacheError.InvalidKey;
            }

            if (self.entries.contains(key)) {
                self.entries.getPtr(key).?.value = value;
                return;
            }

            if (self.entries.count() >= self.max_entries) {
                self.evict();
            }

            const owned_key = self.allocator.dupe(u8, key) catch return CacheError.OutOfMemory;
            errdefer self.allocator.free(owned_key);

            self.entries.put(self.allocator, owned_key, .{
                .value = value,
                .owned_key = owned_key,
            }) catch return CacheError.OutOfMemory;

            self.order.append(self.allocator, owned_key) catch return CacheError.OutOfMemory;
        }

        pub fn remove(self: *Self, key: []const u8) bool {
            if (self.entries.fetchRemove(key)) |kv| {
                self.removeFromOrder(kv.value.owned_key);
                self.allocator.free(kv.value.owned_key);
                return true;
            }
            return false;
        }

        fn evict(self: *Self) void {
            if (self.order.items.len == 0) return;

            const key = self.order.items[0];
            if (self.entries.fetchRemove(key)) |kv| {
                self.allocator.free(kv.value.owned_key);
            }
            _ = self.order.orderedRemove(0);
        }

        fn removeFromOrder(self: *Self, key: []const u8) void {
            for (self.order.items, 0..) |k, i| {
                if (std.mem.eql(u8, k, key)) {
                    _ = self.order.orderedRemove(i);
                    break;
                }
            }
        }
    };
}

/// Validate cache key (prevents path traversal, etc.)
pub fn isValidKey(key: []const u8, max_len: usize) bool {
    if (key.len == 0 or key.len > max_len) return false;
    if (std.mem.indexOf(u8, key, "..") != null) return false;
    if (key[0] == '/' or key[0] == '\\') return false;

    // Check for invalid characters
    for (key) |c| {
        if (c < 32 or c == 127) return false;
        if (c == '<' or c == '>' or c == ':' or c == '"' or c == '|' or c == '?' or c == '*') return false;
    }

    return true;
}

/// Sanitize a filename for safe use
pub fn sanitizeFilename(name: []const u8, buffer: []u8) []u8 {
    var len: usize = 0;
    for (name) |c| {
        if (len >= buffer.len) break;
        if (c >= 'a' and c <= 'z' or c >= 'A' and c <= 'Z' or c >= '0' and c <= '9' or c == '_' or c == '-' or c == '.') {
            buffer[len] = c;
            len += 1;
        }
    }
    return buffer[0..len];
}

test "cache basic" {
    const allocator = std.testing.allocator;

    var cache = Cache(u32, []const u8).init(allocator, 3, .lru);
    defer cache.deinit();

    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    try std.testing.expectEqualStrings("one", cache.get(1).?);
    try std.testing.expectEqualStrings("two", cache.get(2).?);

    // Eviction
    try cache.put(4, "four");
    try std.testing.expect(cache.get(3) == null); // LRU evicted

    try std.testing.expectEqual(@as(usize, 3), cache.count());
}

test "key validation" {
    try std.testing.expect(isValidKey("valid_key", 256));
    try std.testing.expect(!isValidKey("", 256));
    try std.testing.expect(!isValidKey("../path", 256));
    try std.testing.expect(!isValidKey("/absolute", 256));
    try std.testing.expect(!isValidKey("has\x00null", 256));
}

test "sanitize filename" {
    var buf: [64]u8 = undefined;
    const result = sanitizeFilename("file<name>.txt", &buf);
    try std.testing.expectEqualStrings("filename.txt", result);
}
