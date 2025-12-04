//! Common Patterns and Utilities Module
//! Consolidates frequently used patterns across the codebase

const std = @import("std");
const collections = @import("../core/collections.zig");

/// Standard initialization patterns for common data structures
pub const InitPatterns = struct {
    /// Initialize an ArrayList with common error handling
    pub fn arrayList(comptime T: type) collections.ArrayList(T) {
        return collections.ArrayList(T){};
    }

    /// Initialize a StringHashMap with allocator
    pub fn stringHashMap(comptime V: type, allocator: std.mem.Allocator) collections.StringHashMap(V) {
        return collections.StringHashMap(V).init(allocator);
    }

    /// Initialize an AutoHashMap with allocator
    pub fn autoHashMap(comptime K: type, comptime V: type, allocator: std.mem.Allocator) collections.AutoHashMap(K, V) {
        return collections.AutoHashMap(K, V).init(allocator);
    }
};

/// Common cleanup patterns
pub const CleanupPatterns = struct {
    /// Clean up a string hash map with allocated keys and values
    pub fn stringHashMapWithAllocatedStrings(map: *collections.StringHashMap([]const u8), allocator: std.mem.Allocator) void {
        var it = map.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        map.deinit();
    }

    /// Clean up an ArrayList and its contents
    pub fn arrayListWithAllocatedItems(comptime T: type, list: *collections.ArrayList(T), allocator: std.mem.Allocator) void {
        for (list.items) |item| {
            if (@TypeOf(item) == []u8 or @TypeOf(item) == []const u8) {
                allocator.free(item);
            }
        }
        list.deinit(allocator);
    }
};

/// Factory patterns for common objects
pub const FactoryPatterns = struct {
    /// Generic factory for types with init/deinit pattern
    pub fn createManaged(comptime T: type, allocator: std.mem.Allocator, args: anytype) !*T {
        const instance = try allocator.create(T);
        errdefer allocator.destroy(instance);

        instance.* = try @call(.auto, T.init, .{allocator} ++ args);
        return instance;
    }

    /// Factory for types that need allocator storage
    pub fn createWithAllocator(comptime T: type, allocator: std.mem.Allocator, args: anytype) !*T {
        const instance = try allocator.create(T);
        errdefer allocator.destroy(instance);

        instance.* = @call(.auto, T.init, .{allocator} ++ args);
        return instance;
    }
};

/// Common test patterns
pub const TestPatterns = struct {
    /// Standard test setup with allocator
    pub const TestSetup = struct {
        allocator: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,

        pub fn init() TestSetup {
            const testing = std.testing;
            var arena = std.heap.ArenaAllocator.init(testing.allocator);
            return .{
                .allocator = arena.allocator(),
                .arena = arena,
            };
        }

        pub fn deinit(self: *TestSetup) void {
            self.arena.deinit();
        }
    };

    /// Common assertions for collections
    pub fn expectCollection(comptime T: type, expected: []const T, actual: []const T) !void {
        try std.testing.expectEqual(expected.len, actual.len);
        for (expected, actual) |exp, act| {
            try std.testing.expectEqual(exp, act);
        }
    }
};

/// Performance monitoring patterns
pub const MonitoringPatterns = struct {
    /// Simple performance timer
    pub const PerfTimer = struct {
        start_time: i128,

        pub fn start() PerfTimer {
            return .{ .start_time = std.time.nanoTimestamp() };
        }

        pub fn elapsedNs(self: *const PerfTimer) u64 {
            const now = std.time.nanoTimestamp();
            return @intCast(now - self.start_time);
        }

        pub fn elapsedMs(self: *const PerfTimer) f64 {
            return @as(f64, @floatFromInt(self.elapsedNs())) / 1_000_000.0;
        }
    };

    /// Memory usage tracker
    pub const MemoryUsage = struct {
        initial_usage: usize,

        pub fn start() MemoryUsage {
            return .{ .initial_usage = getCurrentMemoryUsage() };
        }

        pub fn delta(self: *const MemoryUsage) isize {
            const current = getCurrentMemoryUsage();
            return @as(isize, @intCast(current)) - @as(isize, @intCast(self.initial_usage));
        }

        fn getCurrentMemoryUsage() usize {
            // Placeholder - in real implementation would use OS-specific APIs
            return 0;
        }
    };
};

test "common patterns - initialization" {
    const testing = std.testing;

    var list = InitPatterns.arrayList(u32);
    defer list.deinit(testing.allocator);

    try list.append(testing.allocator, 42);
    try testing.expectEqual(@as(usize, 1), list.items.len);

    var map = InitPatterns.stringHashMap(u32, testing.allocator);
    defer map.deinit();

    try map.put("answer", 42);
    try testing.expectEqual(@as(u32, 42), map.get("answer").?);
}

test "common patterns - test setup" {
    var setup = TestPatterns.TestSetup.init();
    defer setup.deinit();

    const data = try setup.allocator.alloc(u32, 5);
    @memset(data, 42);

    try TestPatterns.expectCollection(u32, &[_]u32{42} ** 5, data);
}
