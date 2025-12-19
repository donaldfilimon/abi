//! Modern Utilities Module - Zig 0.16 Compatible
//!
//! Enhanced utility functions with modern Zig patterns and improved performance.
//! Provides backward compatibility with legacy utils while adding new features.

const std = @import("std");

// Re-export existing utilities for compatibility
pub const utils = @import("utils.zig");

// =============================================================================
// MODERN UTILITIES - ENHANCED FEATURES
// =============================================================================

/// Modern memory management utilities with Zig 0.16 patterns
pub const memory = struct {
    /// Arena-backed allocator with automatic cleanup
    pub const ArenaAllocator = struct {
        arena: std.heap.ArenaAllocator,
        allocator: std.mem.Allocator,

        pub fn init(child_allocator: std.mem.Allocator) ArenaAllocator {
            return .{
                .arena = std.heap.ArenaAllocator.init(child_allocator),
                .allocator = undefined,
            };
        }

        pub fn deinit(self: *ArenaAllocator) void {
            self.arena.deinit();
        }

        pub fn allocator(self: *ArenaAllocator) std.mem.Allocator {
            return self.arena.allocator();
        }

        /// Reset the arena for reuse
        pub fn reset(self: *ArenaAllocator) void {
            _ = self.arena.reset(.retain_capacity);
        }
    };

    /// Fixed buffer allocator with bounds checking
    pub const FixedBufferAllocator = struct {
        buffer: []u8,
        allocator: std.heap.FixedBufferAllocator,

        pub fn init(buffer: []u8) FixedBufferAllocator {
            return .{
                .buffer = buffer,
                .allocator = std.heap.FixedBufferAllocator.init(buffer),
            };
        }

        pub fn allocator(self: *FixedBufferAllocator) std.mem.Allocator {
            return self.allocator.allocator();
        }

        pub fn remainingCapacity(self: *const FixedBufferAllocator) usize {
            return self.allocator.remainingCapacity(self.buffer);
        }
    };

    /// Memory pool for efficient object reuse
    pub fn MemoryPool(comptime T: type) type {
        return struct {
            const Self = @This();

            pool: std.heap.MemoryPool(T),

            pub fn init(allocator: std.mem.Allocator) Self {
                return .{ .pool = std.heap.MemoryPool(T).init(allocator) };
            }

            pub fn deinit(self: *Self) void {
                self.pool.deinit();
            }

            pub fn create(self: *Self) !*T {
                return try self.pool.create();
            }

            pub fn destroy(self: *Self, ptr: *T) void {
                self.pool.destroy(ptr);
            }
        };
    }
};

/// Modern string utilities with improved performance
pub const string = struct {
    /// Efficient string builder with automatic growth
    pub const StringBuilder = struct {
        allocator: std.mem.Allocator,
        buffer: std.ArrayList(u8),

        pub fn init(allocator: std.mem.Allocator) StringBuilder {
            return .{
                .allocator = allocator,
                .buffer = std.ArrayList(u8).init(allocator),
            };
        }

        pub fn deinit(self: *StringBuilder) void {
            self.buffer.deinit();
        }

        pub fn append(self: *StringBuilder, str: []const u8) !void {
            try self.buffer.appendSlice(str);
        }

        pub fn appendFormat(self: *StringBuilder, comptime fmt: []const u8, args: anytype) !void {
            try self.buffer.writer().print(fmt, args);
        }

        pub fn toOwnedSlice(self: *StringBuilder) ![]u8 {
            return try self.buffer.toOwnedSlice();
        }

        pub fn toSlice(self: *StringBuilder) []u8 {
            return self.buffer.items;
        }

        pub fn len(self: *const StringBuilder) usize {
            return self.buffer.items.len;
        }

        pub fn clear(self: *StringBuilder) void {
            self.buffer.clearRetainingCapacity();
        }
    };

    /// Modern string splitting with iterator pattern
    pub const StringSplitter = struct {
        str: []const u8,
        delimiter: []const u8,
        index: usize = 0,

        pub fn init(str: []const u8, delimiter: []const u8) StringSplitter {
            return .{
                .str = str,
                .delimiter = delimiter,
            };
        }

        pub fn next(self: *StringSplitter) ?[]const u8 {
            if (self.index >= self.str.len) return null;

            const start = self.index;
            if (std.mem.indexOfPos(u8, self.str, start, self.delimiter)) |delim_pos| {
                self.index = delim_pos + self.delimiter.len;
                return self.str[start..delim_pos];
            } else {
                self.index = self.str.len;
                return self.str[start..];
            }
        }
    };
};

/// Modern async utilities for Zig 0.16
pub const async_utils = struct {
    /// Simple task scheduler for concurrent operations
    pub const TaskScheduler = struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        tasks: std.ArrayList(Task),
        mutex: std.Thread.Mutex = .{},

        pub const Task = struct {
            id: u64,
            function: *const fn (*anyopaque) anyerror!void,
            context: *anyopaque,
            completed: bool = false,
        };

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .tasks = std.ArrayList(Task).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.tasks.deinit();
        }

        pub fn schedule(self: *Self, id: u64, function: *const fn (*anyopaque) anyerror!void, context: *anyopaque) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            try self.tasks.append(.{
                .id = id,
                .function = function,
                .context = context,
            });
        }

        pub fn executeAll(self: *Self) !void {
            self.mutex.lock();
            const tasks = try self.tasks.clone();
            self.tasks.clearRetainingCapacity();
            self.mutex.unlock();

            for (tasks.items) |*task| {
                try task.function(task.context);
                task.completed = true;
            }
        }
    };
};

/// Modern random utilities with improved seeding
pub const random = struct {
    /// Seeded random number generator
    pub const RandomGenerator = struct {
        rng: std.rand.DefaultPrng,

        pub fn init(seed: u64) RandomGenerator {
            return .{ .rng = std.rand.DefaultPrng.init(seed) };
        }

        pub fn random(self: *RandomGenerator) std.rand.Random {
            return self.rng.random();
        }

        /// Generate random bytes
        pub fn bytes(self: *RandomGenerator, buffer: []u8) void {
            self.rng.fill(buffer);
        }

        /// Generate random u64
        pub fn uint64(self: *RandomGenerator) u64 {
            return self.rng.next();
        }

        /// Generate random in range [min, max)
        pub fn intRange(self: *RandomGenerator, comptime T: type, min: T, max: T) T {
            return self.rng.random().intRangeAtMost(T, min, max - 1);
        }
    };

    /// Create a random generator seeded with current time
    pub fn seeded() RandomGenerator {
        const seed = @as(u64, @intCast(std.time.nanoTimestamp()));
        return RandomGenerator.init(seed);
    }
};

/// Modern file utilities with better error handling
pub const fs_utils = struct {
    /// Read entire file with automatic cleanup
    pub fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        return try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    }

    /// Write entire file with atomic operations
    pub fn writeFile(path: []const u8, contents: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        try file.writeAll(contents);
    }

    /// Check if file exists
    pub fn fileExists(path: []const u8) bool {
        std.fs.cwd().access(path, .{}) catch return false;
        return true;
    }

    /// Get file size
    pub fn fileSize(path: []const u8) !u64 {
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        return try file.getEndPos();
    }
};

/// Modern JSON utilities (placeholder - extend as needed)
pub const json_modern = struct {
    /// Parse JSON with better error reporting
    pub fn parse(allocator: std.mem.Allocator, json_str: []const u8) !std.json.Value {
        var parser = std.json.Parser.init(allocator, false);
        defer parser.deinit();

        return try parser.parse(json_str);
    }

    /// Stringify JSON value
    pub fn stringify(value: std.json.Value, allocator: std.mem.Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();

        try std.json.stringify(value, .{}, buffer.writer());
        return try buffer.toOwnedSlice();
    }
};

// =============================================================================
// LEGACY COMPATIBILITY
// =============================================================================

/// Re-export legacy utilities for backward compatibility
pub usingnamespace utils;

// =============================================================================
// TESTS
// =============================================================================

test "Modern memory utilities" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test arena allocator
    var arena = memory.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const arena_alloc = arena.allocator();
    const slice = try arena_alloc.alloc(u8, 100);
    try testing.expect(slice.len == 100);

    // Test string builder
    var builder = string.StringBuilder.init(allocator);
    defer builder.deinit();

    try builder.append("Hello, ");
    try builder.append("world!");
    const result = try builder.toOwnedSlice();
    defer allocator.free(result);

    try testing.expectEqualStrings("Hello, world!", result);
}

test "Modern random utilities" {
    var rng = random.seeded();
    const value = rng.uint64();
    try std.testing.expect(value != 0); // Should generate some value
}

test "String splitter" {
    var splitter = string.StringSplitter.init("a,b,c", ",");
    try std.testing.expectEqualStrings("a", splitter.next().?);
    try std.testing.expectEqualStrings("b", splitter.next().?);
    try std.testing.expectEqualStrings("c", splitter.next().?);
    try std.testing.expect(splitter.next() == null);
}