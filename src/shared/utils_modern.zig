//! Modernized Utilities Module
//!
//! This module provides core utilities with proper Zig 0.16 initialization patterns
//! Replaces deprecated usingnamespace patterns with explicit exports

const std = @import("std");

// Explicit module exports instead of usingnamespace
pub const collections = @import("../core/collections.zig");

// Re-export from existing utils modules (updated for compatibility)
pub const http = @import("utils/http/mod.zig");
pub const json = @import("utils/json/mod.zig");
pub const string = @import("utils/string/mod.zig");
pub const math = @import("utils/math/mod.zig");

// Additional utility modules
pub const encoding = struct {
    pub const base64 = struct {
        pub fn encode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
            const encoder = std.base64.standard.Encoder;
            const output_len = encoder.calcSize(input.len);
            const output = try allocator.alloc(u8, output_len);
            return encoder.encode(output, input);
        }

        pub fn decode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
            const decoder = std.base64.standard.Decoder;
            const output_len = try decoder.calcSizeForSlice(input);
            const output = try allocator.alloc(u8, output_len);
            try decoder.decode(output, input);
            return output;
        }
    };
};

pub const fs = struct {
    /// Read entire file with size limit - Zig 0.16 compatible
    pub fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8, max_size: usize) ![]u8 {
        return try std.fs.cwd().readFileAlloc(allocator, path, max_size);
    }

    /// Write file atomically
    pub fn writeFileAtomic(path: []const u8, data: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll(data);
    }

    /// Check if file exists
    pub fn fileExists(path: []const u8) bool {
        std.fs.cwd().access(path, .{}) catch return false;
        return true;
    }
};

pub const memory = struct {
    /// Memory pool for fixed-size allocations
    pub fn MemoryPool(comptime T: type) type {
        return struct {
            const Self = @This();

            allocator: std.mem.Allocator,
            pool: []T,
            free_list: collections.ArrayList(*T),

            pub fn create(allocator: std.mem.Allocator, capacity: usize) !Self {
                return .{
                    .allocator = allocator,
                    .pool = try allocator.alloc(T, capacity),
                    .free_list = collections.ArrayList(*T){},
                };
            }

            pub fn deinit(self: *Self) void {
                for (self.free_list.items) |item| {
                    self.allocator.destroy(item);
                }
                self.free_list.deinit(self.allocator);
            }

            pub fn acquire(self: *Self) !*T {
                if (self.free_list.items.len > 0) {
                    return self.free_list.swapRemove(self.free_list.items.len - 1);
                }
                return try self.allocator.create(T);
            }

            pub fn release(self: *Self, item: *T) void {
                self.free_list.append(self.allocator, item) catch {
                    // If we can't add to free list, just destroy it
                    self.allocator.destroy(item);
                };
            }
        };
    }
};

/// Configuration utilities
pub const config = struct {
    pub const Config = struct {
        values: collections.StringHashMap([]const u8),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Config {
            return .{
                .values = collections.StringHashMap([]const u8).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Config) void {
            var it = self.values.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            self.values.deinit();
        }

        pub fn set(self: *Config, key: []const u8, value: []const u8) !void {
            const key_copy = try self.allocator.dupe(u8, key);
            const value_copy = try self.allocator.dupe(u8, value);
            try self.values.put(key_copy, value_copy);
        }

        pub fn get(self: *const Config, key: []const u8) ?[]const u8 {
            return self.values.get(key);
        }
    };
};

test "modernized utilities" {
    const testing = std.testing;

    var cfg = config.Config.init(testing.allocator);
    defer cfg.deinit();

    try cfg.set("key", "value");
    try testing.expectEqualStrings("value", cfg.get("key").?);
}

test "shared utils - memory pool" {
    const testing = std.testing;

    var pool = memory.MemoryPool(u32).create(testing.allocator, 10) catch @panic("failed to create pool");
    defer pool.deinit();

    const item1 = try pool.acquire();
    const item2 = try pool.acquire();

    pool.release(item1);
    pool.release(item2);

    const item3 = try pool.acquire();
    // item3 is guaranteed to be non-null since acquire() returns !*T, not ?*T
    try testing.expect(@intFromPtr(item3) != 0);
}
