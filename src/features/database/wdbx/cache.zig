//! Embedding vector look-back Cache
//!
//! Stores previously generated embeddings corresponding to explicit
//! text data or UUID references, drastically stripping duplicate calls
//! made to outer neural services boundaries.

const std = @import("std");

pub const Cache = struct {
    allocator: std.mem.Allocator,
    capacity: usize,
    segments: u8,
    map: std.StringHashMapUnmanaged([]const f32),
    mutex: std.Thread.RwLock,

    pub fn init(allocator: std.mem.Allocator, capacity: usize, segments: u8) !Cache {
        return .{
            .allocator = allocator,
            .capacity = capacity,
            .segments = segments, // For future sharding support
            .map = .empty,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Cache) void {
        var it = self.map.valueIterator();
        while (it.next()) |val| {
            self.allocator.free(val.*);
        }

        var iter = self.map.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }

        self.map.deinit(self.allocator);
    }

    pub fn get(self: *Cache, text: []const u8) ?[]const f32 {
        self.mutex.lockShared();
        defer self.mutex.unlockShared();

        return self.map.get(text);
    }

    pub fn put(self: *Cache, text: []const u8, embedding: []const f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.map.count() >= self.capacity) {
            // Simplified eviction mapping: remove first retrieved
            var key_it = self.map.keyIterator();
            if (key_it.next()) |key| {
                if (self.map.get(key.*)) |val| self.allocator.free(val);
                _ = self.map.remove(key.*);
                self.allocator.free(key.*);
            }
        }

        const cloned_text = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(cloned_text);

        const cloned_embedding = try self.allocator.dupe(f32, embedding);
        try self.map.put(self.allocator, cloned_text, cloned_embedding);
    }
};

test "Cache memory map basic test" {
    var cache = try Cache.init(std.testing.allocator, 10, 1);
    defer cache.deinit();

    const text = "sample query";
    const embed = [_]f32{ 0.5, 0.5 };

    try cache.put(text, &embed);
    const retrieved = cache.get(text).?;

    try std.testing.expectEqual(@as(usize, 2), retrieved.len);
    try std.testing.expectEqual(@as(f32, 0.5), retrieved[0]);
}
