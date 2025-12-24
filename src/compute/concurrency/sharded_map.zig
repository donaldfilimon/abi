//! Sharded hash map for result caching
//!
//! Lock-based sharded map to reduce contention
//! for concurrent access to result cache.

const std = @import("std");

pub const ShardedMap = struct {
    shards: std.ArrayList(Shard),

    const Shard = struct {
        map: std.AutoHashMap(u64, void),
        mutex: std.Thread.Mutex,
    };

    pub fn init(allocator: std.mem.Allocator, shard_count: usize) !ShardedMap {
        var shards = try std.ArrayList(Shard).initCapacity(allocator, shard_count);
        errdefer shards.deinit();

        var i: usize = 0;
        while (i < shard_count) : (i += 1) {
            try shards.append(.{
                .map = std.AutoHashMap(u64, void).init(allocator),
                .mutex = std.Thread.Mutex{},
            });
        }

        return .{ .shards = shards };
    }

    pub fn deinit(self: *ShardedMap) void {
        for (self.shards.items) |*shard| {
            shard.map.deinit();
        }
        self.shards.deinit();
        self.* = undefined;
    }
};
