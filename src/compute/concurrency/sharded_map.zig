//! Sharded hash map for result caching
//!
//! Lock-based sharded map to reduce contention
//! for concurrent access to result cache.

const std = @import("std");

pub const ShardedMap = struct {
    shards: std.ArrayList(Shard),

    const Shard = struct {
        map: std.AutoHashMap(u64, u64),
        mutex: std.Thread.Mutex,
    };

    pub fn init(allocator: std.mem.Allocator, shard_count: usize) !ShardedMap {
        var shards = try std.ArrayList(Shard).initCapacity(allocator, shard_count);
        errdefer shards.deinit(allocator);

        var i: usize = 0;
        while (i < shard_count) : (i += 1) {
            try shards.append(allocator, .{
                .map = std.AutoHashMap(u64, u64).init(allocator),
                .mutex = std.Thread.Mutex{},
            });
        }

        return .{ .shards = shards };
    }

    pub fn deinit(self: *ShardedMap, allocator: std.mem.Allocator) void {
        for (self.shards.items) |*shard| {
            shard.map.deinit();
        }
        self.shards.deinit(allocator);
        self.* = undefined;
    }

    pub fn put(self: *ShardedMap, key: u64, value: u64) !void {
        const shard = self.getShard(key);
        shard.mutex.lock();
        defer shard.mutex.unlock();

        try shard.map.put(key, value);
    }

    pub fn get(self: *ShardedMap, key: u64) ?u64 {
        const shard = self.getShard(key);
        shard.mutex.lock();
        defer shard.mutex.unlock();

        return shard.map.get(key);
    }

    pub fn remove(self: *ShardedMap, key: u64) ?u64 {
        const shard = self.getShard(key);
        shard.mutex.lock();
        defer shard.mutex.unlock();

        if (shard.map.remove(key)) |value| {
            return value;
        }
        return null;
    }

    pub fn contains(self: *ShardedMap, key: u64) bool {
        const shard = self.getShard(key);
        shard.mutex.lock();
        defer shard.mutex.unlock();

        return shard.map.contains(key);
    }

    fn getShard(self: *ShardedMap, key: u64) *Shard {
        const shard_index = @mod(key, self.shards.items.len);
        return &self.shards.items[shard_index];
    }
};
