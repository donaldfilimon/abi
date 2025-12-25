//! Sharded hash map for concurrent access
//!
//! Lock-based sharded map to reduce contention.
//! Uses generic keys and values with proper hash distribution.

const std = @import("std");

pub fn ShardedMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();
        const Shard = struct {
            map: std.AutoHashMap(K, V),
            mutex: std.Thread.Mutex,
        };

        shards: []Shard,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, shard_count: usize) !Self {
            const shards = try allocator.alloc(Shard, shard_count);
            for (shards) |*shard| {
                shard.* = .{
                    .map = std.AutoHashMap(K, V).init(allocator),
                    .mutex = std.Thread.Mutex{},
                };
            }

            return .{
                .shards = shards,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.shards) |*shard| {
                shard.map.deinit();
            }
            self.allocator.free(self.shards);
            self.* = undefined;
        }

        pub fn put(self: *Self, key: K, value: V) !void {
            const shard = self.getShard(key);
            shard.mutex.lock();
            defer shard.mutex.unlock();
            try shard.map.put(key, value);
        }

        pub fn get(self: *Self, key: K) ?V {
            const shard = self.getShard(key);
            shard.mutex.lock();
            defer shard.mutex.unlock();
            return shard.map.get(key);
        }

        pub fn remove(self: *Self, key: K) ?V {
            const shard = self.getShard(key);
            shard.mutex.lock();
            defer shard.mutex.unlock();
            if (shard.map.fetchRemove(key)) |kv| {
                return kv.value;
            }
            return null;
        }

        pub fn contains(self: *Self, key: K) bool {
            const shard = self.getShard(key);
            shard.mutex.lock();
            defer shard.mutex.unlock();
            return shard.map.contains(key);
        }

        pub fn fetchRemoveIf(self: *Self, key: K, context: anytype, predicate: fn (@TypeOf(context), V) bool) ?V {
            const shard = self.getShard(key);
            shard.mutex.lock();
            defer shard.mutex.unlock();

            if (shard.map.get(key)) |val| {
                if (predicate(context, val)) {
                    _ = shard.map.remove(key);
                    return val;
                }
            }
            return null;
        }

        fn getShard(self: *Self, key: K) *Shard {
            // Use Wyhash for better distribution than simple modulo
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key);
            const hash = hasher.final();
            const index = hash % self.shards.len;
            return &self.shards[index];
        }
    };
}
