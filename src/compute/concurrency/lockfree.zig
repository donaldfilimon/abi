//! Concurrent containers for compute concurrency helpers.
//! Includes lock-free structures and sharded maps for high-performance scenarios.
//!
//! ## Thread Safety Notes
//!
//! ### LockFreeQueue
//! Single-producer single-consumer (SPSC) ring buffer. Safe for one producer
//! thread and one consumer thread.
//!
//! ### LockFreeStack (Treiber Stack)
//! **WARNING: ABA Problem**
//! This implementation is susceptible to the ABA problem under concurrent load.
//! The ABA problem occurs when:
//! 1. Thread A reads head pointer value P
//! 2. Thread B pops P, frees it, allocates new node at same address P
//! 3. Thread A's CAS succeeds (head still equals P) but P points to different data
//!
//! For production use with high concurrency, consider:
//! - Using hazard pointers or epoch-based reclamation
//! - Adding a generation counter to the pointer (tagged pointers)
//! - Using the ShardedMap instead for key-value storage
//!
//! ### ShardedMap
//! Thread-safe via per-shard mutex locking. Safe for multi-producer multi-consumer.
const std = @import("std");

pub fn LockFreeQueue(comptime T: type, comptime capacity: usize) type {
    comptime {
        if (capacity == 0 or !std.math.isPowerOfTwo(capacity)) {
            @compileError("capacity must be a non-zero power of two");
        }
    }

    return struct {
        buffer: [capacity]T = undefined,
        read_index: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        write_index: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        pub fn init() @This() {
            return .{};
        }

        pub fn push(self: *@This(), value: T) bool {
            const write = self.write_index.load(.acquire);
            const read = self.read_index.load(.acquire);
            const used = write -% read;
            if (used >= capacity) return false;

            self.buffer[write & (capacity - 1)] = value;
            self.write_index.store(write +% 1, .release);
            return true;
        }

        pub fn pop(self: *@This()) ?T {
            const read = self.read_index.load(.acquire);
            const write = self.write_index.load(.acquire);
            if (write == read) return null;

            const value = self.buffer[read & (capacity - 1)];
            self.read_index.store(read +% 1, .release);
            return value;
        }

        pub fn len(self: *@This()) usize {
            const read = self.read_index.load(.acquire);
            const write = self.write_index.load(.acquire);
            return write -% read;
        }

        pub fn isEmpty(self: *@This()) bool {
            return self.len() == 0;
        }
    };
}

pub fn LockFreeStack(comptime T: type) type {
    return struct {
        const Node = struct {
            value: T,
            next: ?*Node,
        };

        allocator: std.mem.Allocator,
        head: std.atomic.Value(?*Node) = std.atomic.Value(?*Node).init(null),

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *@This()) void {
            while (self.pop()) |_| {}
            self.* = undefined;
        }

        /// Treiber stack; lock-free but not ABA-safe without reclamation.
        pub fn push(self: *@This(), value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .value = value, .next = null };

            while (true) {
                const current = self.head.load(.acquire);
                node.next = current;
                if (self.head.cmpxchgWeak(current, node, .acq_rel, .acquire) ==
                    null)
                {
                    break;
                }
            }
        }

        /// Pop a value from the stack.
        /// WARNING: This operation is susceptible to the ABA problem. See module docs.
        /// Between loading `current` and the CAS, another thread could pop and free
        /// `current`, then push a new node that reuses the same memory address.
        /// The CAS would succeed incorrectly, potentially causing use-after-free.
        pub fn pop(self: *@This()) ?T {
            while (true) {
                const current = self.head.load(.acquire);
                if (current == null) return null;
                const next = current.?.next;

                if (self.head.cmpxchgWeak(current, next, .acq_rel, .acquire) ==
                    null)
                {
                    const value = current.?.value;
                    self.allocator.destroy(current.?);
                    return value;
                }
            }
        }
    };
}

pub fn ShardedMap(
    comptime K: type,
    comptime V: type,
    comptime shard_count: usize,
) type {
    comptime {
        if (shard_count == 0) {
            @compileError("shard_count must be greater than zero");
        }
    }

    return struct {
        const Context = std.hash_map.AutoContext(K);
        const Map = std.AutoHashMap(K, V);
        const Shard = struct {
            mutex: std.Thread.Mutex = .{},
            map: Map,
        };

        allocator: std.mem.Allocator,
        shards: [shard_count]Shard,

        pub fn init(allocator: std.mem.Allocator) @This() {
            var shards: [shard_count]Shard = undefined;
            for (&shards) |*shard| {
                shard.* = .{ .mutex = .{}, .map = Map.init(allocator) };
            }
            return .{ .allocator = allocator, .shards = shards };
        }

        pub fn deinit(self: *@This()) void {
            for (&self.shards) |*shard| {
                shard.map.deinit();
            }
            self.* = undefined;
        }

        fn shardIndex(key: K) usize {
            const hash = Context.hash(Context{}, key);
            return @intCast(hash % shard_count);
        }

        pub fn get(self: *@This(), key: K) ?V {
            const index = shardIndex(key);
            var shard = &self.shards[index];
            shard.mutex.lock();
            defer shard.mutex.unlock();
            return shard.map.get(key);
        }

        pub fn put(self: *@This(), key: K, value: V) !void {
            const index = shardIndex(key);
            var shard = &self.shards[index];
            shard.mutex.lock();
            defer shard.mutex.unlock();
            try shard.map.put(key, value);
        }

        pub fn remove(self: *@This(), key: K) bool {
            const index = shardIndex(key);
            var shard = &self.shards[index];
            shard.mutex.lock();
            defer shard.mutex.unlock();
            return shard.map.remove(key);
        }

        pub fn count(self: *@This()) usize {
            var total: usize = 0;
            for (&self.shards) |*shard| {
                shard.mutex.lock();
                total += shard.map.count();
                shard.mutex.unlock();
            }
            return total;
        }
    };
}

test "lock-free queue is FIFO" {
    var queue = LockFreeQueue(u32, 8).init();
    try std.testing.expect(queue.push(1));
    try std.testing.expect(queue.push(2));
    try std.testing.expect(queue.push(3));

    try std.testing.expectEqual(@as(?u32, 1), queue.pop());
    try std.testing.expectEqual(@as(?u32, 2), queue.pop());
    try std.testing.expectEqual(@as(?u32, 3), queue.pop());
    try std.testing.expectEqual(@as(?u32, null), queue.pop());
}

test "lock-free stack is LIFO" {
    var stack = LockFreeStack(u32).init(std.testing.allocator);
    defer stack.deinit();

    try stack.push(10);
    try stack.push(20);
    try stack.push(30);

    try std.testing.expectEqual(@as(?u32, 30), stack.pop());
    try std.testing.expectEqual(@as(?u32, 20), stack.pop());
    try std.testing.expectEqual(@as(?u32, 10), stack.pop());
    try std.testing.expectEqual(@as(?u32, null), stack.pop());
}

test "sharded map stores entries" {
    var map = ShardedMap(u32, u32, 4).init(std.testing.allocator);
    defer map.deinit();

    try map.put(1, 10);
    try map.put(2, 20);
    try std.testing.expectEqual(@as(?u32, 10), map.get(1));
    try std.testing.expectEqual(@as(?u32, 20), map.get(2));

    try std.testing.expect(map.remove(1));
    try std.testing.expect(map.get(1) == null);
    try std.testing.expectEqual(@as(usize, 1), map.count());
}

test "concurrent sharded map stress test" {
    const thread_count = 4;
    const ops_per_thread = 100;

    var map = ShardedMap(u32, u32, 8).init(std.testing.allocator);
    defer map.deinit();

    var threads: [thread_count]std.Thread = undefined;
    var successes: [thread_count]std.atomic.Value(usize) = undefined;

    for (&threads, 0..) |*t, thread_id| {
        successes[thread_id] = std.atomic.Value(usize).init(0);
        t.* = try std.Thread.spawn(.{}, struct {
            fn worker(map_ptr: *ShardedMap(u32, u32, 8), tid: usize, success_ptr: *std.atomic.Value(usize)) !void {
                var idx: u32 = 0;
                while (idx < ops_per_thread) : (idx += 1) {
                    const key = tid * 1000 + idx;
                    try map_ptr.put(key, key * 2);
                    success_ptr.fetchAdd(1, .monotonic);

                    const val = map_ptr.get(key);
                    try std.testing.expect(val != null);
                    try std.testing.expectEqual(key * 2, val.?);
                }
            }
        }.worker, .{ &map, thread_id, &successes[thread_id] });
    }

    for (&threads) |*t| {
        t.join();
    }

    var total_ops: usize = 0;
    for (&successes) |*s| {
        total_ops += s.load(.monotonic);
    }

    try std.testing.expectEqual(thread_count * ops_per_thread, total_ops);
    try std.testing.expectEqual(thread_count * ops_per_thread, map.count());
}

test "lock-free queue basic invariants" {
    const TestQueue = LockFreeQueue(u32, 16);

    var queue = TestQueue.init();
    try std.testing.expectEqual(@as(usize, 0), queue.len());
    try std.testing.expect(queue.isEmpty());

    const test_values = [_]u32{ 1, 2, 3, 4, 5 };

    for (test_values) |value| {
        const pushed = queue.push(value);
        try std.testing.expect(pushed);
    }

    try std.testing.expectEqual(@as(usize, 5), queue.len());
    try std.testing.expect(!queue.isEmpty());

    const popped1 = queue.pop();
    try std.testing.expect(popped1 != null);
    try std.testing.expectEqual(@as(u32, 1), popped1.?);
    try std.testing.expectEqual(@as(usize, 4), queue.len());

    for (0..3) |_| {
        _ = queue.pop();
    }

    try std.testing.expectEqual(@as(usize, 1), queue.len());

    const popped_last = queue.pop();
    try std.testing.expect(popped_last != null);
    try std.testing.expectEqual(@as(u32, 5), popped_last.?);
    try std.testing.expectEqual(@as(usize, 0), queue.len());
    try std.testing.expect(queue.isEmpty());
}
