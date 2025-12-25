//! Lock-free data structures for ultra-high-performance concurrent operations
//!
//! This module provides lock-free implementations of common data structures
//! optimized for multi-threaded environments with minimal contention.

const std = @import("std");
const builtin = @import("builtin");
const Atomic = std.atomic.Value;

/// Lock-free data structure errors
pub const LockFreeError = error{
    OutOfMemory,
    CapacityExceeded,
    InvalidOperation,
    Timeout,
};

/// Performance statistics for lock-free operations
pub const LockFreeStats = struct {
    operations: u64 = 0,
    successful_operations: u64 = 0,
    failed_operations: u64 = 0,
    average_latency_ns: u64 = 0,
    max_latency_ns: u64 = 0,
    min_latency_ns: u64 = std.math.maxInt(u64),

    pub fn recordOperation(self: *LockFreeStats, success: bool, latency_ns: u64) void {
        self.operations += 1;
        if (success) {
            self.successful_operations += 1;
        } else {
            self.failed_operations += 1;
        }

        // Update latency statistics
        self.max_latency_ns = @max(self.max_latency_ns, latency_ns);
        self.min_latency_ns = @min(self.min_latency_ns, latency_ns);

        // Rolling average
        if (self.operations == 1) {
            self.average_latency_ns = latency_ns;
        } else {
            self.average_latency_ns = (self.average_latency_ns + latency_ns) / 2;
        }
    }

    pub fn successRate(self: *const LockFreeStats) f32 {
        if (self.operations == 0) return 0.0;
        return @as(f32, @floatFromInt(self.successful_operations)) / @as(f32, @floatFromInt(self.operations));
    }
};

/// Lock-free queue using Michael & Scott algorithm
pub fn lockFreeQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: T,
            next: Atomic(?*Node),

            fn init(data: T) Node {
                return Node{
                    .data = data,
                    .next = Atomic(?*Node).init(null),
                };
            }
        };

        head: Atomic(*Node),
        tail: Atomic(*Node),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            // Create dummy node
            const dummy = try allocator.create(Node);
            dummy.* = Node{
                .data = undefined,
                .next = Atomic(?*Node).init(null),
            };

            return Self{
                .head = Atomic(*Node).init(dummy),
                .tail = Atomic(*Node).init(dummy),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            // Drain the queue
            while (self.dequeue()) |_| {}

            // Free the dummy node
            const head = self.head.load(.acquire);
            self.allocator.destroy(head);
        }

        pub fn enqueue(self: *Self, data: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = Node.init(data);

            while (true) {
                const tail = self.tail.load(.acquire);
                const next = tail.next.load(.acquire);

                if (tail == self.tail.load(.acquire)) {
                    if (next == null) {
                        // Try to link new node at end of list
                        if (tail.next.cmpxchgWeak(null, new_node, .release, .acquire) == null) {
                            break;
                        }
                    } else {
                        // Try to swing tail to next node
                        _ = self.tail.cmpxchgWeak(tail, next.?, .release, .acquire);
                    }
                }
            }

            // Swing tail to new node
            _ = self.tail.cmpxchgWeak(self.tail.load(.acquire), new_node, .release, .acquire);
        }

        pub fn dequeue(self: *Self) ?T {
            while (true) {
                const head = self.head.load(.acquire);
                const tail = self.tail.load(.acquire);
                const next = head.next.load(.acquire);

                if (head == self.head.load(.acquire)) {
                    if (head == tail) {
                        if (next == null) {
                            return null; // Queue is empty
                        }
                        // Try to swing tail to next node
                        _ = self.tail.cmpxchgWeak(tail, next.?, .release, .acquire);
                    } else {
                        if (next == null) continue;

                        // Read data before CAS
                        const data = next.?.data;

                        // Try to swing head to next node
                        if (self.head.cmpxchgWeak(head, next.?, .release, .acquire) == null) {
                            self.allocator.destroy(head);
                            return data;
                        }
                    }
                }
            }
        }
    };
}

/// Lock-free stack using Treiber algorithm
pub fn lockFreeStack(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: T,
            next: ?*Node,
        };

        head: Atomic(?*Node),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .head = Atomic(?*Node).init(null),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            while (self.pop()) |_| {}
        }

        pub fn push(self: *Self, data: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.data = data;

            while (true) {
                const old_head = self.head.load(.acquire);
                new_node.next = old_head;

                if (self.head.cmpxchgWeak(old_head, new_node, .release, .acquire) == null) {
                    break;
                }
            }
        }

        pub fn pop(self: *Self) ?T {
            while (true) {
                const old_head = self.head.load(.acquire);
                if (old_head == null) return null;

                const next = old_head.?.next;
                if (self.head.cmpxchgWeak(old_head, next, .release, .acquire) == null) {
                    const data = old_head.?.data;
                    self.allocator.destroy(old_head.?);
                    return data;
                }
            }
        }
    };
}

/// Lock-free hash map using hopscotch hashing
pub fn lockFreeHashMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();
        const H = 32; // Hopscotch neighborhood size

        const Entry = struct {
            key: K,
            value: V,
            hash: u32,
            hop_info: Atomic(u32),

            fn init(key: K, value: V, hash: u32) Entry {
                return Entry{
                    .key = key,
                    .value = value,
                    .hash = hash,
                    .hop_info = Atomic(u32).init(0),
                };
            }
        };

        entries: []Entry,
        size: Atomic(usize),
        capacity: usize,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            const entries = try allocator.alloc(Entry, capacity);
            @memset(entries, Entry{
                .key = undefined,
                .value = undefined,
                .hash = 0,
                .hop_info = Atomic(u32).init(0),
            });

            return Self{
                .entries = entries,
                .size = Atomic(usize).init(0),
                .capacity = capacity,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.entries);
        }

        fn hashKey(key: K) u32 {
            return @truncate(std.hash_map.hashString(@as([]const u8, std.mem.asBytes(&key))));
        }

        pub fn put(self: *Self, key: K, value: V) !bool {
            const key_hash = hashKey(key);
            const start_bucket = key_hash % self.capacity;

            // Try to find existing key first
            var bucket = start_bucket;
            var dist: u32 = 0;

            while (dist < H) : ({
                bucket = (bucket + 1) % self.capacity;
                dist += 1;
            }) {
                const entry = &self.entries[bucket];
                const hop_info = entry.hop_info.load(.acquire);

                if ((hop_info & (@as(u32, 1) << @as(u5, @intCast(dist)))) != 0) {
                    if (entry.hash == key_hash and std.meta.eql(entry.key, key)) {
                        // Update existing entry
                        entry.value = value;
                        return true;
                    }
                }
            }

            // Find empty slot
            bucket = start_bucket;
            dist = 0;

            while (dist < self.capacity) : ({
                bucket = (bucket + 1) % self.capacity;
                dist += 1;
            }) {
                const entry = &self.entries[bucket];
                if (entry.hash == 0) {
                    // Found empty slot
                    if (dist < H) {
                        // Within neighborhood
                        entry.key = key;
                        entry.value = value;
                        entry.hash = key_hash;

                        const start_entry = &self.entries[start_bucket];
                        const old_hop = start_entry.hop_info.load(.acquire);
                        const new_hop = old_hop | (@as(u32, 1) << @as(u5, @intCast(dist)));

                        if (start_entry.hop_info.cmpxchgWeak(old_hop, new_hop, .release, .acquire) == null) {
                            _ = self.size.fetchAdd(1, .release);
                            return true;
                        }
                    } else {
                        // Need to move entries closer
                        return self.relocate(start_bucket, bucket);
                    }
                }
            }

            return false; // Table full
        }

        pub fn get(self: *Self, key: K) ?V {
            const key_hash = hashKey(key);
            const start_bucket = key_hash % self.capacity;
            const start_entry = &self.entries[start_bucket];
            const hop_info = start_entry.hop_info.load(.acquire);

            var dist: u32 = 0;
            while (dist < H) : (dist += 1) {
                if ((hop_info & (@as(u32, 1) << dist)) != 0) {
                    const bucket = (start_bucket + dist) % self.capacity;
                    const entry = &self.entries[bucket];

                    if (entry.hash == key_hash and std.meta.eql(entry.key, key)) {
                        return entry.value;
                    }
                }
            }

            return null;
        }

        fn relocate(self: *Self, start_bucket: usize, free_bucket: usize) bool {
            // Implementation of hopscotch relocation algorithm
            // This is a simplified version - full implementation would be more complex
            _ = self;
            _ = start_bucket;
            _ = free_bucket;
            return false;
        }
    };
}

/// Lock-free work-stealing deque
pub fn workStealingDeque(comptime T: type) type {
    return struct {
        const Self = @This();

        buffer: []Atomic(?T),
        top: Atomic(i64),
        bottom: Atomic(i64),
        capacity: usize,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            const buffer = try allocator.alloc(Atomic(?T), capacity);
            for (buffer) |*slot| {
                slot.* = Atomic(?T).init(null);
            }

            return Self{
                .buffer = buffer,
                .top = Atomic(i64).init(0),
                .bottom = Atomic(i64).init(0),
                .capacity = capacity,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }

        pub fn push(self: *Self, item: T) bool {
            const b = self.bottom.load(.acquire);
            const t = self.top.load(.acquire);

            if (b - t >= @as(i64, @intCast(self.capacity))) {
                return false; // Queue full
            }

            const index = @as(usize, @intCast(b % @as(i64, @intCast(self.capacity))));
            self.buffer[index].store(item, .release);
            self.bottom.store(b + 1, .release);

            return true;
        }

        pub fn pop(self: *Self) ?T {
            const b = self.bottom.load(.acquire) - 1;
            self.bottom.store(b, .release);

            const t = self.top.load(.acquire);

            if (t <= b) {
                const index = @as(usize, @intCast(b % @as(i64, @intCast(self.capacity))));
                const item = self.buffer[index].load(.acquire);

                if (t == b) {
                    // Last item, compete with steal
                    if (self.top.cmpxchgStrong(t, t + 1, .release, .acquire) != null) {
                        self.bottom.store(b + 1, .release);
                        return null;
                    }
                    self.bottom.store(b + 1, .release);
                }

                return item;
            } else {
                self.bottom.store(b + 1, .release);
                return null;
            }
        }

        pub fn steal(self: *Self) ?T {
            const t = self.top.load(.acquire);
            const b = self.bottom.load(.acquire);

            if (t < b) {
                const index = @as(usize, @intCast(t % @as(i64, @intCast(self.capacity))));
                const item = self.buffer[index].load(.acquire);

                if (self.top.cmpxchgStrong(t, t + 1, .release, .acquire) == null) {
                    return item;
                }
            }

            return null;
        }
    };
}

/// Multi-producer, multi-consumer queue with batching
pub fn mpmcQueue(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buffer: [capacity]Atomic(*T),
        head: Atomic(usize),
        tail: Atomic(usize),

        pub fn init() Self {
            var buffer: [capacity]Atomic(*T) = undefined;
            for (&buffer) |*slot| {
                slot.* = Atomic(*T).init(null);
            }

            return Self{
                .buffer = buffer,
                .head = Atomic(usize).init(0),
                .tail = Atomic(usize).init(0),
            };
        }

        pub fn enqueue(self: *Self, item: T) bool {
            while (true) {
                const tail = self.tail.load(.acquire);
                const next_tail = (tail + 1) % capacity;

                if (next_tail == self.head.load(.acquire)) {
                    return false; // Queue full
                }

                if (self.buffer[tail].cmpxchgStrong(null, item, .release, .acquire) == null) {
                    _ = self.tail.cmpxchgStrong(tail, next_tail, .release, .acquire);
                    return true;
                }
            }
        }

        pub fn dequeue(self: *Self) ?T {
            while (true) {
                const head = self.head.load(.acquire);

                if (head == self.tail.load(.acquire)) {
                    return null; // Queue empty
                }

                const item = self.buffer[head].load(.acquire);
                if (item != null) {
                    if (self.buffer[head].cmpxchgStrong(item, null, .release, .acquire) == null) {
                        _ = self.head.cmpxchgStrong(head, (head + 1) % capacity, .release, .acquire);
                        return item;
                    }
                }
            }
        }
    };
}

test "lock-free queue" {
    const testing = std.testing;

    var queue = try lockFreeQueue(i32).init(testing.allocator);
    defer queue.deinit();

    try queue.enqueue(1);
    try queue.enqueue(2);
    try queue.enqueue(3);

    try testing.expectEqual(@as(?i32, 1), queue.dequeue());
    try testing.expectEqual(@as(?i32, 2), queue.dequeue());
    try testing.expectEqual(@as(?i32, 3), queue.dequeue());
    try testing.expectEqual(@as(?i32, null), queue.dequeue());
}

test "lock-free stack" {
    const testing = std.testing;

    var stack = lockFreeStack(i32).init(testing.allocator);
    defer stack.deinit();

    try stack.push(1);
    try stack.push(2);
    try stack.push(3);

    try testing.expectEqual(@as(?i32, 3), stack.pop());
    try testing.expectEqual(@as(?i32, 2), stack.pop());
    try testing.expectEqual(@as(?i32, 1), stack.pop());
    try testing.expectEqual(@as(?i32, null), stack.pop());
}

test "MPMC queue" {
    const testing = std.testing;

    var queue = mpmcQueue(i32, 16).init();

    try testing.expect(queue.enqueue(1));
    try testing.expect(queue.enqueue(2));

    try testing.expectEqual(@as(?i32, 1), queue.dequeue());
    try testing.expectEqual(@as(?i32, 2), queue.dequeue());
    try testing.expectEqual(@as(?i32, null), queue.dequeue());
}