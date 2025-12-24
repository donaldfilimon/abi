const std = @import("std");

pub fn SpscRingBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        buffer: []T,
        capacity: usize,
        head: std.atomic.Value(usize),
        tail: std.atomic.Value(usize),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            if (capacity < 2) return error.InvalidCapacity;
            const buffer = try allocator.alloc(T, capacity);
            return .{
                .buffer = buffer,
                .capacity = capacity,
                .head = std.atomic.Value(usize).init(0),
                .tail = std.atomic.Value(usize).init(0),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }

        pub fn push(self: *Self, value: T) bool {
            const head = self.head.load(.acquire);
            const tail = self.tail.load(.acquire);
            const next = (head + 1) % self.capacity;
            if (next == tail) return false;
            self.buffer[head] = value;
            self.head.store(next, .release);
            return true;
        }

        pub fn pop(self: *Self) ?T {
            const tail = self.tail.load(.acquire);
            const head = self.head.load(.acquire);
            if (tail == head) return null;
            const value = self.buffer[tail];
            self.tail.store((tail + 1) % self.capacity, .release);
            return value;
        }
    };
}

pub fn LockFreeStack(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            value: T,
            next: ?*Node,
        };

        allocator: std.mem.Allocator,
        head: std.atomic.Value(?*Node),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator, .head = std.atomic.Value(?*Node).init(null) };
        }

        pub fn deinit(self: *Self) void {
            while (self.pop()) |_| {}
        }

        pub fn push(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .value = value, .next = null };

            var current = self.head.load(.acquire);
            while (true) {
                node.next = current;
                if (self.head.cmpxchgWeak(current, node, .release, .acquire) == null) {
                    return;
                }
                current = self.head.load(.acquire);
            }
        }

        pub fn pop(self: *Self) ?T {
            var current = self.head.load(.acquire);
            while (current) |node| {
                const next = node.next;
                if (self.head.cmpxchgWeak(current, next, .acq_rel, .acquire) == null) {
                    const value = node.value;
                    self.allocator.destroy(node);
                    return value;
                }
                current = self.head.load(.acquire);
            }
            return null;
        }
    };
}

pub fn ConcurrentMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        map: std.AutoHashMap(K, V),
        mutex: std.Thread.Mutex,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .map = std.AutoHashMap(K, V).init(allocator),
                .mutex = std.Thread.Mutex{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.map.deinit();
        }

        pub fn put(self: *Self, key: K, value: V) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.map.put(key, value);
        }

        pub fn get(self: *Self, key: K) ?V {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.map.get(key);
        }

        pub fn remove(self: *Self, key: K) bool {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.map.remove(key);
        }
    };
}

test "spsc ring buffer" {
    var ring = try SpscRingBuffer(u32).init(std.testing.allocator, 4);
    defer ring.deinit();

    try std.testing.expect(ring.push(1));
    try std.testing.expect(ring.push(2));
    try std.testing.expectEqual(@as(?u32, 1), ring.pop());
    try std.testing.expectEqual(@as(?u32, 2), ring.pop());
    try std.testing.expectEqual(@as(?u32, null), ring.pop());
}

test "lock-free stack" {
    var stack = LockFreeStack(u32).init(std.testing.allocator);
    defer stack.deinit();

    try stack.push(10);
    try stack.push(20);
    try std.testing.expectEqual(@as(?u32, 20), stack.pop());
    try std.testing.expectEqual(@as(?u32, 10), stack.pop());
    try std.testing.expectEqual(@as(?u32, null), stack.pop());
}
