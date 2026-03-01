//! Embedding vector look-back Cache
//!
//! Stores previously generated embeddings corresponding to explicit
//! text data or UUID references, drastically stripping duplicate calls
//! made to outer neural services boundaries.

const std = @import("std");
const sync = @import("sync_compat.zig");

pub const Cache = struct {
    const Entry = struct {
        embedding: []const f32,
        segment: usize,
        generation: u64,
    };

    const QueueItem = struct {
        key: []const u8,
        generation: u64,
    };

    const SegmentQueue = std.ArrayListUnmanaged(QueueItem);

    allocator: std.mem.Allocator,
    capacity: usize,
    segments: u8,
    map: std.StringHashMapUnmanaged(Entry),
    segment_queues: []SegmentQueue,
    eviction_segment: usize,
    next_generation: u64,
    mutex: sync.RwLock,

    pub fn init(allocator: std.mem.Allocator, capacity: usize, segments: u8) !Cache {
        const segment_count = @max(@as(usize, segments), 1);
        const queues = try allocator.alloc(SegmentQueue, segment_count);
        errdefer allocator.free(queues);
        for (queues) |*queue| {
            queue.* = .empty;
        }

        return .{
            .allocator = allocator,
            .capacity = capacity,
            .segments = @intCast(segment_count),
            .map = .empty,
            .segment_queues = queues,
            .eviction_segment = 0,
            .next_generation = 1,
            .mutex = sync.RwLock.init(),
        };
    }

    pub fn deinit(self: *Cache) void {
        var it = self.map.valueIterator();
        while (it.next()) |val| {
            self.allocator.free(val.embedding);
        }

        var iter = self.map.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }

        self.map.deinit(self.allocator);
        for (self.segment_queues) |*queue| {
            queue.deinit(self.allocator);
        }
        self.allocator.free(self.segment_queues);
    }

    pub fn get(self: *Cache, text: []const u8) ?[]const f32 {
        self.mutex.lockShared();
        defer self.mutex.unlockShared();

        return if (self.map.get(text)) |entry| entry.embedding else null;
    }

    pub fn put(self: *Cache, text: []const u8, embedding: []const f32) !void {
        if (self.capacity == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const cloned_embedding = try self.allocator.dupe(f32, embedding);
        errdefer self.allocator.free(cloned_embedding);
        const generation = self.nextGeneration();

        if (self.map.getEntry(text)) |entry| {
            self.allocator.free(entry.value_ptr.embedding);
            entry.value_ptr.embedding = cloned_embedding;
            entry.value_ptr.generation = generation;
            try self.enqueueLocked(entry.key_ptr.*, entry.value_ptr.segment, generation);
            return;
        }

        if (self.map.count() >= self.capacity) {
            try self.evictOneLocked();
        }

        const cloned_text = try self.allocator.dupe(u8, text);
        errdefer self.allocator.free(cloned_text);

        const segment = self.segmentFor(cloned_text);
        try self.map.put(self.allocator, cloned_text, .{
            .embedding = cloned_embedding,
            .segment = segment,
            .generation = generation,
        });
        errdefer {
            if (self.map.fetchRemove(cloned_text)) |kv| {
                self.allocator.free(kv.value.embedding);
                self.allocator.free(kv.key);
            }
        }
        try self.enqueueLocked(cloned_text, segment, generation);
    }

    fn nextGeneration(self: *Cache) u64 {
        const generation = self.next_generation;
        self.next_generation +%= 1;
        if (self.next_generation == 0) self.next_generation = 1;
        return generation;
    }

    fn segmentFor(self: *const Cache, key: []const u8) usize {
        const hash = std.hash.Wyhash.hash(0, key);
        return @as(usize, @intCast(hash % self.segment_queues.len));
    }

    fn enqueueLocked(self: *Cache, key: []const u8, segment: usize, generation: u64) !void {
        try self.segment_queues[segment].append(self.allocator, .{
            .key = key,
            .generation = generation,
        });
    }

    fn popFront(queue: *SegmentQueue) ?QueueItem {
        if (queue.items.len == 0) return null;
        const front = queue.items[0];
        if (queue.items.len > 1) {
            std.mem.copyForwards(QueueItem, queue.items[0 .. queue.items.len - 1], queue.items[1..]);
        }
        queue.items.len -= 1;
        return front;
    }

    fn evictOneLocked(self: *Cache) !void {
        if (self.map.count() == 0) return;

        const segment_count = self.segment_queues.len;
        for (0..segment_count) |_| {
            const segment = self.eviction_segment;
            self.eviction_segment = (self.eviction_segment + 1) % segment_count;

            const queue = &self.segment_queues[segment];
            while (popFront(queue)) |item| {
                if (self.map.getEntry(item.key)) |entry| {
                    if (entry.value_ptr.segment == segment and entry.value_ptr.generation == item.generation) {
                        if (self.map.fetchRemove(item.key)) |removed| {
                            self.allocator.free(removed.value.embedding);
                            self.allocator.free(removed.key);
                            return;
                        }
                    }
                }
            }
        }

        var key_it = self.map.keyIterator();
        if (key_it.next()) |key| {
            if (self.map.fetchRemove(key.*)) |removed| {
                self.allocator.free(removed.value.embedding);
                self.allocator.free(removed.key);
            }
        }
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

test "Cache deterministic segmented eviction keeps bounded capacity" {
    var cache = try Cache.init(std.testing.allocator, 2, 2);
    defer cache.deinit();

    try cache.put("k1", &[_]f32{1.0});
    try cache.put("k2", &[_]f32{2.0});
    try cache.put("k3", &[_]f32{3.0});

    try std.testing.expect(cache.map.count() <= 2);
    try std.testing.expect(cache.get("k3") != null);
}

test "Cache lock contention get and put" {
    const Worker = struct {
        cache: *Cache,
        id: usize,
    };
    const worker_fn = struct {
        fn run(worker: *Worker) void {
            for (0..100) |i| {
                var key_buf: [64]u8 = undefined;
                const key = std.fmt.bufPrint(&key_buf, "worker-{d}-{d}", .{ worker.id, i }) catch continue;
                const value = [_]f32{
                    @floatFromInt(worker.id),
                    @floatFromInt(i),
                    1.0,
                };
                worker.cache.put(key, &value) catch {};
                _ = worker.cache.get(key);
            }
        }
    }.run;

    var cache = try Cache.init(std.testing.allocator, 128, 8);
    defer cache.deinit();

    var workers = [_]Worker{
        .{ .cache = &cache, .id = 0 },
        .{ .cache = &cache, .id = 1 },
        .{ .cache = &cache, .id = 2 },
        .{ .cache = &cache, .id = 3 },
    };
    var threads: [workers.len]std.Thread = undefined;

    for (&threads, &workers) |*thread, *worker| {
        thread.* = try std.Thread.spawn(.{}, worker_fn, .{worker});
    }
    for (&threads) |*thread| {
        thread.join();
    }

    try std.testing.expect(cache.map.count() <= cache.capacity);
    try std.testing.expect(cache.map.count() > 0);
}
