//! Priority request scheduler.
//!
//! Heap-based priority queue that orders inference requests by priority
//! (higher first) with FIFO tie-breaking via timestamps.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Request = struct {
    id: u64,
    prompt: []const u8,
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    profile: []const u8 = "abi",
    priority: u8 = 128,
    created_at: i64 = 0,
    stream: bool = false,
};

pub const Scheduler = struct {
    const Self = @This();

    allocator: Allocator,
    queue: std.ArrayListUnmanaged(Request),
    max_size: u32,

    pub fn init(allocator: Allocator, max_size: u32) Self {
        return .{
            .allocator = allocator,
            .queue = .empty,
            .max_size = max_size,
        };
    }

    pub fn deinit(self: *Self) void {
        self.queue.deinit(self.allocator);
    }

    pub fn submit(self: *Self, request: Request) !bool {
        if (self.queue.items.len >= self.max_size) return false;
        try self.queue.append(self.allocator, request);
        self.bubbleUp(self.queue.items.len - 1);
        return true;
    }

    pub fn getBatch(self: *Self, max_batch: u32) ![]Request {
        const count = @min(max_batch, @as(u32, @intCast(self.queue.items.len)));
        if (count == 0) return &.{};

        const batch = try self.allocator.alloc(Request, count);
        for (0..count) |i| {
            batch[i] = self.pop().?;
        }
        return batch;
    }

    pub fn pendingCount(self: *const Self) usize {
        return self.queue.items.len;
    }

    fn higherPriority(a: Request, b: Request) bool {
        if (a.priority != b.priority) return a.priority > b.priority;
        return a.created_at < b.created_at;
    }

    fn bubbleUp(self: *Self, idx: usize) void {
        var i = idx;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (higherPriority(self.queue.items[i], self.queue.items[parent])) {
                std.mem.swap(Request, &self.queue.items[i], &self.queue.items[parent]);
                i = parent;
            } else break;
        }
    }

    fn bubbleDown(self: *Self, idx: usize) void {
        var i = idx;
        const n = self.queue.items.len;
        while (true) {
            var best = i;
            const left = 2 * i + 1;
            const right = 2 * i + 2;
            if (left < n and higherPriority(self.queue.items[left], self.queue.items[best])) {
                best = left;
            }
            if (right < n and higherPriority(self.queue.items[right], self.queue.items[best])) {
                best = right;
            }
            if (best == i) break;
            std.mem.swap(Request, &self.queue.items[i], &self.queue.items[best]);
            i = best;
        }
    }

    fn pop(self: *Self) ?Request {
        if (self.queue.items.len == 0) return null;
        const top = self.queue.items[0];
        const last = self.queue.items.len - 1;
        self.queue.items[0] = self.queue.items[last];
        self.queue.items.len = last;
        if (last > 0) self.bubbleDown(0);
        return top;
    }
};

test "scheduler priority ordering" {
    const allocator = std.testing.allocator;
    var sched = Scheduler.init(allocator, 100);
    defer sched.deinit();

    _ = try sched.submit(.{ .id = 1, .prompt = "low", .priority = 10, .created_at = 1 });
    _ = try sched.submit(.{ .id = 2, .prompt = "high", .priority = 200, .created_at = 2 });
    _ = try sched.submit(.{ .id = 3, .prompt = "med", .priority = 100, .created_at = 3 });

    try std.testing.expectEqual(@as(usize, 3), sched.pendingCount());

    const batch = try sched.getBatch(3);
    defer allocator.free(batch);

    try std.testing.expectEqual(@as(u64, 2), batch[0].id);
    try std.testing.expectEqual(@as(u64, 3), batch[1].id);
    try std.testing.expectEqual(@as(u64, 1), batch[2].id);
}

test "scheduler full queue" {
    const allocator = std.testing.allocator;
    var sched = Scheduler.init(allocator, 2);
    defer sched.deinit();

    _ = try sched.submit(.{ .id = 1, .prompt = "a", .priority = 1 });
    _ = try sched.submit(.{ .id = 2, .prompt = "b", .priority = 2 });
    const ok = try sched.submit(.{ .id = 3, .prompt = "c", .priority = 3 });
    try std.testing.expect(!ok);
}

test "scheduler FIFO for same priority" {
    const allocator = std.testing.allocator;
    var sched = Scheduler.init(allocator, 100);
    defer sched.deinit();

    _ = try sched.submit(.{ .id = 1, .prompt = "first", .priority = 100, .created_at = 10 });
    _ = try sched.submit(.{ .id = 2, .prompt = "second", .priority = 100, .created_at = 20 });

    const batch = try sched.getBatch(2);
    defer allocator.free(batch);
    try std.testing.expectEqual(@as(u64, 1), batch[0].id);
}
