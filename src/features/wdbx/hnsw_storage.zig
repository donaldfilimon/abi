const std = @import("std");
const memory = @import("../../core/memory.zig");

pub const VectorStorage = struct {
    allocator: std.mem.Allocator,
    data: std.ArrayListUnmanaged(f32),
    present: std.AutoHashMapUnmanaged(u32, void),
    dimensions: usize = 0,
    capacity: usize = 0,
    tracker: ?*memory.MemoryTracker = null,
    tracked_data_bytes: usize = 0,

    pub fn init(allocator: std.mem.Allocator, dimensions: usize, initial_capacity: usize) VectorStorage {
        return .{
            .allocator = allocator,
            .data = .empty,
            .present = .empty,
            .dimensions = dimensions,
            .capacity = initial_capacity,
        };
    }

    pub fn deinit(self: *VectorStorage) void {
        if (self.tracker) |tracker| {
            if (self.tracked_data_bytes > 0) tracker.trackFreeNoTag(self.tracked_data_bytes);
        }
        self.present.deinit(self.allocator);
        self.data.deinit(self.allocator);
    }

    pub fn setTracker(self: *VectorStorage, tracker: *memory.MemoryTracker) void {
        self.tracker = tracker;
    }

    pub fn insert(self: *VectorStorage, id: u32, values: []const f32) !void {
        if (values.len != self.dimensions) return error.DimensionMismatch;
        // Widen to usize before the multiply: `(id + 1) * dimensions` in the
        // u32 domain overflows for large ids, yielding a too-small `needed` and a
        // truncated backing store.
        const needed = (@as(usize, id) + 1) * self.dimensions;
        if (needed > self.data.items.len) {
            const old_len = self.data.items.len;
            const new_cap = @max(needed, self.data.items.len * 2 + 64);
            try self.data.resize(self.allocator, new_cap);
            if (self.tracker) |tracker| {
                const old_bytes = old_len * @sizeOf(f32);
                const new_bytes = new_cap * @sizeOf(f32);
                const tracked_growth = new_bytes - @min(old_bytes, new_bytes);
                if (tracked_growth > 0) {
                    tracker.trackAllocNoTag(tracked_growth);
                    self.tracked_data_bytes += tracked_growth;
                }
            }
            @memset(self.data.items[old_len..new_cap], 0);
        }
        const offset = id * self.dimensions;
        @memcpy(self.data.items[offset .. offset + self.dimensions], values);
        try self.present.put(self.allocator, id, {});
    }

    /// Borrowed view of a stored vector, or null when `id` was never inserted
    /// (or would fall outside the backing buffer). The presence map is the
    /// authority: a sparse id whose slot is zero-filled but never inserted is
    /// absent, not a zero vector. The bounds check is defensive — `present`
    /// implies an in-range slot — but keeps a stale id from indexing past the
    /// buffer. Index math is widened to usize to match `insert`'s overflow fix.
    pub fn get(self: *const VectorStorage, id: u32) ?[]const f32 {
        if (!self.present.contains(id)) return null;
        const offset = @as(usize, id) * self.dimensions;
        const end = offset + self.dimensions;
        if (end > self.data.items.len) return null;
        return self.data.items[offset..end];
    }

    pub fn contains(self: *const VectorStorage, id: u32) bool {
        return self.present.contains(id);
    }
};

test "VectorStorage insert and get" {
    var storage = VectorStorage.init(std.testing.allocator, 3, 8);
    defer storage.deinit();

    try storage.insert(0, &.{ 1.0, 2.0, 3.0 });
    try storage.insert(5, &.{ 4.0, 5.0, 6.0 });

    const v0 = storage.get(0) orelse return error.MissingVector;
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0 }, v0);

    const v5 = storage.get(5) orelse return error.MissingVector;
    try std.testing.expectEqualSlices(f32, &.{ 4.0, 5.0, 6.0 }, v5);

    try std.testing.expect(storage.contains(0));
    try std.testing.expect(storage.contains(5));
    try std.testing.expect(!storage.contains(10));
}

test "VectorStorage get returns null for an absent id" {
    var storage = VectorStorage.init(std.testing.allocator, 3, 8);
    defer storage.deinit();

    try storage.insert(2, &.{ 7.0, 8.0, 9.0 });

    // A present id returns the stored slice.
    const present = storage.get(2) orelse return error.MissingVector;
    try std.testing.expectEqualSlices(f32, &.{ 7.0, 8.0, 9.0 }, present);

    // An id that was never inserted is null — even id 0, whose slot is
    // zero-filled in the backing buffer but was never written.
    try std.testing.expectEqual(@as(?[]const f32, null), storage.get(0));
    // An id beyond anything inserted is also null.
    try std.testing.expectEqual(@as(?[]const f32, null), storage.get(99));
}

test {
    std.testing.refAllDecls(@This());
}
