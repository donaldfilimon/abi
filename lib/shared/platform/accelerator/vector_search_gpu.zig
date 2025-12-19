//! GPU-accelerated vector search (CPU fallback stub)
//!
//! Minimal implementation to keep examples compiling on Zig 0.16. The current
//! backend stores vectors in CPU memory and performs a brute-force L2 search.
//! The API is intentionally small to match the example in
//! `examples/neural_network_training.zig`.

const std = @import("std");
const accelerator = @import("accelerator.zig");

pub const Error = error{
    InvalidDimension,
};

const VectorEntry = struct {
    id: usize,
    values: []f32,
};

pub const VectorSearchGPU = struct {
    allocator: std.mem.Allocator,
    accel: *accelerator.Accelerator,
    dim: usize,
    next_id: usize = 0,
    vectors: std.ArrayList(VectorEntry),

    pub fn init(allocator: std.mem.Allocator, accel: *accelerator.Accelerator, dim: usize) VectorSearchGPU {
        return .{
            .allocator = allocator,
            .accel = accel,
            .dim = dim,
            .vectors = std.ArrayList(VectorEntry).init(),
        };
    }

    pub fn deinit(self: *VectorSearchGPU) void {
        for (self.vectors.items) |entry| {
            self.allocator.free(entry.values);
        }
        self.vectors.deinit(self.allocator);
    }

    pub fn insert(self: *VectorSearchGPU, embedding: []const f32) !usize {
        if (embedding.len != self.dim) return Error.InvalidDimension;

        const copy = try self.allocator.alloc(f32, self.dim);
        @memcpy(copy, embedding);

        const id = self.next_id;
        self.next_id += 1;

        try self.vectors.append(self.allocator, .{ .id = id, .values = copy });
        return id;
    }

    pub fn search(self: *VectorSearchGPU, query: []const f32, k: usize) ![]usize {
        if (query.len != self.dim) return Error.InvalidDimension;
        if (self.vectors.items.len == 0) return try self.allocator.alloc(usize, 0);

        const Temp = struct { idx: usize, dist: f32 };
        var distances = try self.allocator.alloc(Temp, self.vectors.items.len);
        defer self.allocator.free(distances);

        for (self.vectors.items, 0..) |entry, i| {
            distances[i] = .{ .idx = i, .dist = l2(query, entry.values) };
        }

        std.sort.block(Temp, distances, {}, struct {
            fn lessThan(_: void, a: Temp, b: Temp) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        const count = @min(k, distances.len);
        const results = try self.allocator.alloc(usize, count);
        for (results, 0..) |*dst, i| {
            dst.* = self.vectors.items[distances[i].idx].id;
        }
        return results;
    }
};

fn l2(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, 0..) |v, i| {
        const diff = v - b[i];
        sum += diff * diff;
    }
    return @sqrt(sum);
}

test "vector search insert and search" {
    const testing = std.testing;
    const accel = accelerator.createBestAccelerator(testing.allocator);
    const searcher = VectorSearchGPU.init(testing.allocator, &accel, 4);
    defer searcher.deinit();

    const v0 = [_]f32{ 0, 0, 0, 0 };
    const v1 = [_]f32{ 1, 0, 0, 0 };
    const v2 = [_]f32{ 0, 1, 0, 0 };

    try testing.expectEqual(@as(usize, 0), try searcher.insert(&v0));
    try testing.expectEqual(@as(usize, 1), try searcher.insert(&v1));
    try testing.expectEqual(@as(usize, 2), try searcher.insert(&v2));

    const res = try searcher.search(&[_]f32{ 0.9, 0, 0, 0 }, 2);
    defer testing.allocator.free(res);

    try testing.expectEqual(@as(usize, 1), res[0]);
}
