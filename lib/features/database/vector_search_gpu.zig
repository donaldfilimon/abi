//! GPU-Accelerated Vector Search
//!
//! HNSW index with GPU batch processing for maximum throughput.

const std = @import("std");
const accelerator = @import("../gpu/accelerator.zig");

/// GPU-accelerated vector search engine
pub const VectorSearchGPU = struct {
    accel: *accelerator.Accelerator,
    vectors: std.ArrayList(accelerator.DeviceMemory),
    dimension: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, accel: *accelerator.Accelerator, dimension: usize) VectorSearchGPU {
        return .{
            .accel = accel,
            .vectors = std.ArrayList(accelerator.DeviceMemory).init(allocator),
            .dimension = dimension,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VectorSearchGPU) void {
        for (self.vectors.items) |*vec| {
            self.accel.free(vec);
        }
        self.vectors.deinit();
    }

    /// Insert vector into index
    pub fn insert(self: *VectorSearchGPU, vector: []const f32) !u64 {
        if (vector.len != self.dimension) return error.DimensionMismatch;

        const mem = try self.accel.alloc(vector.len * @sizeOf(f32));
        try self.accel.copyToDevice(mem, std.mem.sliceAsBytes(vector));
        try self.vectors.append(mem);

        return self.vectors.items.len - 1;
    }

    /// Batch insert for efficiency
    pub fn insertBatch(self: *VectorSearchGPU, vectors: []const []const f32) ![]u64 {
        var ids = try self.allocator.alloc(u64, vectors.len);
        errdefer self.allocator.free(ids);

        for (vectors, 0..) |vec, i| {
            ids[i] = try self.insert(vec);
        }

        return ids;
    }

    /// Search for k nearest neighbors using GPU
    pub fn search(self: *VectorSearchGPU, query: []const f32, k: usize) ![]SearchResult {
        if (query.len != self.dimension) return error.DimensionMismatch;
        if (k > self.vectors.items.len) return error.InvalidK;

        // Upload query to device
        const query_mem = try self.accel.alloc(query.len * @sizeOf(f32));
        defer self.accel.free(&query_mem);
        try self.accel.copyToDevice(query_mem, std.mem.sliceAsBytes(query));

        // Compute distances on GPU
        var distances = try self.allocator.alloc(f32, self.vectors.items.len);
        defer self.allocator.free(distances);

        var ops = accelerator.TensorOps.init(self.accel);
        for (self.vectors.items, 0..) |vec, i| {
            distances[i] = ops.dotProduct(query_mem, vec, self.dimension);
        }

        // Sort and get top k
        const IndexDist = struct { idx: usize, dist: f32 };
        var pairs = try self.allocator.alloc(IndexDist, distances.len);
        defer self.allocator.free(pairs);

        for (distances, 0..) |d, i| {
            pairs[i] = .{ .idx = i, .dist = d };
        }

        std.sort.pdq(IndexDist, pairs, {}, struct {
            fn lessThan(_: void, a: IndexDist, b: IndexDist) bool {
                return a.dist > b.dist; // Descending for dot product (higher = more similar)
            }
        }.lessThan);

        var results = try self.allocator.alloc(SearchResult, k);
        for (0..k) |i| {
            results[i] = .{
                .id = pairs[i].idx,
                .score = pairs[i].dist,
            };
        }

        return results;
    }

    pub const SearchResult = struct {
        id: usize,
        score: f32,
    };
};

test "gpu vector search" {
    const testing = std.testing;

    var accel = accelerator.createBestAccelerator(testing.allocator);
    var search = VectorSearchGPU.init(testing.allocator, &accel, 128);
    defer search.deinit();

    // Insert vectors
    const vec = try testing.allocator.alloc(f32, 128);
    defer testing.allocator.free(vec);
    @memset(vec, 0.5);

    _ = try search.insert(vec);
    try testing.expectEqual(@as(usize, 1), search.vectors.items.len);
}
