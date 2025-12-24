//! GPU-accelerated vector search (CPU fallback stub)
//!
//! Minimal implementation to keep GPU integration paths compiling on Zig 0.16.
//! The current backend stores vectors in CPU memory and performs a brute-force
//! L2 search with a deliberately small API surface.

const std = @import("std");
const accelerator = @import("../../shared/platform/accelerator/accelerator.zig");

/// Check if GPU acceleration is available
fn checkGPUAvailability() bool {
    // Note: Implement proper GPU availability check
    // For now, check build options
    const build_options = @import("build_options");
    return build_options.gpu_cuda or build_options.gpu_vulkan or build_options.gpu_metal;
}

pub const Error = error{
    InvalidDimension,
    InputTooLarge,
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

    /// Initialize vector search with GPU acceleration.
    /// @param allocator: Memory allocator
    /// @param accel: GPU accelerator
    /// @param dim: Vector dimension
    /// @return: Initialized VectorSearchGPU
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

    /// Insert a vector embedding.
    /// @param embedding: Vector to insert
    /// @return: ID of inserted vector
    pub fn insert(self: *VectorSearchGPU, embedding: []const f32) !usize {
        if (embedding.len != self.dim) return Error.InvalidDimension;
        if (embedding.len > 10000) return Error.InputTooLarge; // Rate limiting

        const copy = try self.allocator.alloc(f32, self.dim);
        errdefer self.allocator.free(copy);
        @memcpy(copy, embedding);

        const id = self.next_id;
        self.next_id += 1;

        try self.vectors.append(self.allocator, .{ .id = id, .values = copy });
        return id;
    }

    /// Search for k nearest neighbors.
    /// @param query: Query vector
    /// @param k: Number of neighbors to return
    /// @return: Array of neighbor IDs
    pub fn search(self: *VectorSearchGPU, query: []const f32, k: usize) ![]usize {
        if (query.len != self.dim) return Error.InvalidDimension;
        if (k > 100) return error.InvalidParameter; // Limit k for security
        if (self.vectors.items.len == 0) return try self.allocator.alloc(usize, 0);

        // Try GPU acceleration first
        if (self.tryGPUSearch(query, k)) |gpu_result| {
            return gpu_result;
        }

        // Fallback to CPU implementation
        std.log.debug("GPU search failed, falling back to CPU", .{});

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
        const result = try self.allocator.alloc(usize, count);
        for (0..count) |i| {
            result[i] = self.vectors.items[distances[i].idx].id;
        }
        return result;
    }

    /// Try to perform search using GPU acceleration (with SIMD fallback)
    fn tryGPUSearch(self: *VectorSearchGPU, query: []const f32, k: usize) ?[]usize {
        // Check if GPU acceleration is available
        const gpu_available = checkGPUAvailability();
        if (gpu_available) {
            std.log.debug("Attempting GPU-accelerated vector search", .{});
            // Note: Implement actual GPU distance computation
            // For now, fall through to SIMD-accelerated CPU implementation
        }

        // Use SIMD-accelerated CPU implementation as fallback
        return self.simdSearch(query, k);
    }

    /// SIMD-accelerated CPU vector search
    fn simdSearch(self: *VectorSearchGPU, query: []const f32, k: usize) ?[]usize {
        const Temp = struct { idx: usize, dist: f32 };

        // Pre-allocate result buffer
        var distances = std.ArrayList(Temp).initCapacity(self.allocator, self.vectors.items.len) catch return null;
        defer distances.deinit();

        // SIMD-accelerated distance calculation
        for (self.vectors.items, 0..) |entry, i| {
            const dist = simdL2Distance(query, entry.values);
            distances.appendAssumeCapacity(.{ .idx = i, .dist = dist });
        }

        // Sort by distance
        std.sort.block(Temp, distances.items, {}, struct {
            fn lessThan(_: void, a: Temp, b: Temp) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        // Extract top-k results
        const count = @min(k, distances.items.len);
        const result = self.allocator.alloc(usize, count) catch return null;

        for (0..count) |i| {
            result[i] = self.vectors.items[distances.items[i].idx].id;
        }

        return result;
    }
};

/// SIMD-accelerated L2 distance calculation
fn simdL2Distance(a: []const f32, b: []const f32) f32 {
    const len = a.len;
    var sum: f32 = 0.0;

    // Process vectors in chunks of 4 for SIMD
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const va = @as(@Vector(4, f32), a[i .. i + 4].*);
        const vb = @as(@Vector(4, f32), b[i .. i + 4].*);
        const diff = va - vb;
        const squared = diff * diff;
        sum += @reduce(.Add, squared);
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }

    return @sqrt(sum);
}

/// Fallback L2 distance for reference
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
    var accel = accelerator.createBestAccelerator(testing.allocator);
    var searcher = VectorSearchGPU.init(testing.allocator, &accel, 4);
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
