//! Database Search Operations
//!
//! High-performance vector search operations for the WDBX database

const std = @import("std");

/// Search result structure
pub const Result = struct {
    index: u64,
    score: f32,

    pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
        return a.score < b.score;
    }
};

/// Statistics for search operations
pub const SearchStats = struct {
    search_count: u64 = 0,
    total_search_time_us: i64 = 0,

    pub fn getAverageSearchTime(self: *const SearchStats) u64 {
        if (self.search_count == 0) return 0;
        return @intCast(@divTrunc(self.total_search_time_us, @as(i64, @intCast(self.search_count))));
    }
};

/// Compute Euclidean distance between two vectors using SIMD optimizations
pub fn computeEuclideanDistance(query: []const f32, vector: []const f32) f32 {
    std.debug.assert(query.len == vector.len);

    var distance_squared: f32 = 0;

    if (query.len >= 16 and @hasDecl(std.simd, "f32x16")) {
        var i: usize = 0;
        const Vec = std.simd.f32x16;
        while (i + 16 <= query.len) : (i += 16) {
            const a: Vec = vector[i .. i + 16][0..16].*;
            const b: Vec = query[i .. i + 16][0..16].*;
            const diff = a - b;
            distance_squared += @reduce(.Add, diff * diff);
        }
        while (i < query.len) : (i += 1) {
            const d = vector[i] - query[i];
            distance_squared += d * d;
        }
    } else if (query.len >= 8 and @hasDecl(std.simd, "f32x8")) {
        var i: usize = 0;
        const Vec = std.simd.f32x8;
        while (i + 8 <= query.len) : (i += 8) {
            const a: Vec = vector[i .. i + 8][0..8].*;
            const b: Vec = query[i .. i + 8][0..8].*;
            const diff = a - b;
            distance_squared += @reduce(.Add, diff * diff);
        }
        while (i < query.len) : (i += 1) {
            const diff = vector[i] - query[i];
            distance_squared += diff * diff;
        }
    } else if (query.len > 8) {
        var i: usize = 0;
        while (i + 4 <= query.len) : (i += 4) {
            const diff0 = vector[i] - query[i];
            const diff1 = vector[i + 1] - query[i + 1];
            const diff2 = vector[i + 2] - query[i + 2];
            const diff3 = vector[i + 3] - query[i + 3];
            distance_squared += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
        while (i < query.len) : (i += 1) {
            const diff = vector[i] - query[i];
            distance_squared += diff * diff;
        }
    } else {
        for (vector, query) |val, q| {
            const diff = val - q;
            distance_squared += diff * diff;
        }
    }

    return @sqrt(distance_squared);
}

/// Linear search implementation - computes distance to all vectors
pub fn linearSearch(
    file: std.fs.File,
    records_offset: u64,
    record_size: u64,
    row_count: u64,
    dimension: u16,
    query: []const f32,
    top_k: usize,
    allocator: std.mem.Allocator,
    read_buffer: []f32,
) ![]Result {
    const dim_usize: usize = @intCast(dimension);
    const row_count_usize: usize = @intCast(row_count);

    if (query.len != dim_usize) {
        return error.DimensionMismatch;
    }

    if (row_count == 0) {
        return allocator.alloc(Result, 0);
    }

    var all = try allocator.alloc(Result, row_count_usize);
    defer allocator.free(all);

    const row_values = read_buffer[0..dim_usize];
    const row_bytes = std.mem.sliceAsBytes(row_values);

    for (0..row_count_usize) |row| {
        const offset: u64 = records_offset + @as(u64, row) * record_size;
        _ = try file.preadAll(row_bytes, offset);

        const dist = computeEuclideanDistance(query, row_values);
        all[row] = .{ .index = @intCast(row), .score = dist };
    }

    std.mem.sort(Result, all, {}, Result.lessThanAsc);

    const result_len = @min(top_k, all.len);
    const out = try allocator.alloc(Result, result_len);
    @memcpy(out, all[0..result_len]);
    return out;
}

/// Parallel search using multiple threads
pub fn parallelSearch(
    file: std.fs.File,
    records_offset: u64,
    record_size: u64,
    row_count: u64,
    dimension: u16,
    query: []const f32,
    top_k: usize,
    allocator: std.mem.Allocator,
    num_threads: u32,
) ![]Result {
    _ = num_threads; // TODO: Use for parallel implementation

    // TODO: Implement parallel search
    // For now, always fall back to single-threaded search
    const read_buffer = try allocator.alloc(f32, dimension);
    defer allocator.free(read_buffer);
    return linearSearch(file, records_offset, record_size, row_count, dimension, query, top_k, allocator, read_buffer);
}

test "compute euclidean distance" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };

    // Distance should be sqrt((3)^2 + (3)^2 + (3)^2) = sqrt(27) ≈ 5.196
    const dist = computeEuclideanDistance(&a, &b);
    try std.testing.expectApproxEqAbs(5.196, dist, 0.001);
}

test "linear search basic" {
    const allocator = std.testing.allocator;

    // Create a temporary file for testing
    const tmp_path = "test_db.tmp";
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var file = try std.fs.cwd().createFile(tmp_path, .{});
    defer file.close();

    // Write some test vectors (3D vectors)
    const vectors = [_][3]f32{
        [_]f32{ 1.0, 0.0, 0.0 },
        [_]f32{ 0.0, 1.0, 0.0 },
        [_]f32{ 0.0, 0.0, 1.0 },
    };

    for (vectors) |vec| {
        _ = try file.write(std.mem.sliceAsBytes(&vec));
    }

    const query = [_]f32{ 0.0, 0.0, 0.0 };
    const read_buffer = try allocator.alloc(f32, 3);
    defer allocator.free(read_buffer);

    const results = try linearSearch(file, 0, 3 * @sizeOf(f32), 3, 3, &query, 2, allocator, read_buffer);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    // Results should be sorted by distance (closest first)
    try std.testing.expectEqual(@as(u64, 0), results[0].index);
    try std.testing.expectEqual(@as(u64, 1), results[1].index);
}
