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

    var dist: f32 = 0;

    if (query.len >= 16 and @hasDecl(std.simd, "f32x16")) {
        var i: usize = 0;
        const Vec = std.simd.f32x16;
        while (i + 16 <= query.len) : (i += 16) {
            const a: Vec = vector[i .. i + 16][0..16].*;
            const b: Vec = query[i .. i + 16][0..16].*;
            const diff = a - b;
            dist += @reduce(.Add, diff * diff);
        }
        while (i < query.len) : (i += 1) {
            const d = vector[i] - query[i];
            dist += d * d;
        }
    } else if (query.len >= 8 and @hasDecl(std.simd, "f32x8")) {
        var i: usize = 0;
        const Vec = std.simd.f32x8;
        while (i + 8 <= query.len) : (i += 8) {
            const a: Vec = vector[i .. i + 8][0..8].*;
            const b: Vec = query[i .. i + 8][0..8].*;
            const diff = a - b;
            dist += @reduce(.Add, diff * diff);
        }
        while (i < query.len) : (i += 1) {
            const diff = vector[i] - query[i];
            dist += diff * diff;
        }
    } else if (query.len > 8) {
        var i: usize = 0;
        while (i + 4 <= query.len) : (i += 4) {
            const diff0 = vector[i] - query[i];
            const diff1 = vector[i + 1] - query[i + 1];
            const diff2 = vector[i + 2] - query[i + 2];
            const diff3 = vector[i + 3] - query[i + 3];
            dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
        while (i < query.len) : (i += 1) {
            const diff = vector[i] - query[i];
            dist += diff * diff;
        }
    } else {
        for (vector, query) |val, q| {
            const diff = val - q;
            dist += diff * diff;
        }
    }

    return dist;
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
    const dim_usize: usize = @intCast(dimension);
    const row_count_usize: usize = @intCast(row_count);

    if (query.len != dim_usize) {
        return error.DimensionMismatch;
    }

    if (row_count == 0 or top_k == 0) {
        return allocator.alloc(Result, 0);
    }

    if (num_threads <= 1 or row_count_usize <= 1) {
        const read_buffer = try allocator.alloc(f32, dim_usize);
        defer allocator.free(read_buffer);
        return linearSearch(file, records_offset, record_size, row_count, dimension, query, top_k, allocator, read_buffer);
    }

    const thread_count: usize = @min(@as(usize, @intCast(num_threads)), row_count_usize);
    const chunk_size: u64 = (row_count + @as(u64, @intCast(thread_count)) - 1) / @as(u64, @intCast(thread_count));

    var threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    var states = try allocator.alloc(ThreadState, thread_count);
    defer {
        for (states) |state| {
            allocator.free(state.row_values);
            allocator.free(state.results);
        }
        allocator.free(states);
    }

    for (0..thread_count) |i| {
        const start_row = @as(u64, @intCast(i)) * chunk_size;
        const end_row = @min(start_row + chunk_size, row_count);
        const chunk_len = @as(usize, @intCast(end_row - start_row));
        const results_len = @min(top_k, chunk_len);

        states[i] = .{
            .file = file,
            .records_offset = records_offset,
            .record_size = record_size,
            .query = query,
            .start_row = start_row,
            .end_row = end_row,
            .row_values = try allocator.alloc(f32, dim_usize),
            .results = try allocator.alloc(Result, results_len),
            .count = 0,
            .err = null,
        };

        threads[i] = try std.Thread.spawn(.{}, searchThread, .{&states[i]});
    }

    for (threads) |thread| {
        thread.join();
    }

    for (states) |state| {
        if (state.err) |err| return err;
    }

    return mergeThreadResults(states, row_count_usize, top_k, allocator);
}

const ThreadState = struct {
    file: std.fs.File,
    records_offset: u64,
    record_size: u64,
    query: []const f32,
    start_row: u64,
    end_row: u64,
    row_values: []f32,
    results: []Result,
    count: usize,
    err: ?anyerror,
};

fn searchThread(state: *ThreadState) void {
    state.err = searchChunk(state) catch |err| err;
}

fn searchChunk(state: *ThreadState) !void {
    const row_bytes = std.mem.sliceAsBytes(state.row_values);
    var row: u64 = state.start_row;
    while (row < state.end_row) : (row += 1) {
        const offset: u64 = state.records_offset + row * state.record_size;
        _ = try state.file.preadAll(row_bytes, offset);

        const dist = computeEuclideanDistance(state.query, state.row_values);
        insertTopK(state.results, &state.count, .{ .index = row, .score = dist });
    }
}

fn insertTopK(results: []Result, count: *usize, candidate: Result) void {
    if (results.len == 0) return;

    if (count.* < results.len) {
        results[count.*] = candidate;
        count.* += 1;
    } else if (candidate.score < results[count.* - 1].score) {
        results[count.* - 1] = candidate;
    } else {
        return;
    }

    var idx: usize = count.* - 1;
    while (idx > 0 and results[idx].score < results[idx - 1].score) : (idx -= 1) {
        const tmp = results[idx - 1];
        results[idx - 1] = results[idx];
        results[idx] = tmp;
    }
}

fn mergeThreadResults(states: []ThreadState, row_count: usize, top_k: usize, allocator: std.mem.Allocator) ![]Result {
    const result_len = @min(top_k, row_count);
    var merged = try allocator.alloc(Result, result_len);
    var count: usize = 0;

    for (states) |state| {
        for (state.results[0..state.count]) |result| {
            insertTopK(merged, &count, result);
        }
    }

    if (count == merged.len) return merged;

    const trimmed = try allocator.alloc(Result, count);
    @memcpy(trimmed, merged[0..count]);
    allocator.free(merged);
    return trimmed;
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
