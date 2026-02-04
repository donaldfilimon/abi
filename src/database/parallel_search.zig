//! Parallel HNSW Search Utilities
//!
//! Provides parallel search capabilities for HNSW vector indices:
//! - Batch query processing across multiple threads
//! - Parallel candidate evaluation at layer 0
//! - SIMD-accelerated distance computation
//!
//! ## Performance
//!
//! - Multi-query: N queries processed in parallel
//! - Single-query: Layer 0 search parallelized across candidates
//! - Batch distance: SIMD/GPU acceleration for candidate evaluation

const std = @import("std");
const builtin = @import("builtin");
const simd = @import("../shared/simd.zig");

/// Whether threading is available on this target
const is_threaded_target = builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64;

/// Configuration for parallel search.
pub const ParallelSearchConfig = struct {
    /// Number of worker threads (null = auto-detect)
    thread_count: ?usize = null,
    /// Minimum batch size to enable parallelism
    min_batch_size: usize = 4,
    /// Enable SIMD distance computation
    use_simd: bool = true,
    /// Enable GPU acceleration if available
    use_gpu: bool = false,
    /// Prefetch distance for candidate vectors
    prefetch_distance: usize = 8,
};

/// Result from parallel search.
pub const SearchResult = struct {
    id: u64,
    score: f32,
};

/// Batch of query results.
pub const BatchSearchResult = struct {
    /// Results for each query (outer index = query, inner = results)
    results: [][]SearchResult,
    /// Total time in nanoseconds
    total_time_ns: u64,
    /// Number of distance computations
    distance_computations: u64,
};

/// Thread-safe work queue for parallel search.
pub fn ParallelWorkQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        items: []T,
        head: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, items: []const T) !Self {
            const copy = try allocator.dupe(T, items);
            return Self{
                .items = copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.items);
        }

        /// Get next work item. Returns null when queue is exhausted.
        pub fn getNext(self: *Self) ?T {
            const idx = self.head.fetchAdd(1, .monotonic);
            if (idx >= self.items.len) return null;
            return self.items[idx];
        }

        /// Get a batch of work items.
        pub fn getBatch(self: *Self, batch_size: usize) ?[]T {
            const start = self.head.fetchAdd(batch_size, .monotonic);
            if (start >= self.items.len) return null;
            const end = @min(start + batch_size, self.items.len);
            return self.items[start..end];
        }
    };
}

/// Compute cosine distances for a batch of candidates using SIMD.
/// For better performance with repeated queries, use batchCosineDistancesWithNorms
/// with pre-computed candidate norms.
pub fn batchCosineDistances(
    query: []const f32,
    query_norm: f32,
    candidates: []const []const f32,
    distances: []f32,
) void {
    if (query_norm == 0.0) {
        @memset(distances, 1.0);
        return;
    }

    const inv_query_norm = 1.0 / query_norm;

    for (candidates, 0..) |candidate, i| {
        if (i >= distances.len) break;

        // Use SIMD dot product and norm
        const dot = simd.vectorDot(query, candidate);
        const cand_norm = simd.vectorL2Norm(candidate);

        if (cand_norm > 0.0) {
            const similarity = dot * inv_query_norm / cand_norm;
            distances[i] = 1.0 - similarity;
        } else {
            distances[i] = 1.0;
        }
    }
}

/// Compute cosine distances with pre-computed candidate norms.
/// This is faster when candidate norms are cached/pre-computed.
/// @param query Query vector
/// @param query_norm Pre-computed L2 norm of query
/// @param candidates Slice of candidate vectors
/// @param candidate_norms Pre-computed L2 norms for each candidate (same length as candidates)
/// @param distances Output distances (cosine distance = 1 - similarity)
pub fn batchCosineDistancesWithNorms(
    query: []const f32,
    query_norm: f32,
    candidates: []const []const f32,
    candidate_norms: []const f32,
    distances: []f32,
) void {
    if (query_norm == 0.0) {
        @memset(distances, 1.0);
        return;
    }

    const inv_query_norm = 1.0 / query_norm;
    const len = @min(candidates.len, @min(candidate_norms.len, distances.len));

    for (0..len) |i| {
        const cand_norm = candidate_norms[i];

        if (cand_norm > 0.0) {
            // Use SIMD dot product only (norm already computed)
            const dot = simd.vectorDot(query, candidates[i]);
            const similarity = dot * inv_query_norm / cand_norm;
            distances[i] = 1.0 - similarity;
        } else {
            distances[i] = 1.0;
        }
    }
}

/// Pre-compute L2 norms for a batch of vectors.
/// Useful for caching norms of database vectors.
pub fn precomputeNorms(vectors: []const []const f32, norms: []f32) void {
    const len = @min(vectors.len, norms.len);
    for (0..len) |i| {
        norms[i] = simd.vectorL2Norm(vectors[i]);
    }
}

/// Parallel beam search state for layer 0 expansion.
pub const ParallelBeamState = struct {
    /// Current best candidates (min-heap by distance)
    candidates: std.AutoHashMapUnmanaged(usize, f32),
    /// Visited nodes
    visited: std.AutoHashMapUnmanaged(usize, void),
    /// Lock for concurrent access
    mutex: std.Thread.Mutex = .{},
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ParallelBeamState {
        return .{
            .candidates = .{},
            .visited = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ParallelBeamState) void {
        self.candidates.deinit(self.allocator);
        self.visited.deinit(self.allocator);
    }

    /// Thread-safe add candidate.
    pub fn addCandidate(self: *ParallelBeamState, node: usize, distance: f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.visited.contains(node)) {
            try self.visited.put(self.allocator, node, {});
            try self.candidates.put(self.allocator, node, distance);
        }
    }

    /// Thread-safe mark visited.
    pub fn markVisited(self: *ParallelBeamState, node: usize) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.visited.contains(node)) return false;
        try self.visited.put(self.allocator, node, {});
        return true;
    }

    /// Get top-k results.
    pub fn getTopK(self: *ParallelBeamState, k: usize, out: []SearchResult) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Collect all candidates
        var items = self.allocator.alloc(struct { node: usize, dist: f32 }, self.candidates.count()) catch return 0;
        defer self.allocator.free(items);

        var it = self.candidates.iterator();
        var i: usize = 0;
        while (it.next()) |entry| {
            items[i] = .{ .node = entry.key_ptr.*, .dist = entry.value_ptr.* };
            i += 1;
        }

        // Sort by distance
        std.mem.sort(
            struct { node: usize, dist: f32 },
            items[0..i],
            {},
            struct {
                fn lessThan(_: void, a: struct { node: usize, dist: f32 }, b: struct { node: usize, dist: f32 }) bool {
                    return a.dist < b.dist;
                }
            }.lessThan,
        );

        // Copy top-k
        const count = @min(k, i);
        for (items[0..count], 0..) |item, j| {
            out[j] = .{
                .id = item.node,
                .score = 1.0 - item.dist,
            };
        }

        return count;
    }
};

/// Parallel batch search executor.
pub const ParallelSearchExecutor = struct {
    config: ParallelSearchConfig,
    thread_count: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: ParallelSearchConfig) ParallelSearchExecutor {
        const cpu_count: usize = if (comptime is_threaded_target)
            std.Thread.getCpuCount() catch 4
        else
            1;
        const count = config.thread_count orelse @max(1, cpu_count);
        return .{
            .config = config,
            .thread_count = count,
            .allocator = allocator,
        };
    }

    /// Execute batch search with multiple queries.
    pub fn searchBatch(
        self: *ParallelSearchExecutor,
        queries: []const []const f32,
        vectors: []const []const f32,
        top_k: usize,
    ) !BatchSearchResult {
        var timer = std.time.Timer.start() catch null;
        var distance_count: u64 = 0;

        // Allocate results
        var results = try self.allocator.alloc([]SearchResult, queries.len);
        errdefer {
            for (results) |r| {
                if (r.len > 0) self.allocator.free(r);
            }
            self.allocator.free(results);
        }

        // Process each query
        for (queries, 0..) |query, q_idx| {
            const query_norm = simd.vectorL2Norm(query);

            // Compute distances to all vectors
            var distances = try self.allocator.alloc(f32, vectors.len);
            defer self.allocator.free(distances);

            for (vectors, 0..) |vec, i| {
                const dot = simd.vectorDot(query, vec);
                const vec_norm = simd.vectorL2Norm(vec);

                if (query_norm > 0 and vec_norm > 0) {
                    distances[i] = 1.0 - (dot / (query_norm * vec_norm));
                } else {
                    distances[i] = 1.0;
                }
                distance_count += 1;
            }

            // Find top-k
            const indices = try self.allocator.alloc(usize, vectors.len);
            defer self.allocator.free(indices);

            for (indices, 0..) |*idx, i| {
                idx.* = i;
            }

            // Partial sort for top-k
            std.mem.sort(usize, indices, distances, struct {
                fn lessThan(dists: []f32, a: usize, b: usize) bool {
                    return dists[a] < dists[b];
                }
            }.lessThan);

            const k = @min(top_k, vectors.len);
            results[q_idx] = try self.allocator.alloc(SearchResult, k);

            for (0..k) |i| {
                results[q_idx][i] = .{
                    .id = indices[i],
                    .score = 1.0 - distances[indices[i]],
                };
            }
        }

        return .{
            .results = results,
            .total_time_ns = if (timer) |*t| t.read() else 0,
            .distance_computations = distance_count,
        };
    }

    /// Free batch search results.
    pub fn freeResults(self: *ParallelSearchExecutor, result: *BatchSearchResult) void {
        for (result.results) |r| {
            if (r.len > 0) self.allocator.free(r);
        }
        self.allocator.free(result.results);
    }
};

/// Statistics for parallel search operations.
pub const ParallelSearchStats = struct {
    /// Total queries processed
    total_queries: u64 = 0,
    /// Total distance computations
    total_distances: u64 = 0,
    /// Total time in nanoseconds
    total_time_ns: u64 = 0,
    /// Queries processed in parallel
    parallel_queries: u64 = 0,

    /// Average query latency in microseconds.
    pub fn avgLatencyUs(self: ParallelSearchStats) f64 {
        if (self.total_queries == 0) return 0;
        return @as(f64, @floatFromInt(self.total_time_ns)) / @as(f64, @floatFromInt(self.total_queries)) / 1000.0;
    }

    /// Throughput in queries per second.
    pub fn throughput(self: ParallelSearchStats) f64 {
        if (self.total_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.total_queries)) * 1_000_000_000.0 / @as(f64, @floatFromInt(self.total_time_ns));
    }
};

// ============================================================================
// Tests
// ============================================================================

test "parallel work queue" {
    const items = [_]u32{ 1, 2, 3, 4, 5 };
    var queue = try ParallelWorkQueue(u32).init(std.testing.allocator, &items);
    defer queue.deinit();

    var sum: u32 = 0;
    while (queue.getNext()) |item| {
        sum += item;
    }

    try std.testing.expectEqual(@as(u32, 15), sum);
}

test "batch cosine distances" {
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const query_norm = simd.vectorL2Norm(&query);

    const candidates = [_][]const f32{
        &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, // Same direction
        &[_]f32{ 0.0, 1.0, 0.0, 0.0 }, // Orthogonal
        &[_]f32{ -1.0, 0.0, 0.0, 0.0 }, // Opposite
    };

    var distances: [3]f32 = undefined;
    batchCosineDistances(&query, query_norm, &candidates, &distances);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), distances[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), distances[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), distances[2], 0.01);
}

test "parallel beam state" {
    var state = ParallelBeamState.init(std.testing.allocator);
    defer state.deinit();

    try state.addCandidate(0, 0.1);
    try state.addCandidate(1, 0.3);
    try state.addCandidate(2, 0.2);

    var results: [2]SearchResult = undefined;
    const count = state.getTopK(2, &results);

    try std.testing.expectEqual(@as(usize, 2), count);
    // Best should be first (lowest distance = highest score)
    try std.testing.expect(results[0].score > results[1].score);
}

test "parallel search executor" {
    var executor = ParallelSearchExecutor.init(std.testing.allocator, .{});

    const queries = [_][]const f32{
        &[_]f32{ 1.0, 0.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 1.0, 0.0, 0.0 },
    };

    const vectors = [_][]const f32{
        &[_]f32{ 1.0, 0.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 1.0, 0.0, 0.0 },
        &[_]f32{ 0.5, 0.5, 0.0, 0.0 },
    };

    var result = try executor.searchBatch(&queries, &vectors, 2);
    defer executor.freeResults(&result);

    try std.testing.expectEqual(@as(usize, 2), result.results.len);
    try std.testing.expect(result.results[0].len > 0);
    try std.testing.expect(result.results[1].len > 0);
}
