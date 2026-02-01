const std = @import("std");
const types = @import("types.zig");

pub const ParallelSearchConfig = struct {
    thread_count: ?usize = null,
    min_batch_size: usize = 4,
    use_simd: bool = true,
    use_gpu: bool = false,
    prefetch_distance: usize = 8,
};

pub const ParallelSearchExecutor = struct {
    config: ParallelSearchConfig,
    thread_count: usize = 1,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: ParallelSearchConfig) ParallelSearchExecutor {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn searchBatch(_: *ParallelSearchExecutor, _: []const []const f32, _: []const []const f32, _: usize) !BatchSearchResult {
        return error.DatabaseDisabled;
    }

    pub fn freeResults(_: *ParallelSearchExecutor, _: *BatchSearchResult) void {}
};

pub const ParallelBeamState = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ParallelBeamState {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *ParallelBeamState) void {}

    pub fn addCandidate(_: *ParallelBeamState, _: usize, _: f32) !void {
        return error.DatabaseDisabled;
    }

    pub fn markVisited(_: *ParallelBeamState, _: usize) !bool {
        return error.DatabaseDisabled;
    }

    pub fn getTopK(_: *ParallelBeamState, _: usize, _: []types.SearchResult) usize {
        return 0;
    }
};

pub fn ParallelWorkQueue(comptime T: type) type {
    return struct {
        const Self = @This();
        items: []T = &.{},
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: []const T) !Self {
            return Self{ .allocator = allocator };
        }

        pub fn deinit(_: *Self) void {}

        pub fn getNext(_: *Self) ?T {
            return null;
        }

        pub fn getBatch(_: *Self, _: usize) ?[]T {
            return null;
        }
    };
}

pub const BatchSearchResult = struct {
    results: [][]types.SearchResult = &.{},
    total_time_ns: u64 = 0,
    distance_computations: u64 = 0,
};

pub const ParallelSearchStats = struct {
    total_queries: u64 = 0,
    total_distances: u64 = 0,
    total_time_ns: u64 = 0,
    parallel_queries: u64 = 0,

    pub fn avgLatencyUs(_: ParallelSearchStats) f64 {
        return 0;
    }

    pub fn throughput(_: ParallelSearchStats) f64 {
        return 0;
    }
};

pub fn batchCosineDistances(_: []const f32, _: f32, _: []const []const f32, distances: []f32) void {
    @memset(distances, 1.0);
}
