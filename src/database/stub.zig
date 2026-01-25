//! Minimal stub for Database feature when disabled.
//!
//! Provides API compatibility with mod.zig while returning DatabaseDisabled for all operations.
//! Types are kept minimal - only essential ones needed for compile-time checking.

const std = @import("std");
const config_module = @import("../config/mod.zig");

// ============================================================================
// Error Types
// ============================================================================

pub const DatabaseFeatureError = error{DatabaseDisabled};
pub const DatabaseError = error{
    DuplicateId,
    VectorNotFound,
    InvalidDimension,
    PoolExhausted,
    PersistenceError,
    ConcurrencyError,
    DatabaseDisabled,
};

// ============================================================================
// Core Types (minimal definitions for type compatibility)
// ============================================================================

pub const DatabaseHandle = struct { db: ?*anyopaque = null };
pub const SearchResult = struct { id: u64 = 0, score: f32 = 0.0 };
pub const VectorView = struct { id: u64 = 0, vector: []const f32 = &.{}, metadata: ?[]const u8 = null };
pub const Stats = struct { count: usize = 0, dimension: usize = 0 };
pub const BatchItem = struct { id: u64 = 0, vector: []const f32 = &.{}, metadata: ?[]const u8 = null };
pub const DiagnosticsInfo = struct {
    name: []const u8 = "",
    vector_count: usize = 0,
    dimension: usize = 0,
    pub fn isHealthy(_: DiagnosticsInfo) bool {
        return false;
    }
};

// ============================================================================
// Context (Framework integration)
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.DatabaseConfig,
    handle: ?DatabaseHandle = null,

    pub fn init(_: std.mem.Allocator, _: config_module.DatabaseConfig) !*Context {
        return error.DatabaseDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getHandle(_: *Context) !*DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn openDatabase(_: *Context, _: []const u8) !*DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn insertVector(_: *Context, _: u64, _: []const f32, _: ?[]const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn searchVectors(_: *Context, _: []const f32, _: usize) ![]SearchResult {
        return error.DatabaseDisabled;
    }
    pub fn getStats(_: *Context) !Stats {
        return error.DatabaseDisabled;
    }
    pub fn optimize(_: *Context) !void {
        return error.DatabaseDisabled;
    }
};

// ============================================================================
// Sub-module Stubs (minimal namespace compatibility)
// ============================================================================

const stub_root = @This();

pub const wdbx = struct {
    pub const DatabaseHandle = stub_root.DatabaseHandle;
    pub const SearchResult = stub_root.SearchResult;
    pub const VectorView = stub_root.VectorView;
    pub const Stats = stub_root.Stats;
    pub const BatchItem = stub_root.BatchItem;
    pub const DatabaseConfig = struct { cache_norms: bool = false, initial_capacity: usize = 0, use_vector_pool: bool = false, thread_safe: bool = false };

    pub fn createDatabase(_: std.mem.Allocator, _: []const u8) !stub_root.DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn createDatabaseWithConfig(_: std.mem.Allocator, _: []const u8, _: DatabaseConfig) !stub_root.DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn connectDatabase(_: std.mem.Allocator, _: []const u8) !stub_root.DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn closeDatabase(_: *stub_root.DatabaseHandle) void {}
    pub fn insertVector(_: *stub_root.DatabaseHandle, _: u64, _: []const f32, _: ?[]const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn insertBatch(_: *stub_root.DatabaseHandle, _: []const stub_root.BatchItem) !void {
        return error.DatabaseDisabled;
    }
    pub fn searchVectors(_: *stub_root.DatabaseHandle, _: std.mem.Allocator, _: []const f32, _: usize) ![]stub_root.SearchResult {
        return error.DatabaseDisabled;
    }
    pub fn deleteVector(_: *stub_root.DatabaseHandle, _: u64) bool {
        return false;
    }
    pub fn updateVector(_: *stub_root.DatabaseHandle, _: u64, _: []const f32) !bool {
        return error.DatabaseDisabled;
    }
    pub fn getVector(_: *stub_root.DatabaseHandle, _: u64) ?stub_root.VectorView {
        return null;
    }
    pub fn listVectors(_: *stub_root.DatabaseHandle, _: std.mem.Allocator, _: usize) ![]stub_root.VectorView {
        return error.DatabaseDisabled;
    }
    pub fn getStats(_: *stub_root.DatabaseHandle) stub_root.Stats {
        return .{};
    }
    pub fn optimize(_: *stub_root.DatabaseHandle) !void {
        return error.DatabaseDisabled;
    }
    pub fn backup(_: *stub_root.DatabaseHandle, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn restore(_: *stub_root.DatabaseHandle, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
};

pub const cli = struct {
    pub fn run(_: std.mem.Allocator, _: []const [:0]const u8) !void {
        return error.DatabaseDisabled;
    }
};

// Parallel search stubs
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

    pub fn getTopK(_: *ParallelBeamState, _: usize, _: []SearchResult) usize {
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
    results: [][]SearchResult = &.{},
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

// Empty namespace stubs for less commonly used features
pub const parallel_search = struct {};
pub const database = struct {};
pub const db_helpers = struct {};
pub const storage = struct {};
pub const http = struct {};
pub const fulltext = struct {
    pub const InvertedIndex = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const hybrid = struct {
    pub const HybridSearchEngine = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const filter = struct {
    pub const FilterBuilder = struct {
        pub fn init() @This() {
            return .{};
        }
    };
};
pub const batch = struct {
    pub const BatchProcessor = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const clustering = struct {
    pub const KMeans = struct {
        pub fn init(_: std.mem.Allocator, _: usize, _: usize) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const quantization = struct {
    pub const ScalarQuantizer = struct {
        pub fn init(_: u8) @This() {
            return .{};
        }
    };
};
pub const formats = struct {
    pub const UnifiedFormat = struct {
        pub fn deinit(_: *@This()) void {}
    };
    pub const DataType = enum { f32, f16, bf16, i32, i16, i8, u8, q4_0, q4_1, q8_0 };
};

// ============================================================================
// Module Lifecycle
// ============================================================================

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    return error.DatabaseDisabled;
}
pub fn deinit() void {
    initialized = false;
}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return initialized;
}

// ============================================================================
// Core Database Operations
// ============================================================================

pub fn open(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn connect(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn close(_: *DatabaseHandle) void {}
pub fn insert(_: *DatabaseHandle, _: u64, _: []const f32, _: ?[]const u8) !void {
    return error.DatabaseDisabled;
}
pub fn search(_: *DatabaseHandle, _: std.mem.Allocator, _: []const f32, _: usize) ![]SearchResult {
    return error.DatabaseDisabled;
}
pub fn remove(_: *DatabaseHandle, _: u64) bool {
    return false;
}
pub fn update(_: *DatabaseHandle, _: u64, _: []const f32) !bool {
    return error.DatabaseDisabled;
}
pub fn get(_: *DatabaseHandle, _: u64) ?VectorView {
    return null;
}
pub fn list(_: *DatabaseHandle, _: std.mem.Allocator, _: usize) ![]VectorView {
    return error.DatabaseDisabled;
}
pub fn stats(_: *DatabaseHandle) Stats {
    return .{};
}
pub fn diagnostics(_: *DatabaseHandle) DiagnosticsInfo {
    return .{};
}
pub fn optimize(_: *DatabaseHandle) !void {
    return error.DatabaseDisabled;
}
pub fn backup(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}
pub fn restore(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}
pub fn openFromFile(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn openOrCreate(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
