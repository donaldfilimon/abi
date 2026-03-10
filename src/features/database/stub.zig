//! Database Feature Stub Module
//!
//! API-compatible no-op implementations when the database feature is disabled.
//! Provides matching public signatures to `mod.zig` and `core/database/mod.zig`
//! so that callers compile regardless of the `feat_database` build flag.

const std = @import("std");
const stub_context = @import("../../core/stub_context");
const core_config = @import("../../core/config/database");

pub const DatabaseConfig = core_config.DatabaseConfig;

// ============================================================================
// Error types
// ============================================================================

pub const DatabaseFeatureError = error{
    DatabaseDisabled,
};

pub const DatabaseError = error{
    DatabaseDisabled,
    OutOfMemory,
    FileNotFound,
    InvalidFormat,
    CorruptedData,
    IndexError,
    WriteError,
    ReadError,
};

// ============================================================================
// Core types
// ============================================================================

pub const DatabaseHandle = struct {
    _unused: u8 = 0,
};

pub const SearchResult = struct {
    id: u64 = 0,
    score: f32 = 0.0,
    metadata: ?[]const u8 = null,
};

pub const VectorView = struct {
    id: u64 = 0,
    vector: []const f32 = &.{},
    metadata: ?[]const u8 = null,
};

pub const Stats = struct {
    total_vectors: u64 = 0,
    dimensions: u32 = 0,
    memory_bytes: u64 = 0,
    index_size: u64 = 0,
};

pub const BatchItem = struct {
    id: u64 = 0,
    vector: []const f32 = &.{},
    metadata: ?[]const u8 = null,
};

pub const StoreHandle = DatabaseHandle;

pub const DiagnosticsInfo = struct {
    status: []const u8 = "disabled",
    vector_count: u64 = 0,
};

pub const KMeans = struct {
    _unused: u8 = 0,
};

pub const ScalarQuantizer = struct {
    _unused: u8 = 0,
};

pub const ProductQuantizer = struct {
    _unused: u8 = 0,
};

// ============================================================================
// Context (framework integration)
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: DatabaseConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }

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

    pub fn searchVectorsInto(_: *Context, _: []const f32, _: usize, _: []SearchResult) !usize {
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
// Module-level functions
// ============================================================================

pub fn init(_: std.mem.Allocator) !void {
    return DatabaseFeatureError.DatabaseDisabled;
}

pub fn deinit() void {}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

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

pub fn search(
    _: *DatabaseHandle,
    _: std.mem.Allocator,
    _: []const f32,
    _: usize,
) ![]SearchResult {
    return error.DatabaseDisabled;
}

pub fn searchInto(
    _: *DatabaseHandle,
    _: []const f32,
    _: usize,
    _: []SearchResult,
) usize {
    return 0;
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

pub fn backupToPath(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}

pub fn restoreFromPath(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}

pub fn openFromFile(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}

pub fn openOrCreate(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}

// ============================================================================
// Sub-module stubs (parity with core/database/mod.zig)
// ============================================================================

/// Stub sub-module providing empty namespace.
fn StubSubmodule() type {
    return struct {};
}

pub const engine = StubSubmodule();
pub const hnsw = StubSubmodule();
pub const distance = StubSubmodule();
pub const simd = StubSubmodule();
pub const quantize = StubSubmodule();
pub const batch = StubSubmodule();
pub const fulltext = StubSubmodule();
pub const wdbx = StubSubmodule();
pub const hybrid = StubSubmodule();
pub const filter = StubSubmodule();
pub const clustering = StubSubmodule();
pub const formats = StubSubmodule();
pub const index = StubSubmodule();
pub const quantization = StubSubmodule();
pub const parallel_hnsw = StubSubmodule();
pub const parallel_search = StubSubmodule();
pub const database = StubSubmodule();
pub const storage = StubSubmodule();
pub const cli = StubSubmodule();
pub const neural = StubSubmodule();

const root = @This();

/// Semantic store stub for AI memory and other consumers.
pub const semantic_store = struct {
    pub const StoreHandle = root.DatabaseHandle;
    pub const SearchResult = root.SearchResult;

    pub const WeightInputs = struct {
        similarity: f32 = 0.0,
        importance: f32 = 0.0,
        recency: f32 = 1.0,
        custom_boost: f32 = 0.0,

        pub fn combinedScore(self: @This()) f32 {
            return self.similarity * 0.7 +
                self.importance * 0.2 +
                self.recency * 0.1 +
                self.custom_boost;
        }
    };

    pub const InfluenceTrace = struct {
        source: Source = .semantic_store,
        block_id: ?u64 = null,
        weight_inputs: WeightInputs = .{},

        pub const Source = enum {
            semantic_store,
            local_memory,
            distributed_replica,
        };

        pub fn forRetrieval(block_id: u64, similarity: f32, importance: f32) InfluenceTrace {
            return .{
                .source = .semantic_store,
                .block_id = block_id,
                .weight_inputs = .{
                    .similarity = similarity,
                    .importance = importance,
                },
            };
        }
    };

    pub const RetrievalHit = struct {
        block_id: u64 = 0,
        score: f32 = 0.0,
        similarity: f32 = 0.0,
        importance: f32 = 0.0,
        trace: InfluenceTrace = .{},
    };

    const Self = @This();

    pub fn openStore(_: std.mem.Allocator, _: []const u8) !Self.StoreHandle {
        return error.DatabaseDisabled;
    }

    pub fn closeStore(_: *Self.StoreHandle) void {}

    pub fn storeVector(_: *Self.StoreHandle, _: u64, _: []const f32, _: ?[]const u8) !void {
        return error.DatabaseDisabled;
    }

    pub fn searchStore(_: *Self.StoreHandle, _: std.mem.Allocator, _: []const f32, _: usize) ![]Self.SearchResult {
        return error.DatabaseDisabled;
    }
};

test {
    std.testing.refAllDecls(@This());
}
