//! Database Module - Vector Database and Semantic Store API
//!
//! This module provides ABI's canonical semantic-store surface for high-performance
//! similarity search, weighted retrieval, and distributed lineage tracking.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");

pub const semantic_store = @import("semantic_store/mod.zig");

/// Compatibility alias for the legacy WDBX handle surface.
pub const wdbx = semantic_store;

/// Neural vector engine surface (ANN/HNSW internals and engine API).
pub const neural = @import("../../wdbx/wdbx.zig");

pub const DatabaseHandle = semantic_store.DatabaseHandle;
pub const SearchResult = semantic_store.SearchResult;
pub const VectorView = semantic_store.VectorView;
pub const Stats = semantic_store.Stats;
pub const BatchItem = semantic_store.BatchItem;

pub const DatabaseFeatureError = error{
    DatabaseDisabled,
};

/// Database Context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.DatabaseConfig,
    handle: ?DatabaseHandle = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.DatabaseConfig) !*Context {
        if (!isEnabled()) return error.DatabaseDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .handle = null,
        };

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.handle) |*h| {
            semantic_store.closeStore(h);
        }
        self.allocator.destroy(self);
    }

    pub fn getHandle(self: *Context) !*DatabaseHandle {
        if (self.handle) |*h| {
            return h;
        }
        self.handle = try semantic_store.openStore(self.allocator, self.config.path);
        return &self.handle.?;
    }

    pub fn insertVector(self: *Context, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        const h = try self.getHandle();
        try semantic_store.storeVector(h, id, vector, metadata);
    }

    pub fn searchVectors(self: *Context, query: []const f32, top_k: usize) ![]SearchResult {
        const h = try self.getHandle();
        return semantic_store.searchStore(h, self.allocator, query, top_k);
    }
};

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    if (!isEnabled()) return DatabaseFeatureError.DatabaseDisabled;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.feat_database;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn open(allocator: std.mem.Allocator, name: []const u8) !DatabaseHandle {
    return semantic_store.openStore(allocator, name);
}

pub fn close(handle: *DatabaseHandle) void {
    semantic_store.closeStore(handle);
}

test "database module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}
