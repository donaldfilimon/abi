//! Database Module - Vector Database and Semantic Store API
//!
//! This module provides ABI's canonical semantic-store surface for high-performance
//! similarity search, weighted retrieval, and distributed lineage tracking.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");

pub const database = @import("database.zig");
pub const storage = @import("storage.zig");
pub const semantic_store = @import("semantic_store/mod.zig");

/// Compatibility surface for legacy callers.
pub const wdbx = @import("wdbx.zig");

/// Neural vector engine surface (ANN/HNSW internals and engine API).
pub const neural = @import("../../wdbx/wdbx.zig");

/// Additional public modules still used by in-tree callers.
pub const cli = @import("cli.zig");
pub const batch = @import("batch.zig");
pub const formats = @import("formats/vector_db.zig");

pub const StoreHandle = semantic_store.StoreHandle;
pub const DatabaseHandle = wdbx.DatabaseHandle;
pub const SearchResult = wdbx.SearchResult;
pub const VectorView = wdbx.VectorView;
pub const Stats = wdbx.Stats;
pub const BatchItem = wdbx.BatchItem;
pub const DatabaseError = database.DatabaseError;
pub const DiagnosticsInfo = database.DiagnosticsInfo;

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

        if (cfg.path.len > 0) {
            ctx.handle = try wdbx.createDatabase(allocator, cfg.path);
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.handle) |*h| {
            wdbx.closeDatabase(h);
        }
        self.allocator.destroy(self);
    }

    pub fn getHandle(self: *Context) !*DatabaseHandle {
        if (self.handle) |*h| {
            return h;
        }
        self.handle = try wdbx.createDatabase(self.allocator, self.config.path);
        return &self.handle.?;
    }

    pub fn openDatabase(self: *Context, name: []const u8) !*DatabaseHandle {
        if (self.handle) |*h| {
            wdbx.closeDatabase(h);
            self.handle = null;
        }
        self.handle = try wdbx.createDatabase(self.allocator, name);
        return &self.handle.?;
    }

    pub fn insertVector(self: *Context, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        const h = try self.getHandle();
        try wdbx.insertVector(h, id, vector, metadata);
    }

    pub fn searchVectors(self: *Context, query: []const f32, top_k: usize) ![]SearchResult {
        const h = try self.getHandle();
        return wdbx.searchVectors(h, self.allocator, query, top_k);
    }

    pub fn searchVectorsInto(self: *Context, query: []const f32, top_k: usize, results: []SearchResult) !usize {
        const h = try self.getHandle();
        return wdbx.searchVectorsInto(h, query, top_k, results);
    }

    pub fn getStats(self: *Context) !Stats {
        const h = try self.getHandle();
        return wdbx.getStats(h);
    }

    pub fn optimize(self: *Context) !void {
        const h = try self.getHandle();
        try wdbx.optimize(h);
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
    return wdbx.createDatabase(allocator, name);
}

pub fn connect(allocator: std.mem.Allocator, name: []const u8) !DatabaseHandle {
    return wdbx.connectDatabase(allocator, name);
}

pub fn close(handle: *DatabaseHandle) void {
    wdbx.closeDatabase(handle);
}

pub fn insert(handle: *DatabaseHandle, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
    try wdbx.insertVector(handle, id, vector, metadata);
}

pub fn search(
    handle: *DatabaseHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]SearchResult {
    return wdbx.searchVectors(handle, allocator, query, top_k);
}

pub fn searchInto(
    handle: *DatabaseHandle,
    query: []const f32,
    top_k: usize,
    results: []SearchResult,
) usize {
    return wdbx.searchVectorsInto(handle, query, top_k, results);
}

pub fn remove(handle: *DatabaseHandle, id: u64) bool {
    return wdbx.deleteVector(handle, id);
}

pub fn update(handle: *DatabaseHandle, id: u64, vector: []const f32) !bool {
    return wdbx.updateVector(handle, id, vector);
}

pub fn get(handle: *DatabaseHandle, id: u64) ?VectorView {
    return wdbx.getVector(handle, id);
}

pub fn list(handle: *DatabaseHandle, allocator: std.mem.Allocator, limit: usize) ![]VectorView {
    return wdbx.listVectors(handle, allocator, limit);
}

pub fn stats(handle: *DatabaseHandle) Stats {
    return wdbx.getStats(handle);
}

pub fn diagnostics(handle: *DatabaseHandle) DiagnosticsInfo {
    return handle.db.diagnostics();
}

pub fn optimize(handle: *DatabaseHandle) !void {
    try wdbx.optimize(handle);
}

pub fn backup(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.backup(handle, path);
}

pub fn restore(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.restore(handle, path);
}

pub fn backupToPath(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.backupToPath(handle, path);
}

pub fn restoreFromPath(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.restoreFromPath(handle, path);
}

pub fn openFromFile(allocator: std.mem.Allocator, path: []const u8) !DatabaseHandle {
    const db = try storage.loadDatabase(allocator, path);
    return .{ .db = db };
}

pub fn openOrCreate(allocator: std.mem.Allocator, path: []const u8) !DatabaseHandle {
    const loaded = storage.loadDatabase(allocator, path);
    if (loaded) |db| {
        return .{ .db = db };
    } else |err| switch (err) {
        error.FileNotFound => return wdbx.createDatabase(allocator, path),
        else => return err,
    }
}

test "database module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test "database module exports stable compatibility surfaces" {
    _ = database.Database;
    _ = database.DatabaseError;
    _ = semantic_store.StoreHandle;
    _ = wdbx.DatabaseHandle;
    _ = batch.ImportFormat;
    _ = neural.Engine;
}
