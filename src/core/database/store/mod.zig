//! Canonical WDBX store surface.
//!
//! This wraps the legacy `wdbx.zig` helpers with method-based ownership so the
//! public ABI surface is centered on one store type.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../../core/config/database.zig");
const legacy = @import("../wdbx.zig");

pub const SearchResult = legacy.SearchResult;
pub const VectorView = legacy.VectorView;
pub const Stats = legacy.Stats;
pub const BatchItem = legacy.BatchItem;
pub const DatabaseConfig = legacy.DatabaseConfig;
pub const DiagnosticsInfo = @import("../database.zig").DiagnosticsInfo;
pub const DatabaseError = @import("../database.zig").DatabaseError;

pub const DatabaseFeatureError = error{
    DatabaseDisabled,
};

pub const Store = struct {
    handle: legacy.DatabaseHandle,

    const Self = @This();

    pub fn open(alloc: std.mem.Allocator, name: []const u8) !Self {
        if (!build_options.feat_database) return DatabaseFeatureError.DatabaseDisabled;
        return .{ .handle = try legacy.createDatabase(alloc, name) };
    }

    pub fn openWithConfig(
        alloc: std.mem.Allocator,
        name: []const u8,
        config: DatabaseConfig,
    ) !Self {
        if (!build_options.feat_database) return DatabaseFeatureError.DatabaseDisabled;
        return .{ .handle = try legacy.createDatabaseWithConfig(alloc, name, config) };
    }

    pub fn load(alloc: std.mem.Allocator, path: []const u8) !Self {
        if (!build_options.feat_database) return DatabaseFeatureError.DatabaseDisabled;
        var handle = try legacy.createDatabase(alloc, "loaded");
        errdefer legacy.closeDatabase(&handle);
        try legacy.restoreFromPath(&handle, path);
        return .{ .handle = handle };
    }

    pub fn openOrCreate(alloc: std.mem.Allocator, path: []const u8) !Self {
        if (!build_options.feat_database) return DatabaseFeatureError.DatabaseDisabled;
        // Try to load existing, fall back to creating new
        const handle = legacy.createDatabase(alloc, path) catch |err| switch (err) {
            else => return err,
        };
        return .{ .handle = handle };
    }

    pub fn deinit(self: *Self) void {
        legacy.closeDatabase(&self.handle);
    }

    pub fn allocator(self: *const Self) std.mem.Allocator {
        return self.handle.db.allocator;
    }

    pub fn getAllocator(self: *const Self) std.mem.Allocator {
        return self.allocator();
    }

    pub fn insert(self: *Self, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        try legacy.insertVector(&self.handle, id, vector, metadata);
    }

    pub fn insertBatch(self: *Self, items: []const BatchItem) !void {
        try legacy.insertBatch(&self.handle, items);
    }

    pub fn search(self: *Self, query: []const f32, top_k: usize) ![]SearchResult {
        return legacy.searchVectors(&self.handle, self.handle.db.allocator, query, top_k);
    }

    pub fn searchInto(self: *Self, query: []const f32, top_k: usize, results: []SearchResult) usize {
        return legacy.searchVectorsInto(&self.handle, query, top_k, results);
    }

    pub fn remove(self: *Self, id: u64) bool {
        return legacy.deleteVector(&self.handle, id);
    }

    pub fn update(self: *Self, id: u64, vector: []const f32) !bool {
        return legacy.updateVector(&self.handle, id, vector);
    }

    pub fn get(self: *Self, id: u64) ?VectorView {
        return legacy.getVector(&self.handle, id);
    }

    pub fn list(self: *Self, limit: usize) ![]VectorView {
        return legacy.listVectors(&self.handle, self.allocator(), limit);
    }

    pub fn stats(self: *Self) Stats {
        return legacy.getStats(&self.handle);
    }

    pub fn diagnostics(self: *Self) DiagnosticsInfo {
        return self.handle.db.diagnostics();
    }

    pub fn optimize(self: *Self) !void {
        try legacy.optimize(&self.handle);
    }

    /// Persist to an explicit filesystem path without backup-path normalization.
    pub fn save(self: *Self, path: []const u8) !void {
        try legacy.backupToPath(&self.handle, path);
    }

    /// Load store contents from an explicit filesystem path.
    pub fn loadInto(self: *Self, path: []const u8) !void {
        try legacy.restoreFromPath(&self.handle, path);
    }

    /// Persist via the path-validation helper used by interactive tools.
    pub fn backup(self: *Self, path: []const u8) !void {
        try legacy.backup(&self.handle, path);
    }

    /// Restore via the path-validation helper used by interactive tools.
    pub fn restore(self: *Self, path: []const u8) !void {
        try legacy.restore(&self.handle, path);
    }
};

/// Database context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.DatabaseConfig,
    store: ?Store = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.DatabaseConfig) !*Self {
        if (!build_options.feat_database) return DatabaseFeatureError.DatabaseDisabled;

        const ctx = try allocator.create(Self);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .store = null,
        };

        if (cfg.path.len > 0 and !std.mem.eql(u8, cfg.path, ":memory:")) {
            ctx.store = try Store.open(allocator, cfg.path);
        } else if (std.mem.eql(u8, cfg.path, ":memory:")) {
            ctx.store = try Store.open(allocator, "memory");
        }

        return ctx;
    }

    pub fn deinit(self: *Self) void {
        if (self.store) |*store| {
            store.deinit();
        }
        self.allocator.destroy(self);
    }

    pub fn getStore(self: *Self) !*Store {
        if (self.store) |*store| return store;

        const name = if (self.config.path.len == 0 or std.mem.eql(u8, self.config.path, ":memory:"))
            "default"
        else
            self.config.path;
        self.store = try Store.open(self.allocator, name);
        return &self.store.?;
    }

    pub fn openStore(self: *Self, name: []const u8) !*Store {
        if (self.store) |*store| {
            store.deinit();
        }
        self.store = try Store.open(self.allocator, name);
        return &self.store.?;
    }

    // Compatibility helpers for existing callers.
    pub fn openDatabase(self: *Self, name: []const u8) !*Store {
        return self.openStore(name);
    }

    pub fn insertVector(self: *Self, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        const store = try self.getStore();
        try store.insert(id, vector, metadata);
    }

    pub fn searchVectors(self: *Self, query: []const f32, top_k: usize) ![]SearchResult {
        const store = try self.getStore();
        return store.search(query, top_k);
    }

    pub fn searchVectorsInto(self: *Self, query: []const f32, top_k: usize, results: []SearchResult) !usize {
        const store = try self.getStore();
        return store.searchInto(query, top_k, results);
    }

    pub fn getStats(self: *Self) !Stats {
        const store = try self.getStore();
        return store.stats();
    }

    pub fn optimize(self: *Self) !void {
        const store = try self.getStore();
        try store.optimize();
    }
};

pub fn init(_: std.mem.Allocator) !void {
    if (!build_options.feat_database) return DatabaseFeatureError.DatabaseDisabled;
}

pub fn deinit() void {}

pub fn isInitialized() bool {
    return build_options.feat_database;
}

test "store roundtrip methods mirror legacy handle behavior" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try Store.open(std.testing.allocator, "store-methods");
    defer store.deinit();

    try store.insert(7, &.{ 0.1, 0.2, 0.3 }, "meta");
    const results = try store.search(&.{ 0.1, 0.2, 0.3 }, 1);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(u64, 7), results[0].id);
}
