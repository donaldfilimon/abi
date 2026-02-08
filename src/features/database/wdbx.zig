//! WDBX public surface built on top of the in-memory database and storage helpers.
const std = @import("std");
const database = @import("database.zig");
const storage = @import("storage_v2.zig"); // Unified storage with v1 fallback
const fs = @import("../../services/shared/utils.zig").fs;

pub const DatabaseHandle = struct {
    db: database.Database,
};

pub const SearchResult = database.SearchResult;
pub const VectorView = database.VectorView;
pub const Stats = database.Stats;
pub const DatabaseConfig = database.DatabaseConfig;
pub const BatchItem = database.Database.BatchItem;

pub fn createDatabase(allocator: std.mem.Allocator, name: []const u8) !DatabaseHandle {
    return .{ .db = try database.Database.init(allocator, name) };
}

pub fn createDatabaseWithConfig(
    allocator: std.mem.Allocator,
    name: []const u8,
    config: database.DatabaseConfig,
) !DatabaseHandle {
    return .{ .db = try database.Database.initWithConfig(allocator, name, config) };
}

pub fn connectDatabase(allocator: std.mem.Allocator, name: []const u8) !DatabaseHandle {
    return createDatabase(allocator, name);
}

pub fn closeDatabase(handle: *DatabaseHandle) void {
    handle.db.deinit();
    handle.* = undefined;
}

pub fn insertVector(
    handle: *DatabaseHandle,
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
) !void {
    try handle.db.insert(id, vector, metadata);
}

pub fn insertBatch(handle: *DatabaseHandle, items: []const BatchItem) !void {
    try handle.db.insertBatch(items);
}

pub fn searchVectors(
    handle: *DatabaseHandle,
    allocator: std.mem.Allocator,
    query: []const f32,
    top_k: usize,
) ![]SearchResult {
    return handle.db.search(allocator, query, top_k);
}

pub fn deleteVector(handle: *DatabaseHandle, id: u64) bool {
    return handle.db.delete(id);
}

pub fn updateVector(handle: *DatabaseHandle, id: u64, vector: []const f32) !bool {
    return handle.db.update(id, vector);
}

pub fn getVector(handle: *DatabaseHandle, id: u64) ?VectorView {
    return handle.db.get(id);
}

pub fn listVectors(
    handle: *DatabaseHandle,
    allocator: std.mem.Allocator,
    limit: usize,
) ![]VectorView {
    return handle.db.list(allocator, limit);
}

pub fn getStats(handle: *DatabaseHandle) Stats {
    return handle.db.stats();
}

pub fn optimize(handle: *DatabaseHandle) !void {
    handle.db.optimize();
}

pub fn backup(handle: *DatabaseHandle, path: []const u8) !void {
    if (!fs.isSafeBackupPath(path)) return fs.PathValidationError.InvalidPath;
    const safe_path = try fs.normalizeBackupPath(handle.db.allocator, path);
    defer handle.db.allocator.free(safe_path);
    try storage.saveDatabase(handle.db.allocator, &handle.db, safe_path);
}

pub fn restore(handle: *DatabaseHandle, path: []const u8) !void {
    if (!fs.isSafeBackupPath(path)) return fs.PathValidationError.InvalidPath;

    const allocator = handle.db.allocator;
    const safe_path = try fs.normalizeBackupPath(allocator, path);
    defer allocator.free(safe_path);

    // Uses unified API which auto-detects v2 or v1 format
    const restored = try storage.loadDatabase(allocator, safe_path);
    handle.db.deinit();
    handle.db = restored;
}
