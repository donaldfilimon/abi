//! Unified database interface providing high-level CRUD operations.
//!
//! Abstracts underlying database implementation and provides a consistent
//! API for vector storage, retrieval, and search operations.

const std = @import("std");
const database = @import("./database.zig");
const fs = @import("../../shared/utils/fs/mod.zig");

pub const UnifiedError = error{
    Unsupported,
};

pub const DatabaseHandle = struct {
    db: database.Database,
};

pub const SearchResult = database.SearchResult;
pub const VectorView = database.VectorView;
pub const Stats = database.Stats;

pub fn createDatabase(allocator: std.mem.Allocator, name: []const u8) !DatabaseHandle {
    return .{ .db = try database.Database.init(allocator, name) };
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
    try handle.db.saveToFile(safe_path);
}

pub fn restore(handle: *DatabaseHandle, path: []const u8) !void {
    if (!fs.isSafeBackupPath(path)) return fs.PathValidationError.InvalidPath;

    const allocator = handle.db.allocator;
    const safe_path = try fs.normalizeBackupPath(allocator, path);
    errdefer allocator.free(safe_path);

    const restored = try database.Database.loadFromFile(allocator, safe_path);
    handle.db.deinit();
    handle.db = restored;
}
