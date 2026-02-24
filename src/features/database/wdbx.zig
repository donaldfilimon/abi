//! WDBX public surface built on top of the in-memory database and storage helpers.
const std = @import("std");
const database = @import("database.zig");
const storage = @import("storage.zig"); // Unified storage with v1 fallback
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

pub fn searchVectorsInto(
    handle: *DatabaseHandle,
    query: []const f32,
    top_k: usize,
    results: []SearchResult,
) usize {
    return handle.db.searchInto(query, top_k, results);
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

fn ensureParentDirExists(allocator: std.mem.Allocator, path: []const u8) !void {
    const dir_path = std.fs.path.dirname(path) orelse return;
    if (dir_path.len == 0) return;

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().createDirPath(io, dir_path) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

pub fn backupToPath(handle: *DatabaseHandle, path: []const u8) !void {
    try ensureParentDirExists(handle.db.allocator, path);
    try storage.saveDatabase(handle.db.allocator, &handle.db, path);
}

pub fn restoreFromPath(handle: *DatabaseHandle, path: []const u8) !void {
    const allocator = handle.db.allocator;
    const restored = try storage.loadDatabase(allocator, path);
    handle.db.deinit();
    handle.db = restored;
}

pub fn backup(handle: *DatabaseHandle, path: []const u8) !void {
    if (!fs.isSafeBackupPath(path)) return fs.PathValidationError.InvalidPath;
    const safe_path = try fs.normalizeBackupPath(handle.db.allocator, path);
    defer handle.db.allocator.free(safe_path);
    try backupToPath(handle, safe_path);
}

pub fn restore(handle: *DatabaseHandle, path: []const u8) !void {
    if (!fs.isSafeBackupPath(path)) return fs.PathValidationError.InvalidPath;

    const allocator = handle.db.allocator;
    const safe_path = try fs.normalizeBackupPath(allocator, path);
    defer allocator.free(safe_path);

    // Uses unified API which auto-detects v2 or v1 format
    try restoreFromPath(handle, safe_path);
}

test "backupToPath and restoreFromPath roundtrip with nested directories" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", allocator);
    defer allocator.free(root);

    const db_path = try std.fs.path.join(allocator, &.{ root, "nested", "deep", "vectors.wdbx" });
    defer allocator.free(db_path);

    var source = try createDatabase(allocator, "source-db");
    defer closeDatabase(&source);
    try insertVector(&source, 42, &.{ 0.1, 0.2, 0.3 }, "meta");
    try backupToPath(&source, db_path);

    var restored = try createDatabase(allocator, "restored-db");
    defer closeDatabase(&restored);
    try restoreFromPath(&restored, db_path);

    const view = getVector(&restored, 42) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 42), view.id);
    try std.testing.expectEqual(@as(usize, 3), view.vector.len);
    try std.testing.expectEqualStrings("meta", view.metadata.?);
}

test "safe backup and restore reject unsafe paths" {
    const allocator = std.testing.allocator;

    var handle = try createDatabase(allocator, "safe-path-db");
    defer closeDatabase(&handle);

    try std.testing.expectError(fs.PathValidationError.InvalidPath, backup(&handle, "../bad.wdbx"));
    try std.testing.expectError(fs.PathValidationError.InvalidPath, restore(&handle, "../bad.wdbx"));
}

test {
    std.testing.refAllDecls(@This());
}
