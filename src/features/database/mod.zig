//! Database feature facade and convenience helpers.
const std = @import("std");
const build_options = @import("build_options");

pub const database = @import("database.zig");
pub const db_helpers = @import("db_helpers.zig");
pub const storage = @import("storage.zig");
pub const wdbx = @import("wdbx.zig");
pub const cli = @import("cli.zig");
pub const http = @import("http.zig");

pub const Database = database.Database;
pub const DatabaseHandle = wdbx.DatabaseHandle;
pub const SearchResult = wdbx.SearchResult;
pub const VectorView = wdbx.VectorView;
pub const Stats = wdbx.Stats;

pub const DatabaseFeatureError = error{
    DatabaseDisabled,
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
    return build_options.enable_database;
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

pub fn optimize(handle: *DatabaseHandle) !void {
    try wdbx.optimize(handle);
}

pub fn backup(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.backup(handle, path);
}

pub fn restore(handle: *DatabaseHandle, path: []const u8) !void {
    try wdbx.restore(handle, path);
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
