//! Stub for Database feature when disabled
const std = @import("std");

pub const DatabaseHandle = struct {
    db: ?*anyopaque,
};

pub const SearchResult = struct {
    id: u64,
    score: f32,
};

pub const Stats = struct {
    count: usize,
    dimension: usize,
};

pub fn openOrCreate(allocator: std.mem.Allocator, path: []const u8) !DatabaseHandle {
    _ = allocator;
    _ = path;
    return error.DatabaseDisabled;
}

pub fn insert(handle: *DatabaseHandle, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
    _ = handle;
    _ = id;
    _ = vector;
    _ = metadata;
    return error.DatabaseDisabled;
}

pub fn search(handle: *DatabaseHandle, allocator: std.mem.Allocator, query: []const f32, top_k: usize) ![]SearchResult {
    _ = handle;
    _ = allocator;
    _ = query;
    _ = top_k;
    return error.DatabaseDisabled;
}

pub fn stats(handle: *DatabaseHandle) Stats {
    _ = handle;
    return Stats{ .count = 0, .dimension = 0 };
}

pub fn close(handle: *DatabaseHandle) void {
    _ = handle;
}

pub fn isEnabled() bool {
    return false;
}

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    return error.DatabaseDisabled;
}

pub fn deinit() void {}
