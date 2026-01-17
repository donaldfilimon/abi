//! Database Stub Module

const std = @import("std");
const config_module = @import("../config.zig");

pub const Error = error{
    DatabaseDisabled,
    ConnectionFailed,
    QueryFailed,
    IndexError,
    StorageError,
};

pub const Database = struct {};
pub const helpers = struct {};
pub const cli = struct {
    pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) Error!void {
        _ = allocator;
        _ = args;
        return error.DatabaseDisabled;
    }
};
pub const http = struct {};
pub const wdbx = struct {};

pub const Context = struct {
    pub const SearchResult = struct {
        id: []const u8,
        score: f32,
        metadata: ?[]const u8,
    };

    pub fn init(_: std.mem.Allocator, _: config_module.DatabaseConfig) Error!*Context {
        return error.DatabaseDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn open(_: *Context) Error!void {
        return error.DatabaseDisabled;
    }

    pub fn insertVector(_: *Context, _: []const u8, _: []const f32, _: ?[]const u8) Error!void {
        return error.DatabaseDisabled;
    }

    pub fn searchVectors(_: *Context, _: []const f32, _: usize) Error![]SearchResult {
        return error.DatabaseDisabled;
    }

    pub fn deleteVector(_: *Context, _: []const u8) Error!void {
        return error.DatabaseDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.DatabaseDisabled;
}

pub fn deinit() void {}

pub fn createDatabase(_: anytype) Error!void {
    return error.DatabaseDisabled;
}
pub fn connectDatabase(_: anytype) Error!void {
    return error.DatabaseDisabled;
}
pub fn closeDatabase(_: anytype) void {}
pub fn insertVector(_: anytype, _: anytype, _: anytype, _: anytype) Error!void {
    return error.DatabaseDisabled;
}
pub fn searchVectors(_: anytype, _: anytype, _: anytype) Error!void {
    return error.DatabaseDisabled;
}
pub fn deleteVector(_: anytype, _: anytype) Error!void {
    return error.DatabaseDisabled;
}
pub fn getStats(_: anytype) Error!void {
    return error.DatabaseDisabled;
}
pub fn optimize(_: anytype) Error!void {
    return error.DatabaseDisabled;
}

pub const DatabaseHandle = struct {
    db: ?*anyopaque = null,
};

pub fn openOrCreate(allocator: std.mem.Allocator, path: []const u8) Error!DatabaseHandle {
    _ = allocator;
    _ = path;
    return error.DatabaseDisabled;
}

pub fn insert(handle: *DatabaseHandle, id: u64, vector: []const f32, metadata: ?[]const u8) Error!void {
    _ = handle;
    _ = id;
    _ = vector;
    _ = metadata;
    return error.DatabaseDisabled;
}
