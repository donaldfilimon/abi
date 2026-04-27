//! Stubbed WDBX store surface when the database feature is disabled.

const std = @import("std");
const config_module = @import("../../config/database.zig");

pub const SearchResult = struct {
    id: u64 = 0,
    score: f32 = 0.0,
};

pub const VectorView = struct {
    id: u64 = 0,
    vector: []const f32 = &.{},
    metadata: ?[]const u8 = null,
};

pub const Stats = struct {
    count: usize = 0,
    dimension: usize = 0,
    memory_bytes: usize = 0,
    norm_cache_enabled: bool = false,
};

pub const BatchItem = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8 = null,
};

pub const DatabaseConfig = config_module.DatabaseConfig;
pub const DiagnosticsInfo = @import("../stubs/types.zig").DiagnosticsInfo;
pub const DatabaseError = error{
    FeatureDisabled,
};

pub const DatabaseFeatureError = error{
    DatabaseDisabled,
};

pub const Store = struct {
    pub fn open(_: std.mem.Allocator, _: []const u8) !Store {
        return error.DatabaseDisabled;
    }

    pub fn openWithConfig(_: std.mem.Allocator, _: []const u8, _: DatabaseConfig) !Store {
        return error.DatabaseDisabled;
    }

    pub fn load(_: std.mem.Allocator, _: []const u8) !Store {
        return error.DatabaseDisabled;
    }

    pub fn openOrCreate(_: std.mem.Allocator, _: []const u8) !Store {
        return error.DatabaseDisabled;
    }

    pub fn deinit(_: *Store) void {}
    pub fn allocator(_: *const Store) std.mem.Allocator {
        return std.heap.page_allocator;
    }
    pub fn insert(_: *Store, _: u64, _: []const f32, _: ?[]const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn insertBatch(_: *Store, _: []const BatchItem) !void {
        return error.DatabaseDisabled;
    }
    pub fn search(_: *Store, _: []const f32, _: usize) ![]SearchResult {
        return error.DatabaseDisabled;
    }
    pub fn searchInto(_: *Store, _: []const f32, _: usize, _: []SearchResult) usize {
        return 0;
    }
    pub fn remove(_: *Store, _: u64) bool {
        return false;
    }
    pub fn update(_: *Store, _: u64, _: []const f32) !bool {
        return error.DatabaseDisabled;
    }
    pub fn get(_: *Store, _: u64) ?VectorView {
        return null;
    }
    pub fn list(_: *Store, _: usize) ![]VectorView {
        return error.DatabaseDisabled;
    }
    pub fn stats(_: *Store) Stats {
        return .{};
    }
    pub fn diagnostics(_: *Store) DiagnosticsInfo {
        return .{};
    }
    pub fn optimize(_: *Store) !void {
        return error.DatabaseDisabled;
    }
    pub fn save(_: *Store, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn loadInto(_: *Store, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn backup(_: *Store, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn restore(_: *Store, _: []const u8) !void {
        return error.DatabaseDisabled;
    }
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: DatabaseConfig,
    store: ?Store = null,

    pub fn init(_: std.mem.Allocator, _: DatabaseConfig) !*Context {
        return error.DatabaseDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getStore(_: *Context) !*Store {
        return error.DatabaseDisabled;
    }
    pub fn openStore(_: *Context, _: []const u8) !*Store {
        return error.DatabaseDisabled;
    }
    pub fn openDatabase(_: *Context, _: []const u8) !*Store {
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

pub fn init(_: std.mem.Allocator) !void {
    return error.DatabaseDisabled;
}

pub fn deinit() void {}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
