const std = @import("std");
const gpu = @import("../gpu/stub.zig");

pub const MAX_LAYERS = 4;

pub const VectorRecord = struct {
    id: u32,
    values: []f32,
};

pub const SearchResult = struct {
    id: u32,
    score: f32,
};

pub const ConversationBlock = struct {
    id: [32]u8,
    prev_id: [32]u8,
    timestamp_ms: i64,
    profile: []const u8,
    query_id: u32,
    response_id: u32,
    metadata: []const u8,
};

pub const AccelerationStatus = struct {
    backend: gpu.Backend,
    mode: gpu.ExecutionMode,
    message: []const u8,
};

pub const Store = struct {
    pub fn init(a: std.mem.Allocator) Store {
        _ = a;
        return .{};
    }
    pub fn deinit(self: *Store) void {
        _ = self;
    }
    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        _ = self;
        _ = key;
        _ = val;
    }
    pub fn get(self: *const Store, key: []const u8) ?[]const u8 {
        _ = self;
        _ = key;
        return null;
    }
    pub fn count(self: *const Store) usize {
        _ = self;
        return 0;
    }
    pub fn putVector(self: *Store, values: []const f32) !u32 {
        _ = self;
        _ = values;
        return 0;
    }
    pub fn search(self: *Store, query: []const f32, limit: usize) ![]SearchResult {
        _ = self;
        _ = query;
        _ = limit;
        return error.FeatureDisabled;
    }
    pub fn appendBlock(self: *Store, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![32]u8 {
        _ = self;
        _ = profile;
        _ = query_id;
        _ = response_id;
        _ = metadata;
        return [_]u8{0} * *32;
    }

    pub fn blockCount(self: *const Store) usize {
        _ = self;
        return 0;
    }
    pub fn accelerationStatus(self: *const Store) AccelerationStatus {
        _ = self;
        return .{ .backend = .simulated, .mode = .cpu_fallback, .message = "wdbx feature is disabled" };
    }
};
