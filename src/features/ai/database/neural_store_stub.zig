//! Stubbed AI-local neural ingestion adapter.

const std = @import("std");

pub const Metadata = struct {
    text: []const u8 = "",
    category: ?[]const u8 = null,
    tags: []const []const u8 = &.{},
    score: f32 = 1.0,
    extra: ?[]const u8 = null,
};

pub const SearchOptions = struct {
    k: usize = 10,
};

pub const SearchResult = struct {
    id: []const u8 = "",
    similarity: f32 = 0.0,
    distance: f32 = 0.0,
    metadata: Metadata = .{},
    vector: []const f32 = &.{},
};

pub const WritePolicy = enum {
    allow_duplicate,
    skip_if_same_content,
    replace_by_id,
};

pub const EngineVector = struct {
    id: []const u8 = "",
    vec: []const f32 = &.{},
    metadata: Metadata = .{},
    content_fingerprint: ?u64 = null,
    access_score: f32 = 1.0,
};

pub const Engine = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) !Engine {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Engine) void {}
    pub fn indexByVector(_: *Engine, _: []const u8, _: []const f32, _: Metadata) !void {
        return error.FeatureDisabled;
    }
    pub fn search(_: *Engine, _: []const u8, _: SearchOptions) ![]SearchResult {
        return error.FeatureDisabled;
    }
};

pub fn save(_: *Engine, _: []const u8) !void {
    return error.FeatureDisabled;
}

pub fn load(_: std.mem.Allocator, _: []const u8) !Engine {
    return error.FeatureDisabled;
}
