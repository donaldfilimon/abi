//! Search Stub Module
//!
//! API-compatible no-op implementations when search is disabled.

const std = @import("std");
const core_config = @import("../../core/config/search.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const SearchConfig = core_config.SearchConfig;

pub const SearchError = error{
    FeatureDisabled,
    IndexNotFound,
    InvalidQuery,
    IndexCorrupted,
    OutOfMemory,
    IndexAlreadyExists,
    DocumentNotFound,
};

pub const SearchResult = struct {
    doc_id: []const u8 = "",
    score: f32 = 0.0,
    snippet: []const u8 = "",
};

pub const SearchIndex = struct {
    name: []const u8 = "",
    doc_count: u64 = 0,
    size_bytes: u64 = 0,
};

pub const SearchStats = struct {
    total_indexes: u32 = 0,
    total_documents: u64 = 0,
    total_terms: u64 = 0,
};

pub const Context = stub_context.StubContext(SearchConfig);

pub fn init(_: std.mem.Allocator, _: SearchConfig) SearchError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

pub fn createIndex(_: std.mem.Allocator, _: []const u8) SearchError!SearchIndex {
    return error.FeatureDisabled;
}
pub fn deleteIndex(_: []const u8) SearchError!void {
    return error.FeatureDisabled;
}
pub fn indexDocument(_: []const u8, _: []const u8, _: []const u8) SearchError!void {
    return error.FeatureDisabled;
}
pub fn deleteDocument(_: []const u8, _: []const u8) SearchError!bool {
    return error.FeatureDisabled;
}
pub fn query(_: std.mem.Allocator, _: []const u8, _: []const u8) SearchError![]SearchResult {
    return error.FeatureDisabled;
}
pub fn stats() SearchStats {
    return .{};
}
