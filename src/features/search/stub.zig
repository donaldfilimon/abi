//! Search Stub Module
//!
//! API-compatible no-op implementations when search is disabled.

const std = @import("std");
const stub_context = @import("../../core/stub_context.zig");
const types = @import("types.zig");

pub const SearchConfig = types.SearchConfig;
pub const SearchError = types.SearchError;
pub const SearchResult = types.SearchResult;
pub const SearchIndex = types.SearchIndex;
pub const SearchStats = types.SearchStats;

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

test {
    std.testing.refAllDecls(@This());
}
