//! Search Stub Module
//!
//! API-compatible no-op implementations when search is disabled.

const std = @import("std");
const stub_context = @import("../../core/stub_helpers.zig");
pub const types = @import("types.zig");

pub const SearchConfig = types.SearchConfig;
pub const SearchError = types.SearchError;
pub const Error = SearchError;
pub const SearchResult = types.SearchResult;
pub const SearchIndex = types.SearchIndex;
pub const SearchStats = types.SearchStats;

const feature = stub_context.StubFeature(SearchConfig, SearchError);
pub const Context = feature.Context;
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;

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

// ── Sub-module stubs (parity with mod.zig) ────────────────────────────
pub const tokenizer = struct {
    pub fn tokenize(_: std.mem.Allocator, _: []const u8) SearchError![]const []const u8 {
        return error.FeatureDisabled;
    }
};
pub const scoring = struct {
    pub fn bm25(_: f64, _: f64, _: f64) f64 {
        return 0.0;
    }
};
pub const inverted_index = struct {
    pub fn create(_: std.mem.Allocator) SearchError!void {
        return error.FeatureDisabled;
    }
};
pub const persistence = struct {
    pub fn save(_: std.mem.Allocator, _: []const u8, _: []const u8) SearchError!void {
        return error.FeatureDisabled;
    }
    pub fn load(_: std.mem.Allocator, _: []const u8, _: []const u8) SearchError!void {
        return error.FeatureDisabled;
    }
};
pub fn saveIndex(_: std.mem.Allocator, _: []const u8, _: []const u8) SearchError!void {
    return error.FeatureDisabled;
}
pub fn loadIndex(_: std.mem.Allocator, _: []const u8, _: []const u8) SearchError!void {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
