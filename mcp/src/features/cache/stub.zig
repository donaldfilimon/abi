//! Cache Stub Module
//!
//! API-compatible no-op implementations when cache is disabled.

const std = @import("std");
const stub_context = @import("../core/stub_helpers.zig");
pub const types = @import("types.zig");

pub const CacheConfig = types.CacheConfig;
pub const EvictionPolicy = types.EvictionPolicy;
pub const CacheError = types.CacheError;
pub const Error = CacheError;
pub const CacheEntry = types.CacheEntry;
pub const CacheStats = types.CacheStats;

const feature = stub_context.StubFeature(CacheConfig, CacheError);
pub const Context = feature.Context;
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;

pub fn get(_: []const u8) CacheError!?[]const u8 {
    return error.FeatureDisabled;
}
pub fn put(_: []const u8, _: []const u8) CacheError!void {
    return error.FeatureDisabled;
}
pub fn putWithTtl(_: []const u8, _: []const u8, _: u64) CacheError!void {
    return error.FeatureDisabled;
}
pub fn delete(_: []const u8) CacheError!bool {
    return error.FeatureDisabled;
}
pub fn contains(_: []const u8) bool {
    return false;
}
pub fn clear() void {}
pub fn size() u32 {
    return 0;
}
pub fn stats() CacheStats {
    return .{};
}

test {
    std.testing.refAllDecls(@This());
}
