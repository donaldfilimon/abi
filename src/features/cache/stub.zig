//! Cache Stub Module
//!
//! API-compatible no-op implementations when cache is disabled.

const std = @import("std");
const core_config = @import("../../core/config/cache.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const CacheConfig = core_config.CacheConfig;
pub const EvictionPolicy = core_config.EvictionPolicy;

pub const CacheError = error{
    FeatureDisabled,
    CacheFull,
    KeyNotFound,
    InvalidTTL,
    OutOfMemory,
};

pub const CacheEntry = struct {
    key: []const u8 = "",
    value: []const u8 = "",
    ttl_ms: u64 = 0,
    created_at: u64 = 0,
};

pub const CacheStats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    entries: u32 = 0,
    memory_used: u64 = 0,
    evictions: u64 = 0,
    expired: u64 = 0,
};

pub const Context = stub_context.StubContext(CacheConfig);

pub fn init(_: std.mem.Allocator, _: CacheConfig) CacheError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

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
