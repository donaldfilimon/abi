const std = @import("std");
const core_config = @import("../../core/config/platform.zig");

pub const CacheConfig = core_config.CacheConfig;
pub const EvictionPolicy = core_config.EvictionPolicy;

/// Errors returned by cache operations.
pub const CacheError = error{
    FeatureDisabled,
    CacheFull,
    KeyNotFound,
    InvalidTTL,
    OutOfMemory,
};

/// A single cached key-value pair with time-to-live metadata.
pub const CacheEntry = struct {
    key: []const u8 = "",
    value: []const u8 = "",
    /// Time-to-live in milliseconds (0 = no expiry).
    ttl_ms: u64 = 0,
    /// Timestamp when the entry was inserted (epoch ms).
    created_at: u64 = 0,
};

/// Aggregate statistics for the cache instance.
pub const CacheStats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    /// Current number of live entries.
    entries: u32 = 0,
    /// Approximate heap bytes consumed by keys + values.
    memory_used: u64 = 0,
    evictions: u64 = 0,
    expired: u64 = 0,
};
