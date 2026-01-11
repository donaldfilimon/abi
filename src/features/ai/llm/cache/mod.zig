//! Caching module for efficient autoregressive generation.
//!
//! Provides KV (key-value) cache for storing past attention states,
//! enabling O(1) per-token generation instead of O(n).

const std = @import("std");

pub const kv_cache = @import("kv_cache.zig");
pub const ring_buffer = @import("ring_buffer.zig");

// Re-exports
pub const KvCache = kv_cache.KvCache;
pub const KvCacheConfig = kv_cache.KvCacheConfig;
pub const LayerKvCache = kv_cache.LayerKvCache;
pub const RingBuffer = ring_buffer.RingBuffer;

test "cache module imports" {
    _ = kv_cache;
    _ = ring_buffer;
}
