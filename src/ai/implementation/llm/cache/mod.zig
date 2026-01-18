//! Caching module for efficient autoregressive generation.
//!
//! Provides KV (key-value) cache for storing past attention states,
//! enabling O(1) per-token generation instead of O(n).
//!
//! Features:
//! - Basic KV cache with pre-allocation
//! - Sliding window support (ring buffer)
//! - Paged attention for efficient memory management

const std = @import("std");

pub const kv_cache = @import("kv_cache.zig");
pub const ring_buffer = @import("ring_buffer.zig");
pub const paged_kv_cache = @import("paged_kv_cache.zig");

// Re-exports - Basic KV cache
pub const KvCache = kv_cache.KvCache;
pub const KvCacheConfig = kv_cache.KvCacheConfig;
pub const LayerKvCache = kv_cache.LayerKvCache;
pub const RingBuffer = ring_buffer.RingBuffer;

// Re-exports - Paged KV cache
pub const PagedKvCache = paged_kv_cache.PagedKvCache;
pub const PagedKvCacheConfig = paged_kv_cache.PagedKvCacheConfig;
pub const KvPage = paged_kv_cache.KvPage;
pub const SequenceKvState = paged_kv_cache.SequenceKvState;
pub const PagedCacheStats = paged_kv_cache.PagedCacheStats;

test "cache module imports" {
    _ = kv_cache;
    _ = ring_buffer;
    _ = paged_kv_cache;
}
