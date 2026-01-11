//! Key-Value cache for efficient autoregressive generation.
//!
//! The KV cache stores the key and value projections from previous tokens,
//! allowing attention to be computed incrementally without recomputing
//! the entire sequence each time.

const std = @import("std");

/// Configuration for KV cache.
pub const KvCacheConfig = struct {
    /// Number of transformer layers
    num_layers: u32,
    /// Number of KV heads per layer
    num_kv_heads: u32,
    /// Dimension per head
    head_dim: u32,
    /// Maximum sequence length to cache
    max_seq_len: u32,
    /// Data type for cache (affects memory usage)
    dtype: CacheDType = .f32,

    pub const CacheDType = enum {
        f32,
        f16,
        bf16,
    };

    /// Calculate total memory requirement in bytes.
    pub fn memoryBytes(self: KvCacheConfig) u64 {
        const kv_dim = @as(u64, self.num_kv_heads) * self.head_dim;
        const per_layer = self.max_seq_len * kv_dim * 2; // K and V
        const total_elements = per_layer * self.num_layers;

        return switch (self.dtype) {
            .f32 => total_elements * 4,
            .f16, .bf16 => total_elements * 2,
        };
    }
};

/// KV cache for a single layer.
pub const LayerKvCache = struct {
    /// Key cache: [max_seq_len, num_kv_heads * head_dim]
    k_cache: []f32,
    /// Value cache: [max_seq_len, num_kv_heads * head_dim]
    v_cache: []f32,
    /// Current length (number of cached tokens)
    len: u32,
    /// Configuration
    kv_dim: u32,
    max_len: u32,

    pub fn init(allocator: std.mem.Allocator, num_kv_heads: u32, head_dim: u32, max_seq_len: u32) !LayerKvCache {
        const kv_dim = num_kv_heads * head_dim;
        const cache_size = @as(usize, max_seq_len) * kv_dim;

        const k_cache = try allocator.alloc(f32, cache_size);
        errdefer allocator.free(k_cache);
        const v_cache = try allocator.alloc(f32, cache_size);

        @memset(k_cache, 0);
        @memset(v_cache, 0);

        return .{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .len = 0,
            .kv_dim = kv_dim,
            .max_len = max_seq_len,
        };
    }

    pub fn deinit(self: *LayerKvCache, allocator: std.mem.Allocator) void {
        allocator.free(self.k_cache);
        allocator.free(self.v_cache);
        self.* = undefined;
    }

    /// Update cache with new K, V for a single position.
    pub fn update(self: *LayerKvCache, k: []const f32, v: []const f32, pos: u32) void {
        if (pos >= self.max_len) return;

        const offset = @as(usize, pos) * self.kv_dim;
        @memcpy(self.k_cache[offset .. offset + self.kv_dim], k);
        @memcpy(self.v_cache[offset .. offset + self.kv_dim], v);

        if (pos >= self.len) {
            self.len = pos + 1;
        }
    }

    /// Update cache with multiple positions (batch update).
    pub fn updateBatch(self: *LayerKvCache, k: []const f32, v: []const f32, start_pos: u32, seq_len: u32) void {
        for (0..seq_len) |i| {
            const pos = start_pos + @as(u32, @intCast(i));
            if (pos >= self.max_len) break;

            const src_offset = i * self.kv_dim;
            const dst_offset = @as(usize, pos) * self.kv_dim;

            @memcpy(
                self.k_cache[dst_offset .. dst_offset + self.kv_dim],
                k[src_offset .. src_offset + self.kv_dim],
            );
            @memcpy(
                self.v_cache[dst_offset .. dst_offset + self.kv_dim],
                v[src_offset .. src_offset + self.kv_dim],
            );
        }

        const end_pos = start_pos + seq_len;
        if (end_pos > self.len) {
            self.len = @min(end_pos, self.max_len);
        }
    }

    /// Get cached K values up to current length.
    pub fn getK(self: *const LayerKvCache) []const f32 {
        return self.k_cache[0 .. @as(usize, self.len) * self.kv_dim];
    }

    /// Get cached V values up to current length.
    pub fn getV(self: *const LayerKvCache) []const f32 {
        return self.v_cache[0 .. @as(usize, self.len) * self.kv_dim];
    }

    /// Get K at a specific position.
    pub fn getKAt(self: *const LayerKvCache, pos: u32) ?[]const f32 {
        if (pos >= self.len) return null;
        const offset = @as(usize, pos) * self.kv_dim;
        return self.k_cache[offset .. offset + self.kv_dim];
    }

    /// Get V at a specific position.
    pub fn getVAt(self: *const LayerKvCache, pos: u32) ?[]const f32 {
        if (pos >= self.len) return null;
        const offset = @as(usize, pos) * self.kv_dim;
        return self.v_cache[offset .. offset + self.kv_dim];
    }

    /// Clear the cache.
    pub fn clear(self: *LayerKvCache) void {
        self.len = 0;
    }

    /// Get current cache length.
    pub fn length(self: *const LayerKvCache) u32 {
        return self.len;
    }

    /// Check if cache is full.
    pub fn isFull(self: *const LayerKvCache) bool {
        return self.len >= self.max_len;
    }

    /// Get remaining capacity.
    pub fn remaining(self: *const LayerKvCache) u32 {
        return self.max_len - self.len;
    }
};

/// Full KV cache for all layers.
pub const KvCache = struct {
    allocator: std.mem.Allocator,
    layers: []LayerKvCache,
    config: KvCacheConfig,

    pub fn init(allocator: std.mem.Allocator, config: KvCacheConfig) !KvCache {
        const layers = try allocator.alloc(LayerKvCache, config.num_layers);
        errdefer allocator.free(layers);

        for (0..config.num_layers) |i| {
            layers[i] = try LayerKvCache.init(
                allocator,
                config.num_kv_heads,
                config.head_dim,
                config.max_seq_len,
            );
        }

        return .{
            .allocator = allocator,
            .layers = layers,
            .config = config,
        };
    }

    pub fn deinit(self: *KvCache) void {
        for (self.layers) |*layer| {
            layer.deinit(self.allocator);
        }
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    /// Get cache for a specific layer.
    pub fn getLayer(self: *KvCache, layer_idx: u32) *LayerKvCache {
        return &self.layers[layer_idx];
    }

    /// Get cache for a specific layer (const).
    pub fn getLayerConst(self: *const KvCache, layer_idx: u32) *const LayerKvCache {
        return &self.layers[layer_idx];
    }

    /// Update K, V for a layer at a position.
    pub fn update(self: *KvCache, layer_idx: u32, k: []const f32, v: []const f32, pos: u32) void {
        self.layers[layer_idx].update(k, v, pos);
    }

    /// Clear all layer caches.
    pub fn clear(self: *KvCache) void {
        for (self.layers) |*layer| {
            layer.clear();
        }
    }

    /// Get current sequence length (from first layer).
    pub fn sequenceLength(self: *const KvCache) u32 {
        if (self.layers.len == 0) return 0;
        return self.layers[0].len;
    }

    /// Check if any layer's cache is full.
    pub fn isFull(self: *const KvCache) bool {
        if (self.layers.len == 0) return true;
        return self.layers[0].isFull();
    }

    /// Get memory usage in bytes.
    pub fn memoryUsed(self: *const KvCache) u64 {
        var total: u64 = 0;
        for (self.layers) |layer| {
            total += @as(u64, layer.len) * layer.kv_dim * 2 * @sizeOf(f32);
        }
        return total;
    }

    /// Get statistics about cache usage.
    pub fn getStats(self: *const KvCache) CacheStats {
        return .{
            .num_layers = @intCast(self.layers.len),
            .sequence_length = self.sequenceLength(),
            .max_sequence_length = self.config.max_seq_len,
            .memory_bytes = self.memoryUsed(),
            .max_memory_bytes = self.config.memoryBytes(),
            .utilization = if (self.config.max_seq_len > 0)
                @as(f32, @floatFromInt(self.sequenceLength())) /
                    @as(f32, @floatFromInt(self.config.max_seq_len))
            else
                0,
        };
    }
};

pub const CacheStats = struct {
    num_layers: u32,
    sequence_length: u32,
    max_sequence_length: u32,
    memory_bytes: u64,
    max_memory_bytes: u64,
    utilization: f32,
};

test "layer kv cache basic" {
    const allocator = std.testing.allocator;

    var cache = try LayerKvCache.init(allocator, 4, 64, 128);
    defer cache.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 0), cache.length());
    try std.testing.expect(!cache.isFull());

    // Add some K, V
    const k = [_]f32{1.0} ** 256;
    const v = [_]f32{2.0} ** 256;

    cache.update(&k, &v, 0);
    try std.testing.expectEqual(@as(u32, 1), cache.length());

    cache.update(&k, &v, 1);
    try std.testing.expectEqual(@as(u32, 2), cache.length());

    // Check retrieval
    const cached_k = cache.getKAt(0).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cached_k[0], 0.001);
}

test "full kv cache" {
    const allocator = std.testing.allocator;

    var cache = try KvCache.init(allocator, .{
        .num_layers = 2,
        .num_kv_heads = 4,
        .head_dim = 64,
        .max_seq_len = 128,
    });
    defer cache.deinit();

    try std.testing.expectEqual(@as(u32, 0), cache.sequenceLength());

    // Update layer 0
    const k = [_]f32{1.0} ** 256;
    const v = [_]f32{2.0} ** 256;
    cache.update(0, &k, &v, 0);

    // Stats
    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.num_layers);
}

test "cache config memory calculation" {
    const config = KvCacheConfig{
        .num_layers = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .max_seq_len = 2048,
    };

    // 32 layers * 2048 seq * 8 heads * 128 dim * 2 (K+V) * 4 bytes
    // = 32 * 2048 * 1024 * 2 * 4 = 536,870,912 bytes = 512 MB
    const expected = @as(u64, 32) * 2048 * 8 * 128 * 2 * 4;
    try std.testing.expectEqual(expected, config.memoryBytes());
}
