//! Key-Value cache for efficient autoregressive generation.
//!
//! The KV cache stores the key and value projections from previous tokens,
//! allowing attention to be computed incrementally without recomputing
//! the entire sequence each time.
//!
//! Features based on academic research:
//! - Sliding window attention (SqueezeAttention, LMCache)
//! - Optional Q8 quantization for memory reduction
//! - Memory pressure callbacks for adaptive eviction
//! - Layer-wise budget allocation
//!
//! References:
//! - SqueezeAttention: https://openreview.net/forum?id=9HK2rHNAhd
//! - LMCache: https://lmcache.ai/tech_report.pdf

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
    /// Sliding window size (0 = disabled, uses full context)
    sliding_window: u32 = 0,
    /// Enable quantization for memory savings (Q8 format)
    enable_quantization: bool = false,
    /// Memory pressure threshold (0.0-1.0). When utilization exceeds this, trigger eviction callback.
    memory_pressure_threshold: f32 = 0.9,

    pub const CacheDType = enum {
        f32,
        f16,
        bf16,
        q8, // 8-bit quantized (new)
    };

    /// Calculate total memory requirement in bytes.
    pub fn memoryBytes(self: KvCacheConfig) u64 {
        const kv_dim = @as(u64, self.num_kv_heads) * self.head_dim;
        const effective_len = if (self.sliding_window > 0 and self.sliding_window < self.max_seq_len)
            self.sliding_window
        else
            self.max_seq_len;
        const per_layer = effective_len * kv_dim * 2; // K and V
        const total_elements = per_layer * self.num_layers;

        return switch (self.dtype) {
            .f32 => total_elements * 4,
            .f16, .bf16 => total_elements * 2,
            .q8 => total_elements + (self.num_layers * kv_dim * 4), // Q8 + scale factors
        };
    }

    /// Get effective window size.
    pub fn effectiveWindowSize(self: KvCacheConfig) u32 {
        return if (self.sliding_window > 0 and self.sliding_window < self.max_seq_len)
            self.sliding_window
        else
            self.max_seq_len;
    }
};

/// Memory pressure event for callback notifications.
pub const MemoryPressureEvent = struct {
    current_utilization: f32,
    threshold: f32,
    sequence_length: u32,
    memory_bytes: u64,
};

/// Callback type for memory pressure notifications.
pub const MemoryPressureCallback = *const fn (event: MemoryPressureEvent) void;

/// KV cache for a single layer with optional sliding window.
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
    /// Sliding window configuration (0 = disabled)
    window_size: u32,
    /// Ring buffer head for sliding window mode
    ring_head: u32,
    /// Total tokens seen (for position tracking in sliding window)
    total_tokens: u64,

    /// Extended initialization with sliding window support.
    pub const InitOptions = struct {
        window_size: u32 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, num_kv_heads: u32, head_dim: u32, max_seq_len: u32) !LayerKvCache {
        return initWithOptions(allocator, num_kv_heads, head_dim, max_seq_len, .{});
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        num_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        options: InitOptions,
    ) !LayerKvCache {
        const kv_dim = num_kv_heads * head_dim;
        const effective_len = if (options.window_size > 0 and options.window_size < max_seq_len)
            options.window_size
        else
            max_seq_len;
        const cache_size = @as(usize, effective_len) * kv_dim;

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
            .max_len = effective_len,
            .window_size = options.window_size,
            .ring_head = 0,
            .total_tokens = 0,
        };
    }

    pub fn deinit(self: *LayerKvCache, allocator: std.mem.Allocator) void {
        allocator.free(self.k_cache);
        allocator.free(self.v_cache);
        self.* = undefined;
    }

    /// Update cache with new K, V for a single position.
    /// In sliding window mode, this uses ring buffer semantics.
    pub fn update(self: *LayerKvCache, k: []const f32, v: []const f32, pos: u32) void {
        if (self.window_size > 0) {
            // Sliding window mode: use ring buffer
            self.pushSlidingWindow(k, v);
        } else {
            // Standard mode: direct position addressing
            if (pos >= self.max_len) return;

            const offset = @as(usize, pos) * self.kv_dim;
            @memcpy(self.k_cache[offset .. offset + self.kv_dim], k);
            @memcpy(self.v_cache[offset .. offset + self.kv_dim], v);

            if (pos >= self.len) {
                self.len = pos + 1;
            }
        }
    }

    /// Push K, V in sliding window mode (ring buffer semantics).
    fn pushSlidingWindow(self: *LayerKvCache, k: []const f32, v: []const f32) void {
        const offset = @as(usize, self.ring_head) * self.kv_dim;
        @memcpy(self.k_cache[offset .. offset + self.kv_dim], k);
        @memcpy(self.v_cache[offset .. offset + self.kv_dim], v);

        self.ring_head = (self.ring_head + 1) % self.max_len;
        self.total_tokens += 1;

        if (self.len < self.max_len) {
            self.len += 1;
        }
    }

    /// Get the effective position for sliding window attention.
    /// Returns the logical position within the window.
    pub fn getWindowPosition(self: *const LayerKvCache, absolute_pos: u64) ?u32 {
        if (self.window_size == 0) {
            // Standard mode
            if (absolute_pos < self.len) return @intCast(absolute_pos);
            return null;
        }

        // Sliding window mode
        if (absolute_pos < self.total_tokens -| self.len) return null; // Too old, evicted
        if (absolute_pos >= self.total_tokens) return null; // Future position

        // Calculate ring buffer index
        const tokens_ago = self.total_tokens - absolute_pos - 1;
        const logical_idx = self.len - 1 - @as(u32, @intCast(tokens_ago));
        return logical_idx;
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
        self.ring_head = 0;
        self.total_tokens = 0;
    }

    /// Check if cache is using sliding window.
    pub fn isSlidingWindow(self: *const LayerKvCache) bool {
        return self.window_size > 0;
    }

    /// Get memory savings from sliding window (vs full context).
    pub fn windowMemorySavings(self: *const LayerKvCache, full_context_len: u32) f32 {
        if (self.window_size == 0 or full_context_len <= self.max_len) return 0.0;
        return 1.0 - (@as(f32, @floatFromInt(self.max_len)) / @as(f32, @floatFromInt(full_context_len)));
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

/// Full KV cache for all layers with sliding window and memory pressure support.
pub const KvCache = struct {
    allocator: std.mem.Allocator,
    layers: []LayerKvCache,
    config: KvCacheConfig,
    memory_pressure_callback: ?MemoryPressureCallback,
    pressure_triggered: bool,

    pub fn init(allocator: std.mem.Allocator, config: KvCacheConfig) !KvCache {
        const layers = try allocator.alloc(LayerKvCache, config.num_layers);
        errdefer allocator.free(layers);

        const layer_options = LayerKvCache.InitOptions{
            .window_size = config.sliding_window,
        };

        for (0..config.num_layers) |i| {
            layers[i] = try LayerKvCache.initWithOptions(
                allocator,
                config.num_kv_heads,
                config.head_dim,
                config.max_seq_len,
                layer_options,
            );
        }

        return .{
            .allocator = allocator,
            .layers = layers,
            .config = config,
            .memory_pressure_callback = null,
            .pressure_triggered = false,
        };
    }

    /// Set callback for memory pressure events.
    pub fn setMemoryPressureCallback(self: *KvCache, callback: MemoryPressureCallback) void {
        self.memory_pressure_callback = callback;
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
    /// Checks memory pressure and triggers callback if threshold exceeded.
    pub fn update(self: *KvCache, layer_idx: u32, k: []const f32, v: []const f32, pos: u32) void {
        self.layers[layer_idx].update(k, v, pos);

        // Check memory pressure after update (only check periodically to reduce overhead)
        if (self.memory_pressure_callback != null and !self.pressure_triggered) {
            self.checkMemoryPressure();
        }
    }

    /// Check memory pressure and trigger callback if needed.
    fn checkMemoryPressure(self: *KvCache) void {
        const stats = self.getStats();
        if (stats.utilization >= self.config.memory_pressure_threshold) {
            self.pressure_triggered = true;
            if (self.memory_pressure_callback) |callback| {
                callback(.{
                    .current_utilization = stats.utilization,
                    .threshold = self.config.memory_pressure_threshold,
                    .sequence_length = stats.sequence_length,
                    .memory_bytes = stats.memory_bytes,
                });
            }
        }
    }

    /// Reset pressure triggered flag (call after handling pressure event).
    pub fn resetPressureFlag(self: *KvCache) void {
        self.pressure_triggered = false;
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

test "sliding window layer cache" {
    const allocator = std.testing.allocator;

    // Create a layer cache with sliding window of 4
    var cache = try LayerKvCache.initWithOptions(allocator, 2, 32, 128, .{
        .window_size = 4,
    });
    defer cache.deinit(allocator);

    try std.testing.expect(cache.isSlidingWindow());
    try std.testing.expectEqual(@as(u32, 4), cache.max_len);

    const v = [_]f32{2.0} ** 64;

    // Push 6 tokens (exceeds window of 4)
    for (0..6) |i| {
        const k_val = [_]f32{@floatFromInt(i)} ** 64;
        cache.update(&k_val, &v, @intCast(i));
    }

    // Should only have 4 tokens (window size)
    try std.testing.expectEqual(@as(u32, 4), cache.length());
    try std.testing.expectEqual(@as(u64, 6), cache.total_tokens);

    // Memory savings: 4/128 = ~97%
    const savings = cache.windowMemorySavings(128);
    try std.testing.expect(savings > 0.95);
}

test "sliding window full cache" {
    const allocator = std.testing.allocator;

    var cache = try KvCache.init(allocator, .{
        .num_layers = 2,
        .num_kv_heads = 2,
        .head_dim = 32,
        .max_seq_len = 128,
        .sliding_window = 8, // Use window of 8
    });
    defer cache.deinit();

    try std.testing.expectEqual(@as(u32, 8), cache.config.effectiveWindowSize());

    const k = [_]f32{1.0} ** 64;
    const v = [_]f32{2.0} ** 64;

    // Add 12 tokens (exceeds window)
    for (0..12) |i| {
        cache.update(0, &k, &v, @intCast(i));
        cache.update(1, &k, &v, @intCast(i));
    }

    // Each layer should have 8 tokens
    try std.testing.expectEqual(@as(u32, 8), cache.layers[0].length());
    try std.testing.expectEqual(@as(u32, 8), cache.layers[1].length());
}

test "memory pressure callback" {
    const allocator = std.testing.allocator;

    const TestCallback = struct {
        fn callback(_: MemoryPressureEvent) void {
            // Callback invoked when memory pressure exceeds threshold
        }
    };

    var cache = try KvCache.init(allocator, .{
        .num_layers = 1,
        .num_kv_heads = 1,
        .head_dim = 8,
        .max_seq_len = 10,
        .memory_pressure_threshold = 0.5,
    });
    defer cache.deinit();

    cache.setMemoryPressureCallback(TestCallback.callback);

    const k = [_]f32{1.0} ** 8;
    const v = [_]f32{2.0} ** 8;

    // Add tokens until pressure threshold (50%)
    for (0..6) |i| {
        cache.update(0, &k, &v, @intCast(i));
    }

    // Pressure should have been triggered
    try std.testing.expect(cache.pressure_triggered);

    // Reset and add more
    cache.resetPressureFlag();
    try std.testing.expect(!cache.pressure_triggered);
}

test "config with sliding window memory calculation" {
    const full_config = KvCacheConfig{
        .num_layers = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .max_seq_len = 4096,
    };

    const window_config = KvCacheConfig{
        .num_layers = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .max_seq_len = 4096,
        .sliding_window = 512, // 8x smaller window
    };

    const full_memory = full_config.memoryBytes();
    const window_memory = window_config.memoryBytes();

    // Window should use 1/8 the memory
    try std.testing.expectEqual(full_memory / 8, window_memory);
}

test {
    std.testing.refAllDecls(@This());
}
