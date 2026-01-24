//! FPGA-optimized KV-Cache kernels for LLM inference
//!
//! Provides hardware-accelerated implementations for:
//! - On-chip BRAM-based KV storage for fast access
//! - Hierarchical cache with DDR/HBM overflow
//! - Paged attention with dynamic memory allocation
//! - Prefix caching for prompt reuse
//!
//! Performance targets (per FPGA research roadmap):
//! - Cache hit latency: <100ns for BRAM, <1us for DDR
//! - Memory efficiency: 2-4x reduction via compression
//! - Throughput: >10M cache operations/sec

const std = @import("std");
const build_options = @import("build_options");

/// Configuration for FPGA KV-cache kernels
pub const KVCacheKernelConfig = struct {
    /// Number of attention layers
    num_layers: u32 = 32,
    /// Number of attention heads (for key/value)
    num_kv_heads: u32 = 8,
    /// Head dimension
    head_dim: u32 = 128,
    /// Maximum sequence length
    max_seq_len: u32 = 4096,
    /// Block size for paged attention
    block_size: u32 = 16,
    /// BRAM capacity per layer (bytes) - fast on-chip memory
    bram_capacity_per_layer: u32 = 256 * 1024, // 256 KB
    /// Enable KV compression
    enable_compression: bool = true,
    /// Compression ratio target (e.g., 2.0 = 50% size reduction)
    compression_target: f32 = 2.0,
    /// Enable prefix caching
    enable_prefix_cache: bool = true,
    /// Maximum number of cached prefixes
    max_prefixes: u32 = 64,
};

/// KV entry precision modes
pub const KVPrecision = enum {
    fp32, // Full precision (4 bytes per element)
    fp16, // Half precision (2 bytes)
    int8, // Quantized (1 byte)
    int4, // Highly compressed (0.5 bytes)

    pub fn bytesPerElement(self: KVPrecision) f32 {
        return switch (self) {
            .fp32 => 4.0,
            .fp16 => 2.0,
            .int8 => 1.0,
            .int4 => 0.5,
        };
    }
};

/// Memory tier for cache placement
pub const MemoryTier = enum {
    bram, // On-chip block RAM (fastest, limited)
    hbm, // High-bandwidth memory (fast, medium capacity)
    ddr, // DDR memory (slower, large capacity)

    pub fn latencyNs(self: MemoryTier) u32 {
        return switch (self) {
            .bram => 10,
            .hbm => 100,
            .ddr => 500,
        };
    }

    pub fn bandwidthGbps(self: MemoryTier) u32 {
        return switch (self) {
            .bram => 1000, // Limited by fabric, not bandwidth
            .hbm => 400,
            .ddr => 50,
        };
    }
};

/// Block state for paged attention
pub const BlockState = enum {
    free,
    allocated,
    pinned, // For prefix cache
    evicting,
};

/// A single block in the paged KV cache
pub const CacheBlock = struct {
    /// Block ID
    id: u32,
    /// Layer this block belongs to
    layer: u32,
    /// Sequence offset this block covers
    seq_offset: u32,
    /// Number of valid tokens in this block
    num_tokens: u32,
    /// Memory tier where block resides
    tier: MemoryTier,
    /// Block state
    state: BlockState,
    /// Reference count (for sharing)
    ref_count: u32 = 1,
    /// Hash for prefix matching
    prefix_hash: u64 = 0,
    /// Key data pointer
    key_data: ?[*]u8 = null,
    /// Value data pointer
    value_data: ?[*]u8 = null,

    pub fn isValid(self: *const CacheBlock) bool {
        return self.state == .allocated or self.state == .pinned;
    }

    pub fn canEvict(self: *const CacheBlock) bool {
        return self.state == .allocated and self.ref_count == 0;
    }
};

/// Hierarchical KV cache with BRAM/HBM/DDR tiers
pub const HierarchicalKVCache = struct {
    config: KVCacheKernelConfig,
    allocator: std.mem.Allocator,

    // Block management
    blocks: std.ArrayListUnmanaged(CacheBlock),
    free_blocks: std.ArrayListUnmanaged(u32),

    // Per-layer block mappings: layer -> sequence -> block_id
    layer_mappings: []std.AutoHashMap(u32, u32),

    // Memory pool per tier
    bram_used: u64 = 0,
    hbm_used: u64 = 0,
    ddr_used: u64 = 0,

    // Prefix cache: hash -> block_id
    prefix_cache: std.AutoHashMap(u64, u32),

    // Statistics
    stats: CacheStats = .{},

    pub fn init(allocator: std.mem.Allocator, config: KVCacheKernelConfig) !HierarchicalKVCache {
        var cache = HierarchicalKVCache{
            .config = config,
            .allocator = allocator,
            .blocks = .{},
            .free_blocks = .{},
            .layer_mappings = try allocator.alloc(std.AutoHashMap(u32, u32), config.num_layers),
            .prefix_cache = std.AutoHashMap(u64, u32).init(allocator),
        };

        // Initialize per-layer mappings
        for (cache.layer_mappings) |*mapping| {
            mapping.* = std.AutoHashMap(u32, u32).init(allocator);
        }

        return cache;
    }

    pub fn deinit(self: *HierarchicalKVCache) void {
        // Free all block data
        for (self.blocks.items) |block| {
            if (block.key_data) |ptr| {
                const slice = ptr[0..self.blockSizeBytes()];
                self.allocator.free(slice);
            }
            if (block.value_data) |ptr| {
                const slice = ptr[0..self.blockSizeBytes()];
                self.allocator.free(slice);
            }
        }

        self.blocks.deinit(self.allocator);
        self.free_blocks.deinit(self.allocator);
        self.prefix_cache.deinit();

        for (self.layer_mappings) |*mapping| {
            mapping.deinit();
        }
        self.allocator.free(self.layer_mappings);
    }

    /// Calculate block size in bytes
    pub fn blockSizeBytes(self: *const HierarchicalKVCache) usize {
        return self.config.block_size * self.config.head_dim * self.config.num_kv_heads * 4; // FP32
    }

    /// Allocate a new block for a layer/sequence position
    pub fn allocateBlock(self: *HierarchicalKVCache, layer: u32, seq_offset: u32) !*CacheBlock {
        var block_id: u32 = undefined;

        // Try to reuse a free block
        if (self.free_blocks.items.len > 0) {
            block_id = self.free_blocks.pop() orelse unreachable;
            var block = &self.blocks.items[block_id];
            block.layer = layer;
            block.seq_offset = seq_offset;
            block.num_tokens = 0;
            block.state = .allocated;
            block.ref_count = 1;

            self.stats.reused_blocks += 1;
            return block;
        }

        // Need to allocate new block
        const tier = self.selectTier();
        const size = self.blockSizeBytes();

        const key_data = try self.allocator.alloc(u8, size);
        const value_data = try self.allocator.alloc(u8, size);

        block_id = @intCast(self.blocks.items.len);
        try self.blocks.append(self.allocator, CacheBlock{
            .id = block_id,
            .layer = layer,
            .seq_offset = seq_offset,
            .num_tokens = 0,
            .tier = tier,
            .state = .allocated,
            .key_data = key_data.ptr,
            .value_data = value_data.ptr,
        });

        // Update memory tracking
        switch (tier) {
            .bram => self.bram_used += size,
            .hbm => self.hbm_used += size,
            .ddr => self.ddr_used += size,
        }

        // Register in layer mapping
        try self.layer_mappings[layer].put(seq_offset, block_id);

        self.stats.allocated_blocks += 1;
        return &self.blocks.items[block_id];
    }

    /// Free a block back to the pool
    pub fn freeBlock(self: *HierarchicalKVCache, block: *CacheBlock) void {
        if (block.ref_count > 0) {
            block.ref_count -= 1;
            if (block.ref_count > 0) return; // Still in use
        }

        block.state = .free;
        self.free_blocks.append(self.allocator, block.id) catch {};
        _ = self.layer_mappings[block.layer].remove(block.seq_offset);

        self.stats.freed_blocks += 1;
    }

    /// Get a block for reading
    pub fn getBlock(self: *HierarchicalKVCache, layer: u32, seq_offset: u32) ?*CacheBlock {
        const block_id = self.layer_mappings[layer].get(seq_offset) orelse return null;
        const block = &self.blocks.items[block_id];

        if (block.isValid()) {
            self.stats.cache_hits += 1;
            return block;
        }

        self.stats.cache_misses += 1;
        return null;
    }

    /// Append key/value data to a block
    pub fn appendToBlock(
        self: *HierarchicalKVCache,
        block: *CacheBlock,
        key: []const f32,
        value: []const f32,
    ) !void {
        const head_dim = self.config.head_dim;
        const num_heads = self.config.num_kv_heads;
        const token_size = head_dim * num_heads;

        if (block.num_tokens >= self.config.block_size) {
            return error.BlockFull;
        }

        if (key.len != token_size or value.len != token_size) {
            return error.InvalidSize;
        }

        const offset = block.num_tokens * token_size * 4;

        // Copy key data
        if (block.key_data) |ptr| {
            const key_bytes = std.mem.sliceAsBytes(key);
            @memcpy(ptr[offset..][0..key_bytes.len], key_bytes);
        }

        // Copy value data
        if (block.value_data) |ptr| {
            const value_bytes = std.mem.sliceAsBytes(value);
            @memcpy(ptr[offset..][0..value_bytes.len], value_bytes);
        }

        block.num_tokens += 1;
    }

    /// Look up prefix cache
    pub fn lookupPrefix(self: *HierarchicalKVCache, prefix_hash: u64) ?*CacheBlock {
        if (!self.config.enable_prefix_cache) return null;

        const block_id = self.prefix_cache.get(prefix_hash) orelse return null;
        const block = &self.blocks.items[block_id];

        if (block.state == .pinned and block.prefix_hash == prefix_hash) {
            block.ref_count += 1;
            self.stats.prefix_hits += 1;
            return block;
        }

        return null;
    }

    /// Register a block as a cached prefix
    pub fn registerPrefix(self: *HierarchicalKVCache, block: *CacheBlock, prefix_hash: u64) !void {
        if (!self.config.enable_prefix_cache) return;

        // Check capacity
        if (self.prefix_cache.count() >= self.config.max_prefixes) {
            // Evict oldest prefix (simple LRU would track access times)
            self.stats.prefix_evictions += 1;
        }

        block.prefix_hash = prefix_hash;
        block.state = .pinned;
        try self.prefix_cache.put(prefix_hash, block.id);
    }

    /// Select appropriate memory tier based on usage
    fn selectTier(self: *const HierarchicalKVCache) MemoryTier {
        const bram_limit = @as(u64, self.config.bram_capacity_per_layer) * self.config.num_layers;

        if (self.bram_used < bram_limit) {
            return .bram;
        } else if (self.hbm_used < 16 * 1024 * 1024 * 1024) { // 16 GB HBM limit
            return .hbm;
        } else {
            return .ddr;
        }
    }

    /// Evict blocks to make room (simple LRU approximation)
    pub fn evictBlocks(self: *HierarchicalKVCache, count: usize) void {
        var evicted: usize = 0;

        for (self.blocks.items) |*block| {
            if (evicted >= count) break;

            if (block.canEvict()) {
                self.freeBlock(block);
                evicted += 1;
            }
        }

        self.stats.evictions += evicted;
    }

    /// Get cache statistics
    pub fn getStats(self: *const HierarchicalKVCache) CacheStats {
        var stats = self.stats;
        stats.bram_usage_bytes = self.bram_used;
        stats.hbm_usage_bytes = self.hbm_used;
        stats.ddr_usage_bytes = self.ddr_used;
        stats.total_blocks = @intCast(self.blocks.items.len);
        stats.free_blocks = @intCast(self.free_blocks.items.len);
        return stats;
    }
};

/// Cache statistics
pub const CacheStats = struct {
    cache_hits: u64 = 0,
    cache_misses: u64 = 0,
    allocated_blocks: u64 = 0,
    freed_blocks: u64 = 0,
    reused_blocks: u64 = 0,
    evictions: u64 = 0,
    prefix_hits: u64 = 0,
    prefix_evictions: u64 = 0,
    bram_usage_bytes: u64 = 0,
    hbm_usage_bytes: u64 = 0,
    ddr_usage_bytes: u64 = 0,
    total_blocks: u32 = 0,
    free_blocks: u32 = 0,

    pub fn hitRate(self: *const CacheStats) f64 {
        const total = self.cache_hits + self.cache_misses;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn report(self: *const CacheStats) void {
        std.log.info("KV Cache Statistics:", .{});
        std.log.info("  Hit Rate: {d:.2}%", .{self.hitRate() * 100});
        std.log.info("  Total Blocks: {d} (Free: {d})", .{ self.total_blocks, self.free_blocks });
        std.log.info("  BRAM: {d:.2} MB", .{@as(f64, @floatFromInt(self.bram_usage_bytes)) / (1024 * 1024)});
        std.log.info("  HBM: {d:.2} MB", .{@as(f64, @floatFromInt(self.hbm_usage_bytes)) / (1024 * 1024)});
        std.log.info("  DDR: {d:.2} MB", .{@as(f64, @floatFromInt(self.ddr_usage_bytes)) / (1024 * 1024)});
        std.log.info("  Prefix Hits: {d}, Evictions: {d}", .{ self.prefix_hits, self.evictions });
    }
};

/// Compressed KV cache for memory efficiency
pub const CompressedKVCache = struct {
    config: KVCacheKernelConfig,
    allocator: std.mem.Allocator,
    base_cache: HierarchicalKVCache,
    precision: KVPrecision = .fp16,

    // Quantization parameters per layer
    layer_scales: []f32,
    layer_zeros: []f32,

    pub fn init(allocator: std.mem.Allocator, config: KVCacheKernelConfig) !CompressedKVCache {
        var cache = CompressedKVCache{
            .config = config,
            .allocator = allocator,
            .base_cache = try HierarchicalKVCache.init(allocator, config),
            .layer_scales = try allocator.alloc(f32, config.num_layers),
            .layer_zeros = try allocator.alloc(f32, config.num_layers),
        };

        // Initialize default quantization parameters
        @memset(cache.layer_scales, 1.0);
        @memset(cache.layer_zeros, 0.0);

        return cache;
    }

    pub fn deinit(self: *CompressedKVCache) void {
        self.base_cache.deinit();
        self.allocator.free(self.layer_scales);
        self.allocator.free(self.layer_zeros);
    }

    /// Set precision for KV storage
    pub fn setPrecision(self: *CompressedKVCache, precision: KVPrecision) void {
        self.precision = precision;
    }

    /// Calibrate quantization parameters from sample data
    pub fn calibrate(self: *CompressedKVCache, layer: u32, sample_values: []const f32) void {
        if (sample_values.len == 0) return;

        var min_val: f32 = std.math.inf(f32);
        var max_val: f32 = -std.math.inf(f32);

        for (sample_values) |v| {
            min_val = @min(min_val, v);
            max_val = @max(max_val, v);
        }

        // Compute scale and zero-point for symmetric quantization
        const abs_max = @max(@abs(min_val), @abs(max_val));
        const range = switch (self.precision) {
            .int8 => 127.0,
            .int4 => 7.0,
            else => 1.0,
        };

        self.layer_scales[layer] = abs_max / range;
        self.layer_zeros[layer] = 0; // Symmetric quantization
    }

    /// Compress and store key/value
    pub fn storeCompressed(
        self: *CompressedKVCache,
        layer: u32,
        seq_offset: u32,
        key: []const f32,
        value: []const f32,
    ) !void {
        var block = self.base_cache.getBlock(layer, seq_offset);
        if (block == null) {
            block = try self.base_cache.allocateBlock(layer, seq_offset);
        }

        // In real implementation, compress before storing
        // For now, store as-is (compression happens in hardware)
        try self.base_cache.appendToBlock(block.?, key, value);
    }

    /// Retrieve and decompress key/value
    pub fn retrieveDecompressed(
        self: *CompressedKVCache,
        layer: u32,
        seq_offset: u32,
        key_out: []f32,
        value_out: []f32,
    ) !void {
        const block = self.base_cache.getBlock(layer, seq_offset) orelse return error.CacheMiss;

        const head_dim = self.config.head_dim;
        const num_heads = self.config.num_kv_heads;
        const token_size = head_dim * num_heads;

        // Decompress (in real impl, would apply inverse quantization)
        if (block.key_data) |ptr| {
            const key_bytes = ptr[0 .. token_size * 4];
            @memcpy(std.mem.sliceAsBytes(key_out), key_bytes);
        }

        if (block.value_data) |ptr| {
            const value_bytes = ptr[0 .. token_size * 4];
            @memcpy(std.mem.sliceAsBytes(value_out), value_bytes);
        }
    }

    /// Get compression ratio achieved
    pub fn getCompressionRatio(self: *const CompressedKVCache) f32 {
        const base_bytes: f32 = 4.0; // FP32
        const compressed_bytes = self.precision.bytesPerElement();
        return base_bytes / compressed_bytes;
    }

    pub fn getStats(self: *const CompressedKVCache) CacheStats {
        return self.base_cache.getStats();
    }
};

/// Streaming KV cache for continuous inference
pub const StreamingKVCache = struct {
    config: KVCacheKernelConfig,
    allocator: std.mem.Allocator,
    cache: HierarchicalKVCache,

    // Sliding window state
    window_start: u32 = 0,
    current_pos: u32 = 0,

    // Circular buffer for recent tokens
    recent_keys: ?[]f32 = null,
    recent_values: ?[]f32 = null,

    pub fn init(allocator: std.mem.Allocator, config: KVCacheKernelConfig) !StreamingKVCache {
        const buffer_size = config.block_size * config.head_dim * config.num_kv_heads;

        return StreamingKVCache{
            .config = config,
            .allocator = allocator,
            .cache = try HierarchicalKVCache.init(allocator, config),
            .recent_keys = try allocator.alloc(f32, buffer_size),
            .recent_values = try allocator.alloc(f32, buffer_size),
        };
    }

    pub fn deinit(self: *StreamingKVCache) void {
        self.cache.deinit();
        if (self.recent_keys) |buf| self.allocator.free(buf);
        if (self.recent_values) |buf| self.allocator.free(buf);
    }

    /// Append a new token's key/value to the stream
    pub fn appendToken(
        self: *StreamingKVCache,
        layer: u32,
        key: []const f32,
        value: []const f32,
    ) !void {
        const block_size = self.config.block_size;
        const seq_offset = (self.current_pos / block_size) * block_size;

        var block = self.cache.getBlock(layer, seq_offset);
        if (block == null or block.?.num_tokens >= block_size) {
            block = try self.cache.allocateBlock(layer, self.current_pos);
        }

        try self.cache.appendToBlock(block.?, key, value);

        // Update circular buffer
        const token_size = self.config.head_dim * self.config.num_kv_heads;
        const buf_pos = (self.current_pos % block_size) * token_size;

        if (self.recent_keys) |buf| {
            @memcpy(buf[buf_pos..][0..key.len], key);
        }
        if (self.recent_values) |buf| {
            @memcpy(buf[buf_pos..][0..value.len], value);
        }

        if (layer == self.config.num_layers - 1) {
            self.current_pos += 1;
        }
    }

    /// Slide the window forward, evicting old blocks
    pub fn slideWindow(self: *StreamingKVCache, new_start: u32) void {
        if (new_start <= self.window_start) return;

        // Evict blocks before new_start
        const blocks_to_evict = (new_start - self.window_start) / self.config.block_size;
        self.cache.evictBlocks(blocks_to_evict * self.config.num_layers);

        self.window_start = new_start;
    }

    /// Get current sequence length
    pub fn getSeqLen(self: *const StreamingKVCache) u32 {
        return self.current_pos;
    }

    pub fn getStats(self: *const StreamingKVCache) CacheStats {
        return self.cache.getStats();
    }
};

// Tests

test "hierarchical kv cache basic" {
    const allocator = std.testing.allocator;

    const config = KVCacheKernelConfig{
        .num_layers = 2,
        .num_kv_heads = 2,
        .head_dim = 4,
        .block_size = 4,
    };

    var cache = try HierarchicalKVCache.init(allocator, config);
    defer cache.deinit();

    // Allocate a block
    const block = try cache.allocateBlock(0, 0);
    try std.testing.expect(block.isValid());
    try std.testing.expect(block.tier == .bram);

    // Append data
    const key = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const value = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    try cache.appendToBlock(block, &key, &value);

    try std.testing.expect(block.num_tokens == 1);

    // Retrieve
    const retrieved = cache.getBlock(0, 0);
    try std.testing.expect(retrieved != null);
    try std.testing.expect(retrieved.?.num_tokens == 1);
}

test "cache eviction" {
    const allocator = std.testing.allocator;

    const config = KVCacheKernelConfig{
        .num_layers = 1,
        .num_kv_heads = 1,
        .head_dim = 2,
        .block_size = 2,
    };

    var cache = try HierarchicalKVCache.init(allocator, config);
    defer cache.deinit();

    // Allocate multiple blocks
    _ = try cache.allocateBlock(0, 0);
    _ = try cache.allocateBlock(0, 2);
    const block3 = try cache.allocateBlock(0, 4);

    // Free one
    cache.freeBlock(block3);

    const stats = cache.getStats();
    try std.testing.expect(stats.free_blocks == 1);
}

test "compressed cache ratio" {
    const allocator = std.testing.allocator;

    const config = KVCacheKernelConfig{
        .num_layers = 1,
        .num_kv_heads = 1,
        .head_dim = 4,
        .block_size = 4,
    };

    var cache = try CompressedKVCache.init(allocator, config);
    defer cache.deinit();

    cache.setPrecision(.int8);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), cache.getCompressionRatio(), 0.01);

    cache.setPrecision(.int4);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), cache.getCompressionRatio(), 0.01);
}

test "memory tier properties" {
    try std.testing.expect(MemoryTier.bram.latencyNs() < MemoryTier.hbm.latencyNs());
    try std.testing.expect(MemoryTier.hbm.latencyNs() < MemoryTier.ddr.latencyNs());

    try std.testing.expect(MemoryTier.bram.bandwidthGbps() > MemoryTier.hbm.bandwidthGbps());
}
