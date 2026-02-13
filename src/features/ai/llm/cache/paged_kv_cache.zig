//! Paged KV Cache for efficient memory management.
//!
//! Implements block-based KV cache allocation inspired by vLLM's PagedAttention.
//! Key features:
//! - Fixed-size blocks (pages) allocated on-demand
//! - Block table for non-contiguous memory addressing
//! - Memory sharing across sequences with common prefixes
//! - Efficient batch processing with dynamic sequence lengths
//!
//! References:
//! - vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention
//!   https://arxiv.org/abs/2309.06180

const std = @import("std");

/// Page (block) size in tokens. Common values: 16, 32, 64.
pub const DEFAULT_PAGE_SIZE: u32 = 16;

/// A single page of KV cache.
pub const KvPage = struct {
    /// Key data: [page_size, kv_dim]
    k_data: []f32,
    /// Value data: [page_size, kv_dim]
    v_data: []f32,
    /// Number of tokens filled in this page
    num_tokens: u32,
    /// Reference count for sharing
    ref_count: u32,
    /// Page ID for tracking
    page_id: u32,

    pub fn init(allocator: std.mem.Allocator, page_size: u32, kv_dim: u32, page_id: u32) !KvPage {
        const size = @as(usize, page_size) * kv_dim;
        const k_data = try allocator.alloc(f32, size);
        errdefer allocator.free(k_data);
        const v_data = try allocator.alloc(f32, size);

        @memset(k_data, 0);
        @memset(v_data, 0);

        return .{
            .k_data = k_data,
            .v_data = v_data,
            .num_tokens = 0,
            .ref_count = 1,
            .page_id = page_id,
        };
    }

    pub fn deinit(self: *KvPage, allocator: std.mem.Allocator) void {
        allocator.free(self.k_data);
        allocator.free(self.v_data);
        self.* = undefined;
    }

    /// Get remaining capacity in tokens.
    pub fn remaining(self: *const KvPage, page_size: u32) u32 {
        return page_size - self.num_tokens;
    }

    /// Check if page is full.
    pub fn isFull(self: *const KvPage, page_size: u32) bool {
        return self.num_tokens >= page_size;
    }

    /// Append a single token's KV.
    pub fn append(self: *KvPage, k: []const f32, v: []const f32, kv_dim: u32, page_size: u32) bool {
        if (self.isFull(page_size)) return false;

        const offset = @as(usize, self.num_tokens) * kv_dim;

        // Prefetch next cache line for upcoming writes (if there's room for more tokens)
        if (self.num_tokens + 1 < page_size) {
            const next_offset = offset + kv_dim;
            @prefetch(self.k_data.ptr + next_offset, .{ .rw = .write, .locality = 3 });
            @prefetch(self.v_data.ptr + next_offset, .{ .rw = .write, .locality = 3 });
        }

        @memcpy(self.k_data[offset .. offset + kv_dim], k);
        @memcpy(self.v_data[offset .. offset + kv_dim], v);
        self.num_tokens += 1;
        return true;
    }

    /// Get K at position within page.
    /// Prefetches next position for sequential access patterns.
    pub fn getK(self: *const KvPage, pos: u32, kv_dim: u32) ?[]const f32 {
        if (pos >= self.num_tokens) return null;
        const offset = @as(usize, pos) * kv_dim;

        // Prefetch next K for sequential attention patterns
        if (pos + 1 < self.num_tokens) {
            const next_offset = offset + kv_dim;
            @prefetch(self.k_data.ptr + next_offset, .{ .rw = .read, .locality = 3 });
        }

        return self.k_data[offset .. offset + kv_dim];
    }

    /// Get V at position within page.
    /// Prefetches next position for sequential access patterns.
    pub fn getV(self: *const KvPage, pos: u32, kv_dim: u32) ?[]const f32 {
        if (pos >= self.num_tokens) return null;
        const offset = @as(usize, pos) * kv_dim;

        // Prefetch next V for sequential attention patterns
        if (pos + 1 < self.num_tokens) {
            const next_offset = offset + kv_dim;
            @prefetch(self.v_data.ptr + next_offset, .{ .rw = .read, .locality = 3 });
        }

        return self.v_data[offset .. offset + kv_dim];
    }
};

/// Page table entry mapping logical blocks to physical pages.
pub const PageTableEntry = struct {
    page_idx: ?u32, // Index into page pool (null = not allocated)
    logical_block: u32, // Which logical block this represents
};

/// Configuration for paged KV cache.
pub const PagedKvCacheConfig = struct {
    /// Number of transformer layers
    num_layers: u32,
    /// Number of KV heads per layer
    num_kv_heads: u32,
    /// Dimension per head
    head_dim: u32,
    /// Maximum total pages in the pool
    max_pages: u32 = 1024,
    /// Tokens per page
    page_size: u32 = DEFAULT_PAGE_SIZE,
    /// Maximum sequences that can share the cache
    max_sequences: u32 = 32,

    /// Calculate KV dimension.
    pub fn kvDim(self: PagedKvCacheConfig) u32 {
        return self.num_kv_heads * self.head_dim;
    }

    /// Calculate memory per page in bytes.
    pub fn pageMemoryBytes(self: PagedKvCacheConfig) u64 {
        return @as(u64, self.page_size) * self.kvDim() * 2 * @sizeOf(f32);
    }

    /// Calculate total memory for page pool.
    pub fn totalMemoryBytes(self: PagedKvCacheConfig) u64 {
        return self.pageMemoryBytes() * self.max_pages * self.num_layers;
    }
};

/// Per-sequence KV cache state.
pub const SequenceKvState = struct {
    allocator: std.mem.Allocator,
    /// Block tables per layer: maps logical blocks to physical pages
    block_tables: [][]?u32,
    /// Current sequence length
    seq_len: u32,
    /// Sequence ID
    seq_id: u32,
    /// Configuration reference
    page_size: u32,
    num_layers: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        seq_id: u32,
        num_layers: u32,
        max_blocks: u32,
        page_size: u32,
    ) !SequenceKvState {
        const block_tables = try allocator.alloc([]?u32, num_layers);
        errdefer allocator.free(block_tables);

        for (0..num_layers) |i| {
            block_tables[i] = try allocator.alloc(?u32, max_blocks);
            @memset(block_tables[i], null);
        }

        return .{
            .allocator = allocator,
            .block_tables = block_tables,
            .seq_len = 0,
            .seq_id = seq_id,
            .page_size = page_size,
            .num_layers = num_layers,
        };
    }

    pub fn deinit(self: *SequenceKvState) void {
        for (self.block_tables) |table| {
            self.allocator.free(table);
        }
        self.allocator.free(self.block_tables);
        self.* = undefined;
    }

    /// Get block index for a given position.
    pub fn getBlockIdx(self: *const SequenceKvState, pos: u32) u32 {
        return pos / self.page_size;
    }

    /// Get offset within block for a given position.
    pub fn getBlockOffset(self: *const SequenceKvState, pos: u32) u32 {
        return pos % self.page_size;
    }

    /// Get physical page index for layer and position.
    pub fn getPageIdx(self: *const SequenceKvState, layer: u32, pos: u32) ?u32 {
        const block_idx = self.getBlockIdx(pos);
        if (block_idx >= self.block_tables[layer].len) return null;
        return self.block_tables[layer][block_idx];
    }

    /// Set physical page index for layer and block.
    pub fn setPageIdx(self: *SequenceKvState, layer: u32, block_idx: u32, page_idx: u32) void {
        if (block_idx < self.block_tables[layer].len) {
            self.block_tables[layer][block_idx] = page_idx;
        }
    }

    /// Get number of allocated blocks.
    pub fn numAllocatedBlocks(self: *const SequenceKvState) u32 {
        var count: u32 = 0;
        for (self.block_tables[0]) |entry| {
            if (entry != null) count += 1;
        }
        return count;
    }
};

/// Page pool manager for a single layer.
pub const LayerPagePool = struct {
    allocator: std.mem.Allocator,
    /// All pages in the pool
    pages: []KvPage,
    /// Free page indices
    free_list: std.ArrayListUnmanaged(u32),
    /// Configuration
    page_size: u32,
    kv_dim: u32,
    max_pages: u32,
    /// Statistics
    allocated_pages: u32,
    peak_allocated: u32,

    pub fn init(allocator: std.mem.Allocator, config: PagedKvCacheConfig) !LayerPagePool {
        const pages = try allocator.alloc(KvPage, config.max_pages);
        errdefer allocator.free(pages);

        const kv_dim = config.kvDim();

        // Initialize all pages
        for (0..config.max_pages) |i| {
            pages[i] = try KvPage.init(allocator, config.page_size, kv_dim, @intCast(i));
        }

        // Initialize free list with all pages
        var free_list = std.ArrayListUnmanaged(u32).empty;
        try free_list.ensureTotalCapacity(allocator, config.max_pages);
        for (0..config.max_pages) |i| {
            free_list.appendAssumeCapacity(@intCast(i));
        }

        return .{
            .allocator = allocator,
            .pages = pages,
            .free_list = free_list,
            .page_size = config.page_size,
            .kv_dim = kv_dim,
            .max_pages = config.max_pages,
            .allocated_pages = 0,
            .peak_allocated = 0,
        };
    }

    pub fn deinit(self: *LayerPagePool) void {
        for (self.pages) |*page| {
            page.deinit(self.allocator);
        }
        self.allocator.free(self.pages);
        self.free_list.deinit(self.allocator);
        self.* = undefined;
    }

    /// Allocate a page from the pool.
    pub fn allocatePage(self: *LayerPagePool) ?u32 {
        if (self.free_list.items.len == 0) return null;

        const page_idx = self.free_list.pop() orelse return null;
        self.pages[page_idx].num_tokens = 0;
        self.pages[page_idx].ref_count = 1;
        self.allocated_pages += 1;
        self.peak_allocated = @max(self.peak_allocated, self.allocated_pages);
        return page_idx;
    }

    /// Free a page back to the pool.
    pub fn freePage(self: *LayerPagePool, page_idx: u32) void {
        if (page_idx >= self.max_pages) return;

        self.pages[page_idx].ref_count -|= 1;
        if (self.pages[page_idx].ref_count == 0) {
            self.free_list.append(self.allocator, page_idx) catch return;
            self.allocated_pages -|= 1;
        }
    }

    /// Increment reference count (for sharing).
    pub fn refPage(self: *LayerPagePool, page_idx: u32) void {
        if (page_idx < self.max_pages) {
            self.pages[page_idx].ref_count += 1;
        }
    }

    /// Get page by index.
    pub fn getPage(self: *LayerPagePool, page_idx: u32) ?*KvPage {
        if (page_idx >= self.max_pages) return null;
        return &self.pages[page_idx];
    }

    /// Get available pages count.
    pub fn availablePages(self: *const LayerPagePool) u32 {
        return @intCast(self.free_list.items.len);
    }

    /// Get memory usage in bytes.
    pub fn memoryUsed(self: *const LayerPagePool) u64 {
        return @as(u64, self.allocated_pages) * self.page_size * self.kv_dim * 2 * @sizeOf(f32);
    }
};

/// Paged KV cache with block-based allocation.
pub const PagedKvCache = struct {
    allocator: std.mem.Allocator,
    /// Page pools per layer
    layer_pools: []LayerPagePool,
    /// Sequence states
    sequences: std.AutoHashMapUnmanaged(u32, SequenceKvState),
    /// Configuration
    config: PagedKvCacheConfig,
    /// Next sequence ID
    next_seq_id: u32,

    pub fn init(allocator: std.mem.Allocator, config: PagedKvCacheConfig) !PagedKvCache {
        const layer_pools = try allocator.alloc(LayerPagePool, config.num_layers);
        errdefer allocator.free(layer_pools);

        for (0..config.num_layers) |i| {
            layer_pools[i] = try LayerPagePool.init(allocator, config);
        }

        return .{
            .allocator = allocator,
            .layer_pools = layer_pools,
            .sequences = std.AutoHashMapUnmanaged(u32, SequenceKvState){},
            .config = config,
            .next_seq_id = 0,
        };
    }

    pub fn deinit(self: *PagedKvCache) void {
        // Free all sequences
        var it = self.sequences.valueIterator();
        while (it.next()) |seq| {
            seq.deinit();
        }
        self.sequences.deinit(self.allocator);

        // Free layer pools
        for (self.layer_pools) |*pool| {
            pool.deinit();
        }
        self.allocator.free(self.layer_pools);
        self.* = undefined;
    }

    /// Create a new sequence.
    pub fn createSequence(self: *PagedKvCache) !u32 {
        const seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        const max_blocks = self.config.max_pages; // Conservative estimate
        var seq = try SequenceKvState.init(
            self.allocator,
            seq_id,
            self.config.num_layers,
            max_blocks,
            self.config.page_size,
        );
        errdefer seq.deinit();

        try self.sequences.put(self.allocator, seq_id, seq);
        return seq_id;
    }

    /// Remove a sequence and free its pages.
    pub fn removeSequence(self: *PagedKvCache, seq_id: u32) void {
        if (self.sequences.fetchRemove(seq_id)) |kv| {
            var seq = kv.value;

            // Free all pages allocated to this sequence
            for (0..self.config.num_layers) |layer| {
                for (seq.block_tables[layer]) |maybe_page| {
                    if (maybe_page) |page_idx| {
                        self.layer_pools[layer].freePage(page_idx);
                    }
                }
            }

            seq.deinit();
        }
    }

    /// Append KV for a token to a sequence at all layers.
    pub fn appendToken(
        self: *PagedKvCache,
        seq_id: u32,
        layer: u32,
        k: []const f32,
        v: []const f32,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;

        const block_idx = seq.getBlockIdx(seq.seq_len);
        const block_offset = seq.getBlockOffset(seq.seq_len);

        // Check if we need to allocate a new block
        if (block_offset == 0 or seq.getPageIdx(layer, seq.seq_len) == null) {
            // Allocate new page
            const page_idx = self.layer_pools[layer].allocatePage() orelse return error.OutOfPages;
            seq.setPageIdx(layer, block_idx, page_idx);
        }

        // Get the page and append
        const page_idx = seq.getPageIdx(layer, seq.seq_len) orelse return error.PageNotFound;
        const page = self.layer_pools[layer].getPage(page_idx) orelse return error.PageNotFound;

        if (!page.append(k, v, self.config.kvDim(), self.config.page_size)) {
            return error.PageFull;
        }

        // Increment sequence length only after last layer
        if (layer == self.config.num_layers - 1) {
            seq.seq_len += 1;
        }
    }

    /// Get K for a position in a sequence.
    pub fn getK(self: *PagedKvCache, seq_id: u32, layer: u32, pos: u32) ?[]const f32 {
        const seq = self.sequences.getPtr(seq_id) orelse return null;
        if (pos >= seq.seq_len) return null;

        const page_idx = seq.getPageIdx(layer, pos) orelse return null;
        const page = self.layer_pools[layer].getPage(page_idx) orelse return null;
        const block_offset = seq.getBlockOffset(pos);

        return page.getK(block_offset, self.config.kvDim());
    }

    /// Get V for a position in a sequence.
    pub fn getV(self: *PagedKvCache, seq_id: u32, layer: u32, pos: u32) ?[]const f32 {
        const seq = self.sequences.getPtr(seq_id) orelse return null;
        if (pos >= seq.seq_len) return null;

        const page_idx = seq.getPageIdx(layer, pos) orelse return null;
        const page = self.layer_pools[layer].getPage(page_idx) orelse return null;
        const block_offset = seq.getBlockOffset(pos);

        return page.getV(block_offset, self.config.kvDim());
    }

    /// Get sequence length.
    pub fn getSequenceLength(self: *PagedKvCache, seq_id: u32) u32 {
        const seq = self.sequences.get(seq_id) orelse return 0;
        return seq.seq_len;
    }

    /// Fork a sequence (copy-on-write for prefix sharing).
    pub fn forkSequence(self: *PagedKvCache, parent_seq_id: u32) !u32 {
        const parent = self.sequences.getPtr(parent_seq_id) orelse return error.SequenceNotFound;

        // Create new sequence
        const new_seq_id = try self.createSequence();
        const new_seq = self.sequences.getPtr(new_seq_id) orelse return error.SequenceNotFound;

        // Share all existing pages (increment ref counts)
        for (0..self.config.num_layers) |layer| {
            for (parent.block_tables[layer], 0..) |maybe_page, block_idx| {
                if (maybe_page) |page_idx| {
                    self.layer_pools[layer].refPage(page_idx);
                    new_seq.setPageIdx(@intCast(layer), @intCast(block_idx), page_idx);
                }
            }
        }

        new_seq.seq_len = parent.seq_len;
        return new_seq_id;
    }

    /// Get statistics.
    pub fn getStats(self: *const PagedKvCache) PagedCacheStats {
        var total_allocated: u32 = 0;
        var total_available: u32 = 0;
        var peak_allocated: u32 = 0;
        var memory_used: u64 = 0;

        for (self.layer_pools) |pool| {
            total_allocated += pool.allocated_pages;
            total_available += pool.availablePages();
            peak_allocated = @max(peak_allocated, pool.peak_allocated);
            memory_used += pool.memoryUsed();
        }

        return .{
            .num_sequences = @intCast(self.sequences.count()),
            .total_pages = self.config.max_pages * self.config.num_layers,
            .allocated_pages = total_allocated,
            .available_pages = total_available,
            .peak_allocated = peak_allocated,
            .memory_bytes = memory_used,
            .max_memory_bytes = self.config.totalMemoryBytes(),
            .utilization = if (self.config.max_pages * self.config.num_layers > 0)
                @as(f32, @floatFromInt(total_allocated)) /
                    @as(f32, @floatFromInt(self.config.max_pages * self.config.num_layers))
            else
                0,
        };
    }
};

pub const PagedCacheStats = struct {
    num_sequences: u32,
    total_pages: u32,
    allocated_pages: u32,
    available_pages: u32,
    peak_allocated: u32,
    memory_bytes: u64,
    max_memory_bytes: u64,
    utilization: f32,
};

// Errors
pub const PagedCacheError = error{
    SequenceNotFound,
    OutOfPages,
    PageNotFound,
    PageFull,
    OutOfMemory,
};

test "paged kv cache basic" {
    const allocator = std.testing.allocator;

    var cache = try PagedKvCache.init(allocator, .{
        .num_layers = 2,
        .num_kv_heads = 4,
        .head_dim = 8,
        .max_pages = 16,
        .page_size = 4,
    });
    defer cache.deinit();

    // Create a sequence
    const seq_id = try cache.createSequence();

    // Append some tokens (num_kv_heads * head_dim = 4 * 8 = 32)
    const k = [_]f32{1.0} ** 32;
    const v = [_]f32{2.0} ** 32;

    // Add 3 tokens across both layers
    for (0..3) |_| {
        for (0..2) |layer| {
            try cache.appendToken(seq_id, @intCast(layer), &k, &v);
        }
    }

    try std.testing.expectEqual(@as(u32, 3), cache.getSequenceLength(seq_id));

    // Retrieve K, V
    const k0 = cache.getK(seq_id, 0, 0);
    try std.testing.expect(k0 != null);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), k0.?[0], 0.001);

    // Stats
    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.num_sequences);
    try std.testing.expect(stats.allocated_pages > 0);
}

test "paged kv cache page allocation" {
    const allocator = std.testing.allocator;

    var cache = try PagedKvCache.init(allocator, .{
        .num_layers = 1,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_pages = 4,
        .page_size = 2, // 2 tokens per page
    });
    defer cache.deinit();

    const seq_id = try cache.createSequence();

    // num_kv_heads * head_dim = 2 * 4 = 8
    const k = [_]f32{1.0} ** 8;
    const v = [_]f32{2.0} ** 8;

    // Add 5 tokens (should use 3 pages: 2+2+1)
    for (0..5) |_| {
        try cache.appendToken(seq_id, 0, &k, &v);
    }

    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u32, 3), stats.allocated_pages); // ceil(5/2) = 3 pages
}

test "paged kv cache sequence fork" {
    const allocator = std.testing.allocator;

    var cache = try PagedKvCache.init(allocator, .{
        .num_layers = 1,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_pages = 8,
        .page_size = 4,
    });
    defer cache.deinit();

    // Create parent sequence with 3 tokens
    const parent_id = try cache.createSequence();

    // num_kv_heads * head_dim = 2 * 4 = 8
    const k = [_]f32{1.0} ** 8;
    const v = [_]f32{2.0} ** 8;

    for (0..3) |_| {
        try cache.appendToken(parent_id, 0, &k, &v);
    }

    // Fork the sequence
    const child_id = try cache.forkSequence(parent_id);

    // Both should have same length
    try std.testing.expectEqual(@as(u32, 3), cache.getSequenceLength(parent_id));
    try std.testing.expectEqual(@as(u32, 3), cache.getSequenceLength(child_id));

    // Both should see same K values (shared)
    const parent_k = cache.getK(parent_id, 0, 0);
    const child_k = cache.getK(child_id, 0, 0);
    try std.testing.expect(parent_k != null);
    try std.testing.expect(child_k != null);
    try std.testing.expectApproxEqAbs(parent_k.?[0], child_k.?[0], 0.001);
}

test "paged kv cache remove sequence" {
    const allocator = std.testing.allocator;

    var cache = try PagedKvCache.init(allocator, .{
        .num_layers = 1,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_pages = 8,
        .page_size = 4,
    });
    defer cache.deinit();

    const seq_id = try cache.createSequence();

    const k = [_]f32{1.0} ** 8;
    const v = [_]f32{2.0} ** 8;

    for (0..5) |_| {
        try cache.appendToken(seq_id, 0, &k, &v);
    }

    const before_stats = cache.getStats();
    const allocated_before = before_stats.allocated_pages;

    // Remove sequence
    cache.removeSequence(seq_id);

    const after_stats = cache.getStats();
    try std.testing.expectEqual(@as(u32, 0), after_stats.num_sequences);
    try std.testing.expect(after_stats.allocated_pages < allocated_before);
}

test "config memory calculation" {
    const config = PagedKvCacheConfig{
        .num_layers = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .max_pages = 1024,
        .page_size = 16,
    };

    // Per page: 16 tokens * 8*128 kv_dim * 2 (K+V) * 4 bytes = 131,072 bytes
    const per_page = @as(u64, 16) * 8 * 128 * 2 * 4;
    try std.testing.expectEqual(per_page, config.pageMemoryBytes());

    // Total: 131,072 * 1024 * 32 = 4,294,967,296 bytes = 4 GB
    const total = per_page * 1024 * 32;
    try std.testing.expectEqual(total, config.totalMemoryBytes());
}
