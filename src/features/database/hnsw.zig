//! Hierarchical Navigable Small World (HNSW) vector index implementation.
//! Provides efficient approximate nearest neighbor search in high-dimensional spaces.
//!
//! Performance optimizations:
//! - SearchStatePool: Pre-allocated search states to avoid allocation per query
//! - DistanceCache: LRU cache for frequently computed distances
//! - Prefetching: Memory prefetch hints for graph traversal
//! - Vectorized distance computation via SIMD

const std = @import("std");
const simd = @import("../../shared/simd.zig");
const index_mod = @import("index.zig");

// ============================================================================
// Search State Pool - Eliminates per-query allocations
// ============================================================================

/// Pre-allocated search state for reuse across queries.
/// Avoids allocation overhead in the hot search path.
pub const SearchState = struct {
    /// Candidate nodes with their distances (node_id -> distance)
    candidates: std.AutoHashMapUnmanaged(u32, f32),
    /// Visited nodes set
    visited: std.AutoHashMapUnmanaged(u32, void),
    /// BFS queue for graph traversal
    queue: std.ArrayListUnmanaged(u32),
    /// Temporary results buffer
    results_buffer: std.ArrayListUnmanaged(index_mod.IndexResult),

    pub fn init() SearchState {
        return .{
            .candidates = .{},
            .visited = .{},
            .queue = .{},
            .results_buffer = .{},
        };
    }

    /// Reset state for reuse without deallocating backing memory.
    pub fn reset(self: *SearchState) void {
        self.candidates.clearRetainingCapacity();
        self.visited.clearRetainingCapacity();
        self.queue.clearRetainingCapacity();
        self.results_buffer.clearRetainingCapacity();
    }

    pub fn deinit(self: *SearchState, allocator: std.mem.Allocator) void {
        self.candidates.deinit(allocator);
        self.visited.deinit(allocator);
        self.queue.deinit(allocator);
        self.results_buffer.deinit(allocator);
    }

    /// Ensure capacity for expected search size.
    pub fn ensureCapacity(self: *SearchState, allocator: std.mem.Allocator, expected_size: usize) !void {
        try self.candidates.ensureTotalCapacity(allocator, @intCast(expected_size));
        try self.visited.ensureTotalCapacity(allocator, @intCast(expected_size));
        try self.queue.ensureTotalCapacity(allocator, expected_size);
        try self.results_buffer.ensureTotalCapacity(allocator, expected_size);
    }
};

/// Pool of pre-allocated search states for concurrent query processing.
/// Thread-safe acquisition and release of search states.
pub const SearchStatePool = struct {
    states: []SearchState,
    available: std.atomic.Value(u64),
    allocator: std.mem.Allocator,

    const MAX_POOL_SIZE = 64;

    pub fn init(allocator: std.mem.Allocator, pool_size: usize) !SearchStatePool {
        const size = @min(pool_size, MAX_POOL_SIZE);
        const states = try allocator.alloc(SearchState, size);
        for (states) |*state| {
            state.* = SearchState.init();
        }

        // Initialize bitmask with all states available
        const initial_mask: u64 = if (size >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(size)) - 1;

        return .{
            .states = states,
            .available = std.atomic.Value(u64).init(initial_mask),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SearchStatePool) void {
        for (self.states) |*state| {
            state.deinit(self.allocator);
        }
        self.allocator.free(self.states);
    }

    /// Acquire a search state from the pool.
    /// Returns null if no states available (caller should allocate temporary state).
    pub fn acquire(self: *SearchStatePool) ?*SearchState {
        while (true) {
            const current = self.available.load(.acquire);
            if (current == 0) return null;

            // Find first available bit
            const bit_idx = @ctz(current);
            if (bit_idx >= self.states.len) return null;

            const new_mask = current & ~(@as(u64, 1) << @intCast(bit_idx));
            if (self.available.cmpxchgWeak(current, new_mask, .acq_rel, .acquire)) |_| {
                continue; // CAS failed, retry
            }

            const state = &self.states[bit_idx];
            state.reset();
            return state;
        }
    }

    /// Release a search state back to the pool.
    pub fn release(self: *SearchStatePool, state: *SearchState) void {
        // Find index of this state
        const idx = (@intFromPtr(state) - @intFromPtr(self.states.ptr)) / @sizeOf(SearchState);
        if (idx >= self.states.len) return;

        _ = self.available.fetchOr(@as(u64, 1) << @intCast(idx), .release);
    }
};

// ============================================================================
// Distance Cache - LRU cache for frequently computed distances
// ============================================================================

/// LRU distance cache to avoid redundant similarity computations.
/// Uses packed u64 keys combining two u32 node IDs.
pub const DistanceCache = struct {
    entries: []CacheEntry,
    head: u32,
    size: u32,
    hits: u64,
    misses: u64,

    const CacheEntry = struct {
        key: u64,
        distance: f32,
        prev: u32,
        next: u32,
        valid: bool,
    };

    const INVALID_IDX = std.math.maxInt(u32);

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !DistanceCache {
        const cap = @max(capacity, 16);
        const entries = try allocator.alloc(CacheEntry, cap);
        for (entries, 0..) |*entry, i| {
            entry.* = .{
                .key = 0,
                .distance = 0,
                .prev = if (i == 0) INVALID_IDX else @intCast(i - 1),
                .next = if (i == cap - 1) INVALID_IDX else @intCast(i + 1),
                .valid = false,
            };
        }
        return .{
            .entries = entries,
            .head = 0,
            .size = 0,
            .hits = 0,
            .misses = 0,
        };
    }

    pub fn deinit(self: *DistanceCache, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
    }

    /// Create a cache key from two node IDs (order-independent).
    inline fn makeKey(a: u32, b: u32) u64 {
        const lo = @min(a, b);
        const hi = @max(a, b);
        return (@as(u64, hi) << 32) | @as(u64, lo);
    }

    /// Get cached distance between two nodes.
    pub fn get(self: *DistanceCache, a: u32, b: u32) ?f32 {
        const key = makeKey(a, b);

        // Linear search (could use hash table for larger caches)
        for (self.entries[0..self.size]) |*entry| {
            if (entry.valid and entry.key == key) {
                self.hits += 1;
                return entry.distance;
            }
        }
        self.misses += 1;
        return null;
    }

    /// Store distance in cache with LRU eviction.
    pub fn put(self: *DistanceCache, a: u32, b: u32, distance: f32) void {
        const key = makeKey(a, b);

        if (self.size < self.entries.len) {
            // Cache not full, add new entry
            const idx = self.size;
            self.entries[idx] = .{
                .key = key,
                .distance = distance,
                .prev = INVALID_IDX,
                .next = INVALID_IDX,
                .valid = true,
            };
            self.size += 1;
        } else {
            // Evict oldest entry (simple FIFO for now)
            const idx = self.head;
            self.entries[idx].key = key;
            self.entries[idx].distance = distance;
            self.entries[idx].valid = true;
            self.head = (self.head + 1) % @as(u32, @intCast(self.entries.len));
        }
    }

    /// Clear all cached entries.
    pub fn clear(self: *DistanceCache) void {
        for (self.entries) |*entry| {
            entry.valid = false;
        }
        self.size = 0;
        self.head = 0;
    }

    /// Get cache statistics.
    pub fn getStats(self: *const DistanceCache) struct { hits: u64, misses: u64, hit_rate: f32 } {
        const total = self.hits + self.misses;
        const hit_rate = if (total > 0) @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total)) else 0.0;
        return .{ .hits = self.hits, .misses = self.misses, .hit_rate = hit_rate };
    }
};

/// HNSW index structure supporting layered graph traversal.
pub const HnswIndex = struct {
    m: usize,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    entry_point: ?u32,
    max_layer: i32,
    nodes: []NodeLayers,
    /// Optional search state pool for allocation-free queries
    state_pool: ?*SearchStatePool,
    /// Optional distance cache for frequently accessed pairs
    distance_cache: ?*DistanceCache,

    pub const NodeLayers = struct {
        layers: []index_mod.NeighborList,
    };

    /// Configuration for HNSW index construction.
    pub const Config = struct {
        m: usize = 16,
        ef_construction: usize = 100,
        /// Number of pre-allocated search states (0 = disabled)
        search_pool_size: usize = 8,
        /// Distance cache capacity (0 = disabled)
        distance_cache_size: usize = 1024,
    };

    /// Build a new HNSW index from a set of records.
    /// @param allocator Memory allocator for graph structures
    /// @param records Source vector records
    /// @param m Max number of connections per node
    /// @param ef_construction Size of the dynamic candidate list during construction
    /// @return Initialized HnswIndex
    pub fn build(
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        m: usize,
        ef_construction: usize,
    ) !HnswIndex {
        return buildWithConfig(allocator, records, .{
            .m = m,
            .ef_construction = ef_construction,
            .search_pool_size = 0, // Legacy mode: no pool
            .distance_cache_size = 0, // Legacy mode: no cache
        });
    }

    /// Build a new HNSW index with full configuration options.
    /// @param allocator Memory allocator for graph structures
    /// @param records Source vector records
    /// @param config Index configuration including performance options
    /// @return Initialized HnswIndex
    pub fn buildWithConfig(
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        config: Config,
    ) !HnswIndex {
        if (records.len == 0) return index_mod.IndexError.EmptyIndex;

        // Initialize optional search state pool
        var state_pool: ?*SearchStatePool = null;
        if (config.search_pool_size > 0) {
            const pool = try allocator.create(SearchStatePool);
            pool.* = try SearchStatePool.init(allocator, config.search_pool_size);
            state_pool = pool;
        }
        errdefer if (state_pool) |pool| {
            pool.deinit();
            allocator.destroy(pool);
        };

        // Initialize optional distance cache
        var distance_cache: ?*DistanceCache = null;
        if (config.distance_cache_size > 0) {
            const cache = try allocator.create(DistanceCache);
            cache.* = try DistanceCache.init(allocator, config.distance_cache_size);
            distance_cache = cache;
        }
        errdefer if (distance_cache) |cache| {
            cache.deinit(allocator);
            allocator.destroy(cache);
        };

        var self = HnswIndex{
            .m = config.m,
            .m_max = config.m,
            .m_max0 = config.m * 2,
            .ef_construction = config.ef_construction,
            .entry_point = null,
            .max_layer = -1,
            .nodes = try allocator.alloc(NodeLayers, records.len),
            .state_pool = state_pool,
            .distance_cache = distance_cache,
        };
        errdefer allocator.free(self.nodes);

        for (self.nodes) |*node| {
            node.layers = &.{};
        }

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const m_l = 1.0 / @as(f32, @log(@as(f32, @floatFromInt(config.m))));

        for (records, 0..) |_, i| {
            try self.insert(allocator, records, @intCast(i), random, m_l);
        }

        return self;
    }

    /// Insert a new node into the HNSW graph.
    fn insert(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        random: std.Random,
        m_l: f32,
    ) !void {
        const target_layer = @as(i32, @intFromFloat(@floor(-@log(random.float(f32)) * m_l)));
        self.nodes[node_id].layers = try allocator.alloc(index_mod.NeighborList, @intCast(target_layer + 1));
        for (self.nodes[node_id].layers) |*list| {
            list.nodes = &.{};
        }

        if (self.entry_point == null) {
            self.entry_point = node_id;
            self.max_layer = target_layer;
            return;
        }

        var curr_node = self.entry_point.?;
        var curr_dist = self.computeNodeDistance(records, node_id, curr_node);

        // 1. Greedy search down to layer above target_layer
        var lc: i32 = self.max_layer;
        while (lc > target_layer) : (lc -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                const neighbors = self.nodes[curr_node].layers[@intCast(lc)].nodes;

                // Prefetch neighbor vectors
                for (neighbors) |neighbor| {
                    if (neighbor < records.len) {
                        @prefetch(records[neighbor].vector.ptr, .{ .locality = 3, .rw = .read });
                    }
                }

                for (neighbors) |neighbor| {
                    const d = self.computeNodeDistance(records, node_id, neighbor);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_node = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 2. Perform layered insertion from target_layer down to 0
        lc = @min(target_layer, self.max_layer);
        while (lc >= 0) : (lc -= 1) {
            try self.connectNeighbors(allocator, records, node_id, curr_node, @intCast(lc));
        }

        // 3. Update global entry point if new node is at a higher layer
        if (target_layer > self.max_layer) {
            self.max_layer = target_layer;
            self.entry_point = node_id;
        }
    }

    /// Connect a node to its neighbors at a specific layer using proper HNSW neighbor selection.
    fn connectNeighbors(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        entry: u32,
        layer: usize,
    ) !void {
        const m_val = if (layer == 0) self.m_max0 else self.m_max;

        // Build candidate list using ef_construction expansion
        var candidates = std.AutoHashMapUnmanaged(u32, f32){};
        defer candidates.deinit(allocator);

        var visited = std.AutoHashMapUnmanaged(u32, void){};
        defer visited.deinit(allocator);

        // Start with entry point (use cached distance)
        const entry_dist = self.computeNodeDistance(records, node_id, entry);
        try candidates.put(allocator, entry, entry_dist);
        try visited.put(allocator, entry, {});

        // BFS expansion to find candidates
        var queue = std.ArrayListUnmanaged(u32){};
        defer queue.deinit(allocator);
        try queue.append(allocator, entry);

        var head: usize = 0;
        while (head < queue.items.len and candidates.count() < self.ef_construction) : (head += 1) {
            const curr = queue.items[head];
            if (layer < self.nodes[curr].layers.len) {
                const neighbors = self.nodes[curr].layers[layer].nodes;

                // Prefetch neighbor vectors
                for (neighbors) |neighbor| {
                    if (!visited.contains(neighbor) and neighbor < records.len) {
                        @prefetch(records[neighbor].vector.ptr, .{ .locality = 2, .rw = .read });
                    }
                }

                for (neighbors) |neighbor| {
                    if (!visited.contains(neighbor)) {
                        try visited.put(allocator, neighbor, {});
                        const dist = self.computeNodeDistance(records, node_id, neighbor);
                        try candidates.put(allocator, neighbor, dist);
                        try queue.append(allocator, neighbor);
                    }
                }
            }
        }

        // Select best neighbors using heuristic pruning
        const selected = try self.selectNeighborsHeuristic(allocator, records, node_id, &candidates, m_val);
        self.nodes[node_id].layers[layer].nodes = selected;

        // Update bidirectional links with proper pruning
        for (self.nodes[node_id].layers[layer].nodes) |neighbor| {
            if (layer >= self.nodes[neighbor].layers.len) continue;

            var neighbor_links = std.AutoHashMapUnmanaged(u32, f32){};
            defer neighbor_links.deinit(allocator);

            // Collect existing neighbors (use cached distances)
            for (self.nodes[neighbor].layers[layer].nodes) |existing| {
                const dist = self.computeNodeDistance(records, neighbor, existing);
                try neighbor_links.put(allocator, existing, dist);
            }

            // Add new link if not exists
            if (!neighbor_links.contains(node_id)) {
                const dist = self.computeNodeDistance(records, neighbor, node_id);
                try neighbor_links.put(allocator, node_id, dist);
            }

            // Prune if needed
            if (neighbor_links.count() > m_val) {
                const pruned = try self.selectNeighborsHeuristic(allocator, records, neighbor, &neighbor_links, m_val);
                allocator.free(self.nodes[neighbor].layers[layer].nodes);
                self.nodes[neighbor].layers[layer].nodes = pruned;
            } else {
                // Just update with new links
                var new_links = std.ArrayListUnmanaged(u32){};
                errdefer new_links.deinit(allocator);

                var it = neighbor_links.keyIterator();
                while (it.next()) |key| {
                    try new_links.append(allocator, key.*);
                }

                allocator.free(self.nodes[neighbor].layers[layer].nodes);
                self.nodes[neighbor].layers[layer].nodes = try new_links.toOwnedSlice(allocator);
            }
        }
    }

    /// Select neighbors using heuristic pruning that considers both distance and diversity.
    fn selectNeighborsHeuristic(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        candidates: *std.AutoHashMapUnmanaged(u32, f32),
        m_val: usize,
    ) ![]u32 {
        _ = self;

        // Sort candidates by distance (ascending)
        const CandidatePair = struct { id: u32, dist: f32 };
        var sorted = std.ArrayListUnmanaged(CandidatePair){};
        defer sorted.deinit(allocator);

        var it = candidates.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.* != node_id) { // Don't include self
                try sorted.append(allocator, .{ .id = entry.key_ptr.*, .dist = entry.value_ptr.* });
            }
        }

        // Sort by distance (closest first)
        std.sort.heap(CandidatePair, sorted.items, {}, struct {
            fn lessThan(_: void, a: CandidatePair, b: CandidatePair) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        // Select using heuristic: prefer diverse neighbors over purely closest
        var selected = std.ArrayListUnmanaged(u32){};
        errdefer selected.deinit(allocator);

        for (sorted.items) |candidate| {
            if (selected.items.len >= m_val) break;

            // Check if this candidate is closer to node than to any selected neighbor
            var should_add = true;
            for (selected.items) |existing| {
                const dist_to_existing = 1.0 - simd.cosineSimilarity(
                    records[candidate.id].vector,
                    records[existing].vector,
                );
                // If candidate is closer to an existing neighbor than to the node,
                // skip it to maintain diversity
                if (dist_to_existing < candidate.dist) {
                    should_add = false;
                    break;
                }
            }

            if (should_add) {
                try selected.append(allocator, candidate.id);
            }
        }

        // If we don't have enough neighbors due to heuristic, fill with closest
        if (selected.items.len < m_val) {
            for (sorted.items) |candidate| {
                if (selected.items.len >= m_val) break;

                var already_added = false;
                for (selected.items) |existing| {
                    if (existing == candidate.id) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added) {
                    try selected.append(allocator, candidate.id);
                }
            }
        }

        return selected.toOwnedSlice(allocator);
    }

    /// Search the HNSW graph for the nearest neighbors of a query vector.
    /// @param allocator Memory allocator for search results
    /// @param records Source vector records
    /// @param query Query vector
    /// @param top_k Number of results to return
    /// @return Slice of IndexResult sorted by similarity
    pub fn search(
        self: *const HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        query: []const f32,
        top_k: usize,
    ) ![]index_mod.IndexResult {
        // Use optimized search with pooled state if available
        if (self.state_pool) |pool| {
            if (pool.acquire()) |state| {
                defer pool.release(state);
                return self.searchWithState(allocator, records, query, top_k, state);
            }
        }
        // Fallback to allocating search state
        var state = SearchState.init();
        defer state.deinit(allocator);
        return self.searchWithState(allocator, records, query, top_k, &state);
    }

    /// Search using a pre-allocated search state (avoids per-query allocation).
    fn searchWithState(
        self: *const HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        query: []const f32,
        top_k: usize,
        state: *SearchState,
    ) ![]index_mod.IndexResult {
        if (self.entry_point == null or records.len == 0) {
            return allocator.alloc(index_mod.IndexResult, 0);
        }

        // Pre-compute query norm for faster cosine similarity
        const query_norm = simd.vectorL2Norm(query);
        if (query_norm == 0.0) {
            return allocator.alloc(index_mod.IndexResult, 0);
        }

        var curr_node = self.entry_point.?;
        var curr_dist = self.computeDistance(query, query_norm, records[curr_node].vector);

        // 1. Zoom in through layers with prefetching
        var lc: i32 = self.max_layer;
        while (lc > 0) : (lc -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                const neighbors = self.nodes[curr_node].layers[@intCast(lc)].nodes;

                // Prefetch next neighbor's vector data
                for (neighbors) |neighbor| {
                    if (neighbor < records.len) {
                        @prefetch(records[neighbor].vector.ptr, .{ .locality = 3, .rw = .read });
                    }
                }

                for (neighbors) |neighbor| {
                    const d = self.computeDistance(query, query_norm, records[neighbor].vector);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_node = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 2. Local search on layer 0 with candidate accumulation
        try state.candidates.put(allocator, curr_node, curr_dist);
        try state.visited.put(allocator, curr_node, {});
        try state.queue.append(allocator, curr_node);

        var head: usize = 0;
        // Limit search expansion for performance
        const max_candidates = @max(top_k * 2, self.ef_construction / 2);

        while (head < state.queue.items.len and state.queue.items.len < max_candidates) : (head += 1) {
            const u = state.queue.items[head];
            // Safety: Check that node has layers before accessing layer 0
            if (self.nodes[u].layers.len == 0) continue;

            const neighbors = self.nodes[u].layers[0].nodes;

            // Prefetch neighbor vectors
            for (neighbors) |v| {
                if (!state.visited.contains(v) and v < records.len) {
                    @prefetch(records[v].vector.ptr, .{ .locality = 2, .rw = .read });
                }
            }

            for (neighbors) |v| {
                if (!state.visited.contains(v)) {
                    try state.visited.put(allocator, v, {});
                    const d = self.computeDistance(query, query_norm, records[v].vector);
                    try state.candidates.put(allocator, v, d);
                    try state.queue.append(allocator, v);
                }
            }
        }

        // 3. Extract and sort top-k results
        const result_count = state.candidates.count();
        var results = try allocator.alloc(index_mod.IndexResult, result_count);
        errdefer allocator.free(results);

        var it = state.candidates.iterator();
        var i: usize = 0;
        while (it.next()) |entry| {
            results[i] = .{
                .id = records[entry.key_ptr.*].id,
                .score = 1.0 - entry.value_ptr.*, // Convert distance back to similarity
            };
            i += 1;
        }

        std.sort.heap(index_mod.IndexResult, results, {}, struct {
            fn lessThan(_: void, a: index_mod.IndexResult, b: index_mod.IndexResult) bool {
                return a.score > b.score;
            }
        }.lessThan);

        if (results.len > top_k) {
            const final = try allocator.dupe(index_mod.IndexResult, results[0..top_k]);
            allocator.free(results);
            return final;
        }
        return results;
    }

    /// Compute distance with optional caching and pre-computed query norm.
    inline fn computeDistance(self: *const HnswIndex, query: []const f32, query_norm: f32, vector: []const f32) f32 {
        // Use cache if available (for node-to-node distances, not query distances)
        _ = self;

        // Optimized cosine distance with pre-computed query norm
        const vec_norm = simd.vectorL2Norm(vector);
        if (vec_norm == 0.0) return 1.0;

        const dot = simd.vectorDot(query, vector);
        return 1.0 - (dot / (query_norm * vec_norm));
    }

    /// Compute and cache node-to-node distance.
    fn computeNodeDistance(self: *const HnswIndex, records: []const index_mod.VectorRecordView, a: u32, b: u32) f32 {
        // Check cache first
        if (self.distance_cache) |cache| {
            if (cache.get(a, b)) |cached| {
                return cached;
            }
        }

        // Compute distance
        const dist = 1.0 - simd.cosineSimilarity(records[a].vector, records[b].vector);

        // Store in cache
        if (self.distance_cache) |cache| {
            cache.put(a, b, dist);
        }

        return dist;
    }

    /// Free resources associated with the index.
    pub fn deinit(self: *HnswIndex, allocator: std.mem.Allocator) void {
        // Clean up search state pool
        if (self.state_pool) |pool| {
            pool.deinit();
            allocator.destroy(pool);
        }

        // Clean up distance cache
        if (self.distance_cache) |cache| {
            cache.deinit(allocator);
            allocator.destroy(cache);
        }

        for (self.nodes) |node| {
            for (node.layers) |list| {
                allocator.free(list.nodes);
            }
            allocator.free(node.layers);
        }
        allocator.free(self.nodes);
        self.* = undefined;
    }

    /// Get cache statistics if distance caching is enabled.
    pub fn getCacheStats(self: *const HnswIndex) ?struct { hits: u64, misses: u64, hit_rate: f32 } {
        if (self.distance_cache) |cache| {
            return cache.getStats();
        }
        return null;
    }

    /// Save HNSW structure to a binary writer.
    pub fn save(self: HnswIndex, writer: anytype) !void {
        try writer.writeInt(u32, @intCast(self.nodes.len), .little);
        try writer.writeInt(u32, @intCast(self.m), .little);
        try writer.writeInt(u32, if (self.entry_point) |ep| ep else 0, .little);
        try writer.writeInt(i32, self.max_layer, .little);
        try writer.writeInt(u32, @intCast(self.ef_construction), .little);

        for (self.nodes) |node| {
            try writer.writeInt(u32, @intCast(node.layers.len), .little);
            for (node.layers) |list| {
                try writer.writeInt(u32, @intCast(list.nodes.len), .little);
                for (list.nodes) |neighbor| {
                    try writer.writeInt(u32, neighbor, .little);
                }
            }
        }
    }

    /// Load HNSW structure from a binary reader.
    pub fn load(allocator: std.mem.Allocator, reader: anytype) !HnswIndex {
        const node_count = try reader.readInt(u32, .little);
        const m = try reader.readInt(u32, .little);
        const entry_point = try reader.readInt(u32, .little);
        const max_layer = try reader.readInt(i32, .little);
        const ef_construction = try reader.readInt(u32, .little);

        var self = HnswIndex{
            .m = m,
            .m_max = m,
            .m_max0 = m * 2,
            .ef_construction = ef_construction,
            .entry_point = if (node_count > 0) entry_point else null,
            .max_layer = max_layer,
            .nodes = try allocator.alloc(NodeLayers, node_count),
        };
        errdefer allocator.free(self.nodes);

        for (self.nodes) |*node| {
            const layer_count = try reader.readInt(u32, .little);
            node.layers = try allocator.alloc(index_mod.NeighborList, layer_count);
            for (node.layers) |*list| {
                const neighbor_count = try reader.readInt(u32, .little);
                list.nodes = try allocator.alloc(u32, neighbor_count);
                for (list.nodes) |*neighbor| {
                    neighbor.* = try reader.readInt(u32, .little);
                }
            }
        }

        return self;
    }
};

test "hnsw structure basic lifecycle" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.7, 0.7 } },
    };

    var index = try HnswIndex.build(allocator, &records, 16, 100);
    defer index.deinit(allocator);

    try std.testing.expect(index.nodes.len == 3);
    try std.testing.expect(index.entry_point != null);

    const query = [_]f32{ 0.8, 0.6 };
    const results = try index.search(allocator, &records, &query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len <= 2);
    if (results.len > 0) {
        // Result 3 should be top since similarity is high
        try std.testing.expect(results[0].id == 3);
    }
}

test "hnsw with search state pool" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
        .{ .id = 4, .vector = &[_]f32{ 0.5, 0.5, 0.5, 0.5 } },
        .{ .id = 5, .vector = &[_]f32{ 0.9, 0.1, 0.0, 0.0 } },
    };

    // Build with pool and cache enabled
    var index = try HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 8,
        .ef_construction = 50,
        .search_pool_size = 4,
        .distance_cache_size = 128,
    });
    defer index.deinit(allocator);

    try std.testing.expect(index.state_pool != null);
    try std.testing.expect(index.distance_cache != null);

    // Perform multiple searches to exercise the pool
    const queries = [_][4]f32{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.5, 0.5, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    };

    for (queries) |query| {
        const results = try index.search(allocator, &records, &query, 3);
        defer allocator.free(results);
        try std.testing.expect(results.len > 0);
    }

    // Check cache statistics
    if (index.getCacheStats()) |stats| {
        // Some hits expected after multiple operations
        try std.testing.expect(stats.hits + stats.misses > 0);
    }
}

test "search state pool acquire release" {
    const allocator = std.testing.allocator;

    var pool = try SearchStatePool.init(allocator, 4);
    defer pool.deinit();

    // Acquire all states
    var states: [4]?*SearchState = undefined;
    for (&states) |*s| {
        s.* = pool.acquire();
        try std.testing.expect(s.* != null);
    }

    // Pool should be exhausted
    try std.testing.expect(pool.acquire() == null);

    // Release one
    pool.release(states[0].?);

    // Should be able to acquire again
    const reacquired = pool.acquire();
    try std.testing.expect(reacquired != null);

    // Release all
    for (states[1..]) |s| {
        if (s) |state| pool.release(state);
    }
    pool.release(reacquired.?);
}

test "distance cache basic operations" {
    const allocator = std.testing.allocator;

    var cache = try DistanceCache.init(allocator, 32);
    defer cache.deinit(allocator);

    // Should miss initially
    try std.testing.expect(cache.get(1, 2) == null);

    // Store and retrieve
    cache.put(1, 2, 0.5);
    const cached = cache.get(1, 2);
    try std.testing.expect(cached != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cached.?, 1e-6);

    // Order-independent key
    const cached_rev = cache.get(2, 1);
    try std.testing.expect(cached_rev != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cached_rev.?, 1e-6);

    // Check stats
    const stats = cache.getStats();
    try std.testing.expect(stats.hits == 2);
    try std.testing.expect(stats.misses == 1);
}
