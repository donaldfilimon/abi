//! Hierarchical Navigable Small World (HNSW) vector index implementation.
//! Provides efficient approximate nearest neighbor search in high-dimensional spaces.
//!
//! Performance optimizations:
//! - SearchStatePool: Pre-allocated search states to avoid allocation per query
//! - DistanceCache: LRU cache for frequently computed distances
//! - Prefetching: Memory prefetch hints for graph traversal
//! - Vectorized distance computation via SIMD
//! - GPU-accelerated batch distance computation for large neighbor sets

const std = @import("std");
const build_options = @import("build_options");
const simd = @import("../../services/shared/simd/mod.zig");
const index_mod = @import("index.zig");
const gpu_accel = @import("gpu_accel.zig");

// Re-export extracted sub-modules
pub const search_state = @import("search_state.zig");
pub const distance_cache = @import("distance_cache.zig");

pub const SearchState = search_state.SearchState;
pub const SearchStatePool = search_state.SearchStatePool;
pub const DistanceCache = distance_cache.DistanceCache;

/// Configuration for adaptive search that auto-tunes ef for target recall.
pub const AdaptiveSearchConfig = struct {
    initial_ef: u32 = 50,
    max_ef: u32 = 500,
    ef_step: u32 = 50,
    target_recall: f32 = 0.95,
    max_iterations: u32 = 5,
};

/// Statistics about the HNSW index structure.
pub const IndexStats = struct {
    num_vectors: usize,
    num_layers: u32,
    avg_degree: f32,
    memory_bytes: usize,
    entry_point_id: ?u32,
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
    /// Optional GPU accelerator for batch distance computation
    gpu_accelerator: ?*gpu_accel.GpuAccelerator,
    /// Pre-computed L2 norms for each vector (eliminates redundant norm computation)
    norms: []f32,
    /// Allocator used for construction
    allocator: std.mem.Allocator,

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
        /// Enable GPU acceleration for batch distance computation (requires -Denable-gpu)
        enable_gpu: bool = build_options.enable_gpu,
        /// Minimum batch size to trigger GPU acceleration
        gpu_batch_threshold: usize = 256,
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
            .enable_gpu = false, // Legacy mode: no GPU
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
        var dist_cache_ptr: ?*DistanceCache = null;
        if (config.distance_cache_size > 0) {
            const cache = try allocator.create(DistanceCache);
            cache.* = try DistanceCache.init(allocator, config.distance_cache_size);
            dist_cache_ptr = cache;
        }
        errdefer if (dist_cache_ptr) |cache| {
            cache.deinit(allocator);
            allocator.destroy(cache);
        };

        // Initialize optional GPU accelerator for batch distance computation
        var gpu_accelerator: ?*gpu_accel.GpuAccelerator = null;
        if (config.enable_gpu and build_options.enable_gpu) {
            const accel = allocator.create(gpu_accel.GpuAccelerator) catch null;
            if (accel) |a| {
                a.* = gpu_accel.GpuAccelerator.init(allocator, .{
                    .batch_threshold = config.gpu_batch_threshold,
                }) catch {
                    allocator.destroy(a);
                    gpu_accelerator = null;
                };
                if (a.isGpuAvailable() or true) { // Keep even for SIMD acceleration
                    gpu_accelerator = a;
                } else {
                    a.deinit();
                    allocator.destroy(a);
                }
            }
        }
        errdefer if (gpu_accelerator) |accel| {
            accel.deinit();
            allocator.destroy(accel);
        };

        // Pre-compute L2 norms for all vectors (avoids redundant norm computation
        // during construction — each norm would otherwise be computed O(ef_construction) times)
        const norms = try allocator.alloc(f32, records.len);
        errdefer allocator.free(norms);
        for (records, 0..) |rec, idx| {
            norms[idx] = if (rec.vector.len > 0) simd.vectorL2Norm(rec.vector) else 0.0;
        }

        var self = HnswIndex{
            .m = config.m,
            .m_max = config.m,
            .m_max0 = config.m * 2,
            .ef_construction = config.ef_construction,
            .entry_point = null,
            .max_layer = -1,
            .nodes = try allocator.alloc(NodeLayers, records.len),
            .state_pool = state_pool,
            .distance_cache = dist_cache_ptr,
            .gpu_accelerator = gpu_accelerator,
            .norms = norms,
            .allocator = allocator,
        };
        errdefer allocator.free(self.nodes);

        for (self.nodes) |*node| {
            node.layers = &.{};
        }

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const m_l = 1.0 / @as(f32, @log(@as(f32, @floatFromInt(config.m))));

        // Use arena allocator for per-insertion temporaries (HashMaps, ArrayLists)
        // — bulk-freed after each insert instead of individual alloc/free per container
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        for (records, 0..) |_, i| {
            try self.insert(allocator, arena.allocator(), records, @intCast(i), random, m_l);
            // Reset arena for next insertion — frees all temporaries in bulk
            _ = arena.reset(.retain_capacity);
        }

        return self;
    }

    /// Insert a new node into the HNSW graph.
    ///
    /// Implements the standard HNSW insertion algorithm:
    /// 1. Determine target layer using exponential distribution
    /// 2. Greedy descent from max_layer to target_layer + 1
    /// 3. Layer-by-layer neighbor connection from target_layer to 0
    /// 4. Update entry point if new node is at a higher layer
    fn insert(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        temp_allocator: std.mem.Allocator,
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
            try self.connectNeighbors(allocator, temp_allocator, records, node_id, curr_node, @intCast(lc));
        }

        // 3. Update global entry point if new node is at a higher layer
        if (target_layer > self.max_layer) {
            self.max_layer = target_layer;
            self.entry_point = node_id;
        }
    }

    /// Connect a node to its neighbors at a specific layer using proper HNSW neighbor selection.
    ///
    /// Uses ef_construction-based candidate expansion and heuristic pruning to select
    /// diverse, high-quality neighbors. Updates bidirectional links and prunes existing
    /// neighbors if they exceed m_max (or m_max0 for layer 0).
    fn connectNeighbors(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        temp_allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        entry: u32,
        layer: usize,
    ) !void {
        const m_val = if (layer == 0) self.m_max0 else self.m_max;

        // Build candidate list using ef_construction expansion
        // (temporaries use arena allocator — bulk freed by caller)
        var candidates = std.AutoHashMapUnmanaged(u32, f32){};
        defer candidates.deinit(temp_allocator);

        var visited = std.AutoHashMapUnmanaged(u32, void){};
        defer visited.deinit(temp_allocator);

        // Start with entry point (use cached distance)
        const entry_dist = self.computeNodeDistance(records, node_id, entry);
        try candidates.put(temp_allocator, entry, entry_dist);
        try visited.put(temp_allocator, entry, {});

        // BFS expansion to find candidates
        var queue = std.ArrayListUnmanaged(u32).empty;
        defer queue.deinit(temp_allocator);
        try queue.append(temp_allocator, entry);

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
                        try visited.put(temp_allocator, neighbor, {});
                        const dist = self.computeNodeDistance(records, node_id, neighbor);
                        try candidates.put(temp_allocator, neighbor, dist);
                        try queue.append(temp_allocator, neighbor);
                    }
                }
            }
        }

        // Select best neighbors using heuristic pruning
        const selected = try self.selectNeighborsHeuristic(allocator, temp_allocator, records, node_id, &candidates, m_val);
        self.nodes[node_id].layers[layer].nodes = selected;

        // Update bidirectional links with proper pruning
        const node_neighbors = self.nodes[node_id].layers[layer].nodes;
        for (node_neighbors, 0..) |neighbor, neighbor_idx| {
            if (layer >= self.nodes[neighbor].layers.len) continue;

            // Prefetch next neighbor's layer structure for the next iteration
            if (neighbor_idx + 1 < node_neighbors.len) {
                const next_neighbor = node_neighbors[neighbor_idx + 1];
                if (next_neighbor < self.nodes.len and self.nodes[next_neighbor].layers.len > layer) {
                    @prefetch(self.nodes[next_neighbor].layers[layer].nodes.ptr, .{
                        .locality = 2,
                        .rw = .read,
                        .cache = .data,
                    });
                }
            }

            var neighbor_links = std.AutoHashMapUnmanaged(u32, f32){};
            defer neighbor_links.deinit(temp_allocator);

            const existing_neighbors = self.nodes[neighbor].layers[layer].nodes;

            // Prefetch vectors for existing neighbors
            for (existing_neighbors) |existing| {
                if (existing < records.len) {
                    @prefetch(records[existing].vector.ptr, .{
                        .locality = 2,
                        .rw = .read,
                        .cache = .data,
                    });
                }
            }

            // Collect existing neighbors (use cached distances)
            for (existing_neighbors) |existing| {
                const dist = self.computeNodeDistance(records, neighbor, existing);
                try neighbor_links.put(temp_allocator, existing, dist);
            }

            // Add new link if not exists
            if (!neighbor_links.contains(node_id)) {
                const dist = self.computeNodeDistance(records, neighbor, node_id);
                try neighbor_links.put(temp_allocator, node_id, dist);
            }

            // Prune if needed
            if (neighbor_links.count() > m_val) {
                // selectNeighborsHeuristic uses persistent allocator for the owned slice
                const pruned = try self.selectNeighborsHeuristic(allocator, temp_allocator, records, neighbor, &neighbor_links, m_val);
                allocator.free(self.nodes[neighbor].layers[layer].nodes);
                self.nodes[neighbor].layers[layer].nodes = pruned;
            } else {
                // Just update with new links (persistent allocation — outlives function)
                var new_links = std.ArrayListUnmanaged(u32).empty;
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
    ///
    /// Implements the heuristic from the HNSW paper: a candidate is only selected if it
    /// is closer to the query than to any already-selected neighbor. This promotes
    /// graph connectivity by preferring diverse directions over purely closest neighbors.
    /// Falls back to closest neighbors if heuristic yields fewer than m_val results.
    fn selectNeighborsHeuristic(
        self: *const HnswIndex,
        allocator: std.mem.Allocator,
        temp_allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        candidates: *std.AutoHashMapUnmanaged(u32, f32),
        m_val: usize,
    ) ![]u32 {

        // Sort candidates by distance (ascending) — temp allocation
        const CandidatePair = struct { id: u32, dist: f32 };
        var sorted = std.ArrayListUnmanaged(CandidatePair).empty;
        defer sorted.deinit(temp_allocator);

        var it = candidates.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.* != node_id) { // Don't include self
                try sorted.append(temp_allocator, .{ .id = entry.key_ptr.*, .dist = entry.value_ptr.* });
            }
        }

        // Sort by distance (closest first)
        std.sort.heap(CandidatePair, sorted.items, {}, struct {
            fn lessThan(_: void, a: CandidatePair, b: CandidatePair) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        // Select using heuristic: prefer diverse neighbors over purely closest
        var selected = std.ArrayListUnmanaged(u32).empty;
        errdefer selected.deinit(allocator);

        for (sorted.items, 0..) |candidate, idx| {
            if (selected.items.len >= m_val) break;

            // Prefetch next candidate's vector for the next iteration
            if (idx + 1 < sorted.items.len) {
                const next_id = sorted.items[idx + 1].id;
                if (next_id < records.len) {
                    @prefetch(records[next_id].vector.ptr, .{
                        .locality = 3, // High temporal locality - will be used soon
                        .rw = .read,
                        .cache = .data,
                    });
                }
            }

            // Check if this candidate is closer to node than to any selected neighbor
            var should_add = true;
            for (selected.items) |existing| {
                // Use pre-computed norms when available
                const dist_to_existing = if (self.norms.len > candidate.id and self.norms.len > existing) blk: {
                    const na = self.norms[candidate.id];
                    const nb = self.norms[existing];
                    if (na > 0.0 and nb > 0.0) {
                        const dot = simd.vectorDot(
                            records[candidate.id].vector,
                            records[existing].vector,
                        );
                        break :blk 1.0 - dot / (na * nb);
                    }
                    break :blk 1.0;
                } else 1.0 - simd.cosineSimilarity(
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

    /// Compute cosine distance between query and vector using SIMD.
    /// Uses pre-computed query norm to avoid redundant sqrt operations.
    /// Returns 1.0 for zero-length vectors (maximum distance).
    inline fn computeDistance(self: *const HnswIndex, query: []const f32, query_norm: f32, vector: []const f32) f32 {
        // Use cache if available (for node-to-node distances, not query distances)
        _ = self;

        // Optimized cosine distance with pre-computed query norm
        const vec_norm = simd.vectorL2Norm(vector);
        if (vec_norm == 0.0) return 1.0;

        const dot = simd.vectorDot(query, vector);
        return 1.0 - (dot / (query_norm * vec_norm));
    }

    /// Compute node-to-node distance with caching support.
    /// Uses pre-computed norms to avoid redundant L2 norm computation.
    /// Checks distance cache first; on miss, computes cosine distance via
    /// vectorDot + cached norms (saves ~50% FLOPs vs full cosineSimilarity).
    fn computeNodeDistance(self: *const HnswIndex, records: []const index_mod.VectorRecordView, a: u32, b: u32) f32 {
        // Check cache first
        if (self.distance_cache) |cache| {
            if (cache.get(a, b)) |cached| {
                return cached;
            }
        }

        // Compute cosine distance — use pre-computed norms when available
        const dist = if (self.norms.len > a and self.norms.len > b) blk: {
            const na = self.norms[a];
            const nb = self.norms[b];
            if (na > 0.0 and nb > 0.0) {
                const dot = simd.vectorDot(records[a].vector, records[b].vector);
                break :blk 1.0 - dot / (na * nb);
            }
            break :blk 1.0;
        } else 1.0 - simd.cosineSimilarity(records[a].vector, records[b].vector);

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

        // Clean up GPU accelerator
        if (self.gpu_accelerator) |accel| {
            accel.deinit();
            allocator.destroy(accel);
        }

        // Free pre-computed norms
        if (self.norms.len > 0) allocator.free(self.norms);

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
    ///
    /// @return Struct with hits, misses, and computed hit_rate (0.0-1.0), or null if caching disabled
    pub const CacheStats = struct { hits: u64, misses: u64, hit_rate: f32 };

    pub fn getCacheStats(self: *const HnswIndex) ?CacheStats {
        if (self.distance_cache) |cache| {
            const stats = cache.getStats();
            return .{ .hits = stats.hits, .misses = stats.misses, .hit_rate = stats.hit_rate };
        }
        return null;
    }

    /// Get GPU acceleration statistics if GPU is enabled.
    ///
    /// @return GpuAccelStats with operation counts, timing, and throughput metrics, or null if GPU disabled
    pub fn getGpuStats(self: *const HnswIndex) ?gpu_accel.GpuAccelStats {
        if (self.gpu_accelerator) |accel| {
            return accel.getStats();
        }
        return null;
    }

    /// Enable search state pooling for deserialized indexes.
    /// Pre-allocates search states to avoid per-query allocations.
    ///
    /// @param pool_size Number of search states to pre-allocate (capped at 64)
    pub fn enableSearchPool(self: *HnswIndex, pool_size: usize) !void {
        if (self.state_pool != null) return; // Already enabled

        const pool = try self.allocator.create(SearchStatePool);
        errdefer self.allocator.destroy(pool);

        pool.* = try SearchStatePool.init(self.allocator, pool_size);
        self.state_pool = pool;
    }

    /// Enable distance caching for deserialized indexes.
    /// Caches frequently computed distances to reduce redundant calculations.
    ///
    /// @param capacity Maximum number of distance pairs to cache
    pub fn enableDistanceCache(self: *HnswIndex, capacity: usize) !void {
        if (self.distance_cache != null) return; // Already enabled

        const cache = try self.allocator.create(DistanceCache);
        errdefer self.allocator.destroy(cache);

        cache.* = try DistanceCache.init(self.allocator, capacity);
        self.distance_cache = cache;
    }

    /// Result type for batch search operations
    pub const BatchSearchResult = struct {
        /// Query index in the original batch
        query_index: usize,
        /// Search results for this query
        results: []index_mod.IndexResult,
    };

    /// Worker context for parallel batch search
    const BatchSearchWorkerContext = struct {
        index: *const HnswIndex,
        records: []const index_mod.VectorRecordView,
        queries: []const []const f32,
        top_k: usize,
        results: []?[]index_mod.IndexResult,
        allocator: std.mem.Allocator,
        errors_occurred: *std.atomic.Value(usize),
        /// Current index to process (shared counter for work distribution)
        next_index: *std.atomic.Value(usize),
        /// Total number of queries
        total_queries: usize,
    };

    /// Perform batch search on multiple query vectors in parallel.
    /// Uses atomic counter-based work distribution for parallel execution.
    ///
    /// @param allocator Memory allocator for results
    /// @param records Source vector records
    /// @param queries Slice of query vectors
    /// @param top_k Number of results to return per query
    /// @return Array of BatchSearchResult, one per query. Caller must free both
    ///         the outer array and each inner results slice.
    pub fn batchSearch(
        self: *const HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        queries: []const []const f32,
        top_k: usize,
    ) ![]BatchSearchResult {
        if (queries.len == 0) {
            return allocator.alloc(BatchSearchResult, 0);
        }

        // For small batches, just run sequentially
        if (queries.len < 4) {
            return self.batchSearchSequential(allocator, records, queries, top_k);
        }

        // Allocate results array (nullable to track completion)
        const results_slots = try allocator.alloc(?[]index_mod.IndexResult, queries.len);
        defer allocator.free(results_slots);
        @memset(results_slots, null);

        // Shared state for work distribution
        var next_index = std.atomic.Value(usize).init(0);
        var errors_occurred = std.atomic.Value(usize).init(0);

        // Determine number of worker threads
        const cpu_count = std.Thread.getCpuCount() catch 4;
        const num_workers = @min(cpu_count, queries.len);

        // Create worker context
        var context = BatchSearchWorkerContext{
            .index = self,
            .records = records,
            .queries = queries,
            .top_k = top_k,
            .results = results_slots,
            .allocator = allocator,
            .errors_occurred = &errors_occurred,
            .next_index = &next_index,
            .total_queries = queries.len,
        };

        // Spawn worker threads
        var threads = try allocator.alloc(std.Thread, num_workers);
        defer allocator.free(threads);

        for (0..num_workers) |i| {
            threads[i] = std.Thread.spawn(.{}, batchSearchWorker, .{&context}) catch |err| {
                // If we can't spawn a thread, clean up what we have and fallback
                for (0..i) |j| {
                    threads[j].join();
                }
                // Free any results that were allocated
                for (results_slots) |maybe_result| {
                    if (maybe_result) |result| {
                        allocator.free(result);
                    }
                }
                return err;
            };
        }

        // Wait for all workers to complete
        for (threads) |t| {
            t.join();
        }

        // Check for errors
        if (errors_occurred.load(.acquire) > 0) {
            // Free any results that were allocated
            for (results_slots) |maybe_result| {
                if (maybe_result) |result| {
                    allocator.free(result);
                }
            }
            return error.BatchSearchFailed;
        }

        // Convert to final result format
        var final_results = try allocator.alloc(BatchSearchResult, queries.len);
        errdefer allocator.free(final_results);

        for (0..queries.len) |i| {
            final_results[i] = .{
                .query_index = i,
                .results = results_slots[i] orelse &.{},
            };
        }

        return final_results;
    }

    /// Worker function for parallel batch search
    fn batchSearchWorker(context: *BatchSearchWorkerContext) void {
        // Process work items using atomic counter for work distribution
        while (true) {
            // Atomically claim the next index
            const idx = context.next_index.fetchAdd(1, .monotonic);
            if (idx >= context.total_queries) {
                // No more work
                break;
            }
            processQueryIndex(context, idx);
        }
    }

    /// Process a single query in the batch
    fn processQueryIndex(context: *BatchSearchWorkerContext, query_idx: usize) void {
        const query = context.queries[query_idx];

        // Perform the search
        const search_result = context.index.search(
            context.allocator,
            context.records,
            query,
            context.top_k,
        ) catch {
            // Record error and set empty result to prevent uninitialized access
            _ = context.errors_occurred.fetchAdd(1, .monotonic);
            context.results[query_idx] = &.{};
            return;
        };

        // Store result
        context.results[query_idx] = search_result;
    }

    /// Sequential batch search for small batches
    fn batchSearchSequential(
        self: *const HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        queries: []const []const f32,
        top_k: usize,
    ) ![]BatchSearchResult {
        var results = try allocator.alloc(BatchSearchResult, queries.len);
        errdefer {
            for (results) |r| {
                allocator.free(r.results);
            }
            allocator.free(results);
        }

        for (queries, 0..) |query, i| {
            const search_results = try self.search(allocator, records, query, top_k);
            results[i] = .{
                .query_index = i,
                .results = search_results,
            };
        }

        return results;
    }

    /// Free batch search results
    pub fn freeBatchSearchResults(allocator: std.mem.Allocator, results: []BatchSearchResult) void {
        for (results) |r| {
            allocator.free(r.results);
        }
        allocator.free(results);
    }

    /// Check if GPU acceleration is available for this index.
    pub fn hasGpuAcceleration(self: *const HnswIndex) bool {
        if (self.gpu_accelerator) |accel| {
            return accel.isGpuAvailable();
        }
        return false;
    }

    /// Compute batch distances using GPU acceleration when available.
    /// Falls back to SIMD when GPU is unavailable or batch is too small.
    fn computeBatchDistances(
        self: *const HnswIndex,
        query: []const f32,
        query_norm: f32,
        records: []const index_mod.VectorRecordView,
        neighbor_ids: []const u32,
        distances: []f32,
    ) void {
        std.debug.assert(neighbor_ids.len == distances.len);

        if (neighbor_ids.len == 0) return;

        // Try GPU acceleration for large batches
        if (self.gpu_accelerator) |accel| {
            // Build vector slice array for batch computation
            var vectors = self.allocator.alloc([]const f32, neighbor_ids.len) catch {
                // Allocation failed, fall back to sequential
                self.computeBatchDistancesSequential(query, query_norm, records, neighbor_ids, distances);
                return;
            };
            defer self.allocator.free(vectors);

            for (neighbor_ids, 0..) |id, i| {
                if (id < records.len) {
                    vectors[i] = records[id].vector;
                } else {
                    // Invalid ID, use empty slice (will result in 0 similarity)
                    vectors[i] = &[_]f32{};
                }
            }

            // Use GPU accelerator's batch cosine similarity
            accel.batchCosineSimilarity(query, query_norm, vectors, distances) catch {
                // GPU failed, fall back to sequential
                self.computeBatchDistancesSequential(query, query_norm, records, neighbor_ids, distances);
                return;
            };

            // Convert similarities to distances
            for (distances) |*d| {
                d.* = 1.0 - d.*;
            }
        } else {
            // No GPU, use sequential SIMD computation
            self.computeBatchDistancesSequential(query, query_norm, records, neighbor_ids, distances);
        }
    }

    /// Sequential distance computation using SIMD.
    pub fn computeBatchDistancesSequential(
        self: *const HnswIndex,
        query: []const f32,
        query_norm: f32,
        records: []const index_mod.VectorRecordView,
        neighbor_ids: []const u32,
        distances: []f32,
    ) void {
        _ = self;

        // Prefetch first few vectors to warm the cache
        const prefetch_ahead: usize = 4;
        for (0..@min(prefetch_ahead, neighbor_ids.len)) |i| {
            const id = neighbor_ids[i];
            if (id < records.len) {
                @prefetch(records[id].vector.ptr, .{
                    .locality = 2, // Moderate locality - used once per batch
                    .rw = .read,
                    .cache = .data,
                });
            }
        }

        for (neighbor_ids, 0..) |id, i| {
            // Prefetch ahead for upcoming iterations
            if (i + prefetch_ahead < neighbor_ids.len) {
                const future_id = neighbor_ids[i + prefetch_ahead];
                if (future_id < records.len) {
                    @prefetch(records[future_id].vector.ptr, .{
                        .locality = 2,
                        .rw = .read,
                        .cache = .data,
                    });
                }
            }

            if (id < records.len) {
                const vec = records[id].vector;
                const vec_norm = simd.vectorL2Norm(vec);
                if (vec_norm > 0 and query_norm > 0) {
                    const dot = simd.vectorDot(query, vec);
                    distances[i] = 1.0 - (dot / (query_norm * vec_norm));
                } else {
                    distances[i] = 1.0;
                }
            } else {
                distances[i] = 1.0;
            }
        }
    }

    /// Save HNSW structure to a binary writer.
    ///
    /// Binary format (little-endian):
    /// - u32: node count
    /// - u32: m (max neighbors per node)
    /// - u32: entry point node ID (0 if none)
    /// - i32: max layer
    /// - u32: ef_construction
    /// - For each node:
    ///   - u32: layer count
    ///   - For each layer:
    ///     - u32: neighbor count
    ///     - u32[]: neighbor node IDs
    ///
    /// Note: SearchStatePool, DistanceCache, and GPU accelerator are not persisted.
    /// Use enableGpuAcceleration() after loading to restore GPU support.
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
    ///
    /// Reads the binary format written by save(). The loaded index does not include
    /// SearchStatePool, DistanceCache, or GPU accelerator - these can be enabled
    /// after loading via enableGpuAcceleration().
    ///
    /// @param allocator Memory allocator for graph structures
    /// @param reader Binary reader implementing readInt()
    /// @return Loaded HnswIndex ready for searching
    pub fn load(allocator: std.mem.Allocator, reader: anytype) !HnswIndex {
        const node_count = try reader.readInt(u32, .little);
        const m = try reader.readInt(u32, .little);
        const entry_point = try reader.readInt(u32, .little);
        const max_layer = try reader.readInt(i32, .little);
        const ef_construction = try reader.readInt(u32, .little);

        const self = HnswIndex{
            .m = m,
            .m_max = m,
            .m_max0 = m * 2,
            .ef_construction = ef_construction,
            .entry_point = if (node_count > 0) entry_point else null,
            .max_layer = max_layer,
            .nodes = try allocator.alloc(NodeLayers, node_count),
            .state_pool = null, // Not persisted, can be added via enableSearchPool
            .distance_cache = null, // Not persisted, can be added via enableDistanceCache
            .gpu_accelerator = null, // Not persisted, can be added via enableGpuAcceleration
            .norms = &.{}, // Norms not persisted; computeNodeDistance falls back to full cosine
            .allocator = allocator,
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

    /// Enable GPU acceleration on a loaded index.
    ///
    /// Call this after loading an index from disk to enable GPU-accelerated
    /// batch distance computation. If GPU hardware is unavailable, falls back
    /// to SIMD acceleration.
    ///
    /// @param batch_threshold Minimum number of vectors to trigger GPU path
    /// @return error.GpuDisabled if GPU feature was disabled at compile time
    pub fn enableGpuAcceleration(self: *HnswIndex, batch_threshold: usize) !void {
        if (self.gpu_accelerator != null) return; // Already enabled

        if (!build_options.enable_gpu) return error.GpuDisabled;

        const accel = try self.allocator.create(gpu_accel.GpuAccelerator);
        errdefer self.allocator.destroy(accel);

        accel.* = try gpu_accel.GpuAccelerator.init(self.allocator, .{
            .batch_threshold = batch_threshold,
        });
        self.gpu_accelerator = accel;
    }

    /// Get statistics about the HNSW index structure.
    pub fn getStats(self: *const HnswIndex) IndexStats {
        var total_degree: usize = 0;
        var total_connections: usize = 0;
        var memory: usize = 0;

        // Account for nodes array
        memory += self.nodes.len * @sizeOf(NodeLayers);
        // Account for norms array
        memory += self.norms.len * @sizeOf(f32);

        for (self.nodes) |node| {
            memory += node.layers.len * @sizeOf(index_mod.NeighborList);
            for (node.layers) |layer| {
                total_connections += 1;
                total_degree += layer.nodes.len;
                memory += layer.nodes.len * @sizeOf(u32);
            }
        }

        const avg_degree: f32 = if (total_connections > 0)
            @as(f32, @floatFromInt(total_degree)) / @as(f32, @floatFromInt(total_connections))
        else
            0.0;

        return .{
            .num_vectors = self.nodes.len,
            .num_layers = self.getLayerCount(),
            .avg_degree = avg_degree,
            .memory_bytes = memory,
            .entry_point_id = self.entry_point,
        };
    }

    /// Get the number of layers in the index.
    pub fn getLayerCount(self: *const HnswIndex) u32 {
        if (self.max_layer < 0) return 0;
        return @intCast(self.max_layer + 1);
    }

    /// Disable GPU acceleration and release associated resources.
    /// Subsequent searches will use SIMD-only distance computation.
    pub fn disableGpuAcceleration(self: *HnswIndex) void {
        if (self.gpu_accelerator) |accel| {
            accel.deinit();
            self.allocator.destroy(accel);
            self.gpu_accelerator = null;
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "IndexStats fields populated correctly" {
    const allocator = std.testing.allocator;

    // Build a small index to check stats
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 0, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 1, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 0.0, 1.0 } },
    };

    var idx = try HnswIndex.build(allocator, &records, 4, 16);
    defer idx.deinit(allocator);

    const stats = idx.getStats();
    try std.testing.expectEqual(@as(usize, 4), stats.num_vectors);
    try std.testing.expect(stats.num_layers >= 1);
    try std.testing.expect(stats.avg_degree > 0.0);
    try std.testing.expect(stats.memory_bytes > 0);
    try std.testing.expect(stats.entry_point_id != null);
}

test "AdaptiveSearchConfig defaults" {
    const config = AdaptiveSearchConfig{};
    try std.testing.expectEqual(@as(u32, 50), config.initial_ef);
    try std.testing.expectEqual(@as(u32, 500), config.max_ef);
    try std.testing.expectEqual(@as(u32, 50), config.ef_step);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), config.target_recall, 0.001);
    try std.testing.expectEqual(@as(u32, 5), config.max_iterations);
}

test "getLayerCount returns expected value" {
    const allocator = std.testing.allocator;

    const records = [_]index_mod.VectorRecordView{
        .{ .id = 0, .vector = &[_]f32{ 1.0, 0.0, 0.0 } },
        .{ .id = 1, .vector = &[_]f32{ 0.0, 1.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 0.0, 1.0 } },
    };

    var idx = try HnswIndex.build(allocator, &records, 4, 16);
    defer idx.deinit(allocator);

    // Layer count should be at least 1 (layer 0 always exists)
    const layer_count = idx.getLayerCount();
    try std.testing.expect(layer_count >= 1);
    try std.testing.expectEqual(layer_count, @as(u32, @intCast(idx.max_layer + 1)));
}

// Test discovery: pull in extracted test file and sub-modules
comptime {
    if (@import("builtin").is_test) {
        _ = @import("hnsw_test.zig");
        _ = search_state;
        _ = distance_cache;
    }
}
