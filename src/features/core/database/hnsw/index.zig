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
const simd = @import("../../../foundation/mod.zig").simd;
const index_mod = @import("../index.zig");
const gpu_accel = @import("../gpu_accel.zig");

const search_state = @import("../search_state.zig");
const distance_cache = @import("../distance_cache.zig");
const search_types = @import("search.zig");
const persistence = @import("persistence.zig");
const insert_impl = @import("insert.zig");
const search_impl = @import("search_impl.zig");

const SearchStatePool = search_state.SearchStatePool;
const SearchState = search_state.SearchState;
const DistanceCache = distance_cache.DistanceCache;
const AdaptiveSearchConfig = search_types.AdaptiveSearchConfig;
const IndexStats = search_types.IndexStats;

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

    // Compatibility fields for persistence and legacy engine
    node_levels: std.ArrayListUnmanaged(u32) = .empty,
    neighbors: std.ArrayListUnmanaged([][]u32) = .empty,
    vectors: std.ArrayListUnmanaged([]f32) = .empty,

    // Re-export BatchSearchResult inside struct for backward compatibility
    pub const BatchSearchResult = search_types.BatchSearchResult;

    /// Node layer structure — defined in insert_impl to avoid circular imports.
    pub const NodeLayers = insert_impl.NodeLayers;

    pub fn init(allocator: std.mem.Allocator, cfg: anytype, metric: anytype) !HnswIndex {
        _ = metric;
        const m = if (@hasField(@TypeOf(cfg), "hnsw")) cfg.hnsw.m else 16;
        const ef = if (@hasField(@TypeOf(cfg), "hnsw")) cfg.hnsw.ef_construction else 100;
        return initEmpty(allocator, .{
            .m = @intCast(m),
            .ef_construction = @intCast(ef),
        });
    }

    /// Configuration for HNSW index construction.
    pub const Config = struct {
        m: usize = 16,
        ef_construction: usize = 100,
        /// Number of pre-allocated search states (0 = disabled)
        search_pool_size: usize = 8,
        /// Distance cache capacity (0 = disabled)
        distance_cache_size: usize = 1024,
        /// Enable GPU acceleration for batch distance computation (requires -Dfeat-gpu)
        enable_gpu: bool = build_options.feat_gpu,
        /// Minimum batch size to trigger GPU acceleration
        gpu_batch_threshold: usize = 256,
    };

    /// Create an empty HNSW index for lazy population.
    pub fn initEmpty(allocator: std.mem.Allocator, cfg: Config) HnswIndex {
        return .{
            .m = cfg.m,
            .m_max = cfg.m,
            .m_max0 = cfg.m * 2,
            .ef_construction = cfg.ef_construction,
            .entry_point = null,
            .max_layer = -1,
            .nodes = &.{},
            .state_pool = null,
            .distance_cache = null,
            .gpu_accelerator = null,
            .norms = &.{},
            .allocator = allocator,
        };
    }

    /// Build a new HNSW index from a set of records.
    pub fn build(
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        m: usize,
        ef_construction: usize,
    ) !HnswIndex {
        return buildWithConfig(allocator, records, .{
            .m = m,
            .ef_construction = ef_construction,
        });
    }

    /// Build a new HNSW index with full configuration options.
    pub fn buildWithConfig(
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        config: Config,
    ) !HnswIndex {
        if (records.len == 0) return index_mod.IndexError.EmptyIndex;

        // Initialize optional search state pool
        var state_pool_ptr: ?*SearchStatePool = null;
        if (config.search_pool_size > 0) {
            const pool = try allocator.create(SearchStatePool);
            pool.* = try SearchStatePool.init(allocator, config.search_pool_size);
            state_pool_ptr = pool;
        }
        errdefer if (state_pool_ptr) |pool| {
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
        if (config.enable_gpu and build_options.feat_gpu) {
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

        // Pre-compute L2 norms for all vectors
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
            .state_pool = state_pool_ptr,
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

        // Use arena allocator for per-insertion temporaries
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        for (records, 0..) |_, i| {
            // Delegate to extracted insert module
            try insert_impl.insertAt(
                .{
                    .nodes = self.nodes,
                    .m_max = self.m_max,
                    .m_max0 = self.m_max0,
                    .ef_construction = self.ef_construction,
                    .norms = self.norms,
                    .distance_cache = self.distance_cache,
                },
                &self.entry_point,
                &self.max_layer,
                allocator,
                arena.allocator(),
                records,
                @intCast(i),
                random,
                m_l,
            );
            _ = arena.reset(.retain_capacity);
        }

        return self;
    }

    /// Insert a new node into the HNSW graph.
    pub fn insert(self: *HnswIndex, vector: []const f32) !void {
        const cloned = try self.allocator.dupe(f32, vector);
        try self.vectors.append(self.allocator, cloned);
        // Note: Graph is not updated in this compatibility shim.
    }

    /// Search the HNSW graph for the nearest neighbors of a query vector.
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
        var curr_dist = search_impl.computeDistance(query, query_norm, records[curr_node].vector);

        // 1. Zoom in through layers with prefetching
        var lc: i32 = self.max_layer;
        while (lc > 0) : (lc -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                const neighbors = self.nodes[curr_node].layers[@intCast(lc)].nodes;

                for (neighbors) |neighbor| {
                    if (neighbor < records.len) {
                        @prefetch(records[neighbor].vector.ptr, .{ .locality = 3, .rw = .read });
                    }
                }

                for (neighbors) |neighbor| {
                    const d = search_impl.computeDistance(query, query_norm, records[neighbor].vector);
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
        const max_candidates = @max(top_k * 2, self.ef_construction / 2);

        const gpu_batch_threshold: usize = if (self.gpu_accelerator) |accel|
            accel.config.batch_threshold
        else
            std.math.maxInt(usize);

        while (head < state.queue.items.len and state.queue.items.len < max_candidates) : (head += 1) {
            const u = state.queue.items[head];
            if (self.nodes[u].layers.len == 0) continue;

            const neighbors = self.nodes[u].layers[0].nodes;

            // Collect unvisited neighbor count
            var unvisited_count: usize = 0;
            for (neighbors) |v| {
                if (!state.visited.contains(v) and v < records.len) {
                    unvisited_count += 1;
                }
            }

            // Use GPU-accelerated batch distance when batch exceeds threshold
            if (unvisited_count >= gpu_batch_threshold and self.gpu_accelerator != null) {
                const unvisited_ids = allocator.alloc(u32, unvisited_count) catch {
                    search_impl.searchNeighborsSequential(allocator, neighbors, records, query, query_norm, state) catch |e| {
                        std.log.warn("HNSW search: GPU alloc failed and sequential fallback also failed: {}", .{e});
                    };
                    continue;
                };
                defer allocator.free(unvisited_ids);

                const unvisited_vecs = allocator.alloc([]const f32, unvisited_count) catch {
                    search_impl.searchNeighborsSequential(allocator, neighbors, records, query, query_norm, state) catch |e| {
                        std.log.warn("HNSW search: GPU vec alloc failed and sequential fallback also failed: {}", .{e});
                    };
                    continue;
                };
                defer allocator.free(unvisited_vecs);

                var idx: usize = 0;
                for (neighbors) |v| {
                    if (!state.visited.contains(v) and v < records.len) {
                        unvisited_ids[idx] = v;
                        unvisited_vecs[idx] = records[v].vector;
                        idx += 1;
                    }
                }

                const batch_distances = allocator.alloc(f32, unvisited_count) catch {
                    search_impl.searchNeighborsSequential(allocator, neighbors, records, query, query_norm, state) catch |e| {
                        std.log.warn("HNSW search: GPU dist alloc failed and sequential fallback also failed: {}", .{e});
                    };
                    continue;
                };
                defer allocator.free(batch_distances);

                if (self.gpu_accelerator) |accel| {
                    accel.batchCosineSimilarity(query, query_norm, unvisited_vecs, batch_distances) catch {
                        search_impl.searchNeighborsSequential(allocator, neighbors, records, query, query_norm, state) catch |e| {
                            std.log.warn("HNSW search: GPU batch cosine failed and sequential fallback also failed: {}", .{e});
                        };
                        continue;
                    };

                    for (unvisited_ids, 0..) |v, vi| {
                        try state.visited.put(allocator, v, {});
                        const d = 1.0 - batch_distances[vi];
                        try state.candidates.put(allocator, v, d);
                        try state.queue.append(allocator, v);
                    }
                }
            } else {
                // Sequential path: prefetch + compute one by one
                for (neighbors) |v| {
                    if (!state.visited.contains(v) and v < records.len) {
                        @prefetch(records[v].vector.ptr, .{ .locality = 2, .rw = .read });
                    }
                }

                for (neighbors) |v| {
                    if (!state.visited.contains(v)) {
                        try state.visited.put(allocator, v, {});
                        const d = search_impl.computeDistance(query, query_norm, records[v].vector);
                        try state.candidates.put(allocator, v, d);
                        try state.queue.append(allocator, v);
                    }
                }
            }
        }

        // 3. Extract and sort top-k results
        const result_count = state.candidates.count();
        var results = try allocator.alloc(index_mod.IndexResult, result_count);
        errdefer allocator.free(results);

        var cit = state.candidates.iterator();
        var i: usize = 0;
        while (cit.next()) |entry_item| {
            results[i] = .{
                .id = records[entry_item.key_ptr.*].id,
                .score = 1.0 - entry_item.value_ptr.*,
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

    /// Compute node-to-node distance with caching support.
    fn computeNodeDistance(self: *const HnswIndex, records: []const index_mod.VectorRecordView, a: u32, b: u32) f32 {
        return insert_impl.computeNodeDistance(
            .{
                .nodes = self.nodes,
                .m_max = self.m_max,
                .m_max0 = self.m_max0,
                .ef_construction = self.ef_construction,
                .norms = self.norms,
                .distance_cache = self.distance_cache,
            },
            records,
            a,
            b,
        );
    }

    /// Free resources associated with the index.
    pub fn deinit(self: *HnswIndex, allocator: std.mem.Allocator) void {
        if (self.state_pool) |pool| {
            pool.deinit();
            allocator.destroy(pool);
        }

        if (self.distance_cache) |cache| {
            cache.deinit(allocator);
            allocator.destroy(cache);
        }

        if (self.gpu_accelerator) |accel| {
            accel.deinit();
            allocator.destroy(accel);
        }

        if (self.norms.len > 0) allocator.free(self.norms);

        for (self.nodes) |node| {
            for (node.layers) |list| {
                allocator.free(list.nodes);
            }
            allocator.free(node.layers);
        }
        allocator.free(self.nodes);

        for (self.vectors.items) |v| allocator.free(v);
        self.vectors.deinit(allocator);
        self.node_levels.deinit(allocator);
        for (self.neighbors.items) |layer_list| {
            for (layer_list) |nbrs| allocator.free(nbrs);
            allocator.free(layer_list);
        }
        self.neighbors.deinit(allocator);

        self.* = undefined;
    }

    /// Get cache statistics if distance caching is enabled.
    pub const CacheStats = struct { hits: u64, misses: u64, hit_rate: f32 };

    pub fn getCacheStats(self: *const HnswIndex) ?CacheStats {
        if (self.distance_cache) |cache| {
            const stats = cache.getStats();
            return .{ .hits = stats.hits, .misses = stats.misses, .hit_rate = stats.hit_rate };
        }
        return null;
    }

    /// Get GPU acceleration statistics if GPU is enabled.
    pub fn getGpuStats(self: *const HnswIndex) ?gpu_accel.GpuAccelStats {
        if (self.gpu_accelerator) |accel| {
            return accel.getStats();
        }
        return null;
    }

    /// Enable search state pooling for deserialized indexes.
    pub fn enableSearchPool(self: *HnswIndex, pool_size: usize) !void {
        if (self.state_pool != null) return;

        const pool = try self.allocator.create(SearchStatePool);
        errdefer self.allocator.destroy(pool);

        pool.* = try SearchStatePool.init(self.allocator, pool_size);
        self.state_pool = pool;
    }

    /// Enable distance caching for deserialized indexes.
    pub fn enableDistanceCache(self: *HnswIndex, capacity: usize) !void {
        if (self.distance_cache != null) return;

        const cache = try self.allocator.create(DistanceCache);
        errdefer self.allocator.destroy(cache);

        cache.* = try DistanceCache.init(self.allocator, capacity);
        self.distance_cache = cache;
    }

    /// Perform batch search on multiple query vectors in parallel.
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

        // Allocate results array
        const results_slots = try allocator.alloc(?[]index_mod.IndexResult, queries.len);
        defer allocator.free(results_slots);
        @memset(results_slots, null);

        var next_index = std.atomic.Value(usize).init(0);
        var errors_occurred = std.atomic.Value(usize).init(0);

        const cpu_count = std.Thread.getCpuCount() catch 4;
        const num_workers = @min(cpu_count, queries.len);

        const Context = struct {
            index: *const HnswIndex,
            records: []const index_mod.VectorRecordView,
            queries: []const []const f32,
            top_k: usize,
            results: []?[]index_mod.IndexResult,
            alloc: std.mem.Allocator,
            errors: *std.atomic.Value(usize),
            next: *std.atomic.Value(usize),
            total: usize,
        };

        var context = Context{
            .index = self,
            .records = records,
            .queries = queries,
            .top_k = top_k,
            .results = results_slots,
            .alloc = allocator,
            .errors = &errors_occurred,
            .next = &next_index,
            .total = queries.len,
        };

        var threads = try allocator.alloc(std.Thread, num_workers);
        defer allocator.free(threads);

        const worker = struct {
            fn run(ctx: *Context) void {
                while (true) {
                    const idx = ctx.next.fetchAdd(1, .monotonic);
                    if (idx >= ctx.total) break;

                    const search_result = ctx.index.search(
                        ctx.alloc,
                        ctx.records,
                        ctx.queries[idx],
                        ctx.top_k,
                    ) catch {
                        _ = ctx.errors.fetchAdd(1, .monotonic);
                        ctx.results[idx] = &.{};
                        return;
                    };

                    ctx.results[idx] = search_result;
                }
            }
        }.run;

        for (0..num_workers) |wi| {
            threads[wi] = std.Thread.spawn(.{}, worker, .{&context}) catch |err| {
                for (0..wi) |j| {
                    threads[j].join();
                }
                for (results_slots) |maybe_result| {
                    if (maybe_result) |result| {
                        allocator.free(result);
                    }
                }
                return err;
            };
        }

        for (threads) |t| {
            t.join();
        }

        if (errors_occurred.load(.acquire) > 0) {
            for (results_slots) |maybe_result| {
                if (maybe_result) |result| {
                    allocator.free(result);
                }
            }
            return error.BatchSearchFailed;
        }

        var final_results = try allocator.alloc(BatchSearchResult, queries.len);
        errdefer allocator.free(final_results);

        for (0..queries.len) |qi| {
            final_results[qi] = .{
                .query_index = qi,
                .results = results_slots[qi] orelse &.{},
            };
        }

        return final_results;
    }

    /// Sequential batch search for small batches.
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

        for (queries, 0..) |query, qi| {
            const search_results = try self.search(allocator, records, query, top_k);
            results[qi] = .{
                .query_index = qi,
                .results = search_results,
            };
        }

        return results;
    }

    /// Free batch search results.
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

        if (self.gpu_accelerator) |accel| {
            var vectors = self.allocator.alloc([]const f32, neighbor_ids.len) catch {
                search_impl.computeBatchDistancesSequential(query, query_norm, records, neighbor_ids, distances);
                return;
            };
            defer self.allocator.free(vectors);

            for (neighbor_ids, 0..) |id, ni| {
                if (id < records.len) {
                    vectors[ni] = records[id].vector;
                } else {
                    vectors[ni] = &[_]f32{};
                }
            }

            accel.batchCosineSimilarity(query, query_norm, vectors, distances) catch {
                search_impl.computeBatchDistancesSequential(query, query_norm, records, neighbor_ids, distances);
                return;
            };

            for (distances) |*d| {
                d.* = 1.0 - d.*;
            }
        } else {
            search_impl.computeBatchDistancesSequential(query, query_norm, records, neighbor_ids, distances);
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
        search_impl.computeBatchDistancesSequential(query, query_norm, records, neighbor_ids, distances);
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

        const self = HnswIndex{
            .m = m,
            .m_max = m,
            .m_max0 = m * 2,
            .ef_construction = ef_construction,
            .entry_point = if (node_count > 0) entry_point else null,
            .max_layer = max_layer,
            .nodes = try allocator.alloc(NodeLayers, node_count),
            .state_pool = null,
            .distance_cache = null,
            .gpu_accelerator = null,
            .norms = &.{},
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
    pub fn enableGpuAcceleration(self: *HnswIndex, batch_threshold: usize) !void {
        if (self.gpu_accelerator != null) return;

        if (!build_options.feat_gpu) return error.GpuDisabled;

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

        memory += self.nodes.len * @sizeOf(NodeLayers);
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

test "getLayerCount returns expected value" {
    const allocator = std.testing.allocator;

    const records = [_]index_mod.VectorRecordView{
        .{ .id = 0, .vector = &[_]f32{ 1.0, 0.0, 0.0 } },
        .{ .id = 1, .vector = &[_]f32{ 0.0, 1.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 0.0, 1.0 } },
    };

    var idx = try HnswIndex.build(allocator, &records, 4, 16);
    defer idx.deinit(allocator);

    const layer_count = idx.getLayerCount();
    try std.testing.expect(layer_count >= 1);
    try std.testing.expectEqual(layer_count, @as(u32, @intCast(idx.max_layer + 1)));
}

test "HNSW search results identical with and without GPU accelerator" {
    const allocator = std.testing.allocator;

    const records = [_]index_mod.VectorRecordView{
        .{ .id = 0, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 1, .vector = &[_]f32{ 0.9, 0.1, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
        .{ .id = 4, .vector = &[_]f32{ 0.0, 0.0, 0.0, 1.0 } },
        .{ .id = 5, .vector = &[_]f32{ 0.5, 0.5, 0.0, 0.0 } },
    };

    var idx_no_gpu = try HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 4,
        .ef_construction = 16,
        .enable_gpu = false,
        .search_pool_size = 0,
        .distance_cache_size = 0,
    });
    defer idx_no_gpu.deinit(allocator);

    var idx_with_gpu = try HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 4,
        .ef_construction = 16,
        .enable_gpu = true,
        .gpu_batch_threshold = 2,
        .search_pool_size = 0,
        .distance_cache_size = 0,
    });
    defer idx_with_gpu.deinit(allocator);

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    const results_no_gpu = try idx_no_gpu.search(allocator, &records, &query, 3);
    defer allocator.free(results_no_gpu);

    const results_with_gpu = try idx_with_gpu.search(allocator, &records, &query, 3);
    defer allocator.free(results_with_gpu);

    try std.testing.expectEqual(results_no_gpu.len, results_with_gpu.len);

    if (results_no_gpu.len > 0 and results_with_gpu.len > 0) {
        try std.testing.expectEqual(results_no_gpu[0].id, results_with_gpu[0].id);
        try std.testing.expectApproxEqAbs(results_no_gpu[0].score, results_with_gpu[0].score, 0.01);
    }
}

test "HNSW GPU batch threshold behavior" {
    const allocator = std.testing.allocator;

    const records = [_]index_mod.VectorRecordView{
        .{ .id = 0, .vector = &[_]f32{ 1.0, 0.0, 0.0 } },
        .{ .id = 1, .vector = &[_]f32{ 0.0, 1.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 0.0, 1.0 } },
    };

    var idx = try HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 4,
        .ef_construction = 16,
        .enable_gpu = true,
        .gpu_batch_threshold = 10000,
        .search_pool_size = 0,
        .distance_cache_size = 0,
    });
    defer idx.deinit(allocator);

    const query = [_]f32{ 1.0, 0.0, 0.0 };

    const results = try idx.search(allocator, &records, &query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
    try std.testing.expectEqual(@as(u32, 0), results[0].id);

    if (idx.gpu_accelerator) |accel| {
        const stats = accel.getStats();
        try std.testing.expectEqual(@as(u64, 0), stats.gpu_ops);
    }
}

// Test discovery: pull in extracted test file and sub-modules
comptime {
    if (@import("builtin").is_test) {
        _ = @import("../hnsw_test.zig");
        _ = search_state;
        _ = distance_cache;
        _ = search_types;
        _ = persistence;
        _ = insert_impl;
        _ = search_impl;
    }
}

test {
    std.testing.refAllDecls(@This());
}
